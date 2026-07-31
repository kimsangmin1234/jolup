"""모듈별 형태(shape)·불변식 검증.

API 키 없이 임의 텐서만으로 전체 경로가 동작하는지 확인한다.
실행: ``python -m pytest tests/ -v`` 또는 ``python tests/test_modules.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ModelConfig, TECHNICAL_INDICATORS  # noqa: E402
from data.technical_indicators import (  # noqa: E402
    MinMaxScaler,
    average_true_range,
    compute_indicators,
    relative_strength_index,
    simple_moving_average,
)
from model import NewsDrivenStockPredictor  # noqa: E402
from modules import (  # noqa: E402
    AsymmetricCrossAttention,
    GatedResidualFusion,
    NewsSignalEncoder,
    ReturnPredictor,
    TCNEncoder,
)

B, T = 4, 30


def test_news_encoder_shape():
    """모듈 1: (1536 임베딩 + 1 감성) → 512 주 신호."""
    config = ModelConfig()
    encoder = NewsSignalEncoder(config.news)

    embedding = torch.randn(B, config.news.embedding_dim)
    sentiment = torch.empty(B).uniform_(-1, 1)

    out = encoder(embedding, sentiment)
    assert out.shape == (B, 512)

    # 결합 벡터 경로도 동일 결과여야 한다.
    encoder.eval()
    combined = torch.cat([embedding, sentiment.unsqueeze(-1)], dim=-1)
    assert combined.shape == (B, 1537)
    assert torch.allclose(encoder(embedding, sentiment),
                          encoder.forward_from_vector(combined), atol=1e-6)


def test_tcn_shape_and_causality():
    """모듈 2: (B, 30, 9) → (B, 30, 512), 그리고 미래 정보 차단 확인."""
    config = ModelConfig()
    encoder = TCNEncoder(config.tcn).eval()

    x = torch.randn(1, T, len(TECHNICAL_INDICATORS))
    out = encoder(x)
    assert out.shape == (1, T, 512)

    # 수용 영역이 30일을 덮는지
    assert encoder.receptive_field >= config.tcn.lookback

    # 인과성: 마지막 시점 입력을 바꿔도 그 이전 시점 출력은 불변이어야 한다.
    # (LayerNorm은 시점별로 독립이므로 인과성이 유지된다.)
    x2 = x.clone()
    x2[0, -1] = torch.randn(len(TECHNICAL_INDICATORS))
    out2 = encoder(x2)
    assert torch.allclose(out[0, :-1], out2[0, :-1], atol=1e-5), "미래 정보가 누수되었습니다."

    # 학습 모드에서도 인과성이 유지되어야 한다.
    # (Dropout은 끄고 정규화 계층의 통계 혼입만 확인한다.)
    encoder.train()
    for module in encoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    batch = torch.randn(8, T, len(TECHNICAL_INDICATORS))
    batch2 = batch.clone()
    batch2[:, -1] = torch.randn(8, len(TECHNICAL_INDICATORS))
    assert torch.allclose(
        encoder(batch)[:, :-1], encoder(batch2)[:, :-1], atol=1e-5
    ), "학습 모드에서 미래 정보가 누수되었습니다."


def test_cross_attention():
    """모듈 3: 어텐션 가중치 합 = 1, 문맥 벡터 512차원."""
    config = ModelConfig()
    attention = AsymmetricCrossAttention(config.attention)

    primary = torch.randn(B, 512)
    auxiliary = torch.randn(B, T, 512)

    context, weights = attention(primary, auxiliary)
    assert context.shape == (B, 512)
    assert weights.shape == (B, T)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (weights >= 0).all()

    # 마스크된 시점은 가중치 0
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, :10] = False
    _, masked_weights = attention(primary, auxiliary, mask)
    assert torch.allclose(masked_weights[:, :10], torch.zeros(B, 10), atol=1e-6)


def test_gated_fusion():
    """모듈 4: 게이트 0~1, 게이트=0이면 주 신호만 남는다."""
    config = ModelConfig()
    fusion = GatedResidualFusion(config.fusion)

    primary = torch.randn(B, 512)
    context = torch.randn(B, 512)

    fused, gate = fusion(primary, context)
    assert fused.shape == (B, 512)
    assert gate.shape == (B, 512)
    assert ((gate >= 0) & (gate <= 1)).all()

    # 보조 문맥이 0이면 잔차 결합 결과는 주 신호의 정규화 값과 같다.
    fused_zero, _ = fusion(primary, torch.zeros_like(context))
    assert torch.allclose(fused_zero, fusion.norm(primary), atol=1e-6)


def test_return_predictor():
    """모듈 5: 512 → 스칼라, 출력 활성함수 없음."""
    config = ModelConfig()
    predictor = ReturnPredictor(config.predictor)

    out = predictor(torch.randn(B, 512))
    assert out.shape == (B,)

    # 마지막 층이 활성함수 없는 Linear인지 확인
    assert isinstance(predictor.mlp[-1], torch.nn.Linear)
    assert predictor.mlp[-1].out_features == 1

    # 큰 입력에서 음수/양수 모두 나올 수 있어야 한다(포화 없음).
    big = predictor(torch.randn(256, 512) * 10)
    assert big.min() < 0 < big.max()


def test_full_model():
    """전체 모델: 순전파 + 역전파."""
    config = ModelConfig()
    model = NewsDrivenStockPredictor(config)

    embedding = torch.randn(B, 1536)
    sentiment = torch.empty(B).uniform_(-1, 1)
    indicators = torch.rand(B, T, 9)
    label = torch.randn(B) * 0.02

    out = model(embedding, sentiment, indicators)
    assert out.prediction.shape == (B,)
    assert out.fused.shape == (B, 512)
    assert out.attention_weights.shape == (B, T)

    loss = torch.nn.functional.mse_loss(out.prediction, label)
    loss.backward()

    # 모든 모듈에 실제로 기울기가 흐르는지 확인
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name}에 기울기가 없습니다."

    print(f"학습 가능 파라미터: {model.num_parameters():,}")


def test_technical_indicators():
    """지표 계산: 형태와 알려진 값 검증."""
    rng = np.random.default_rng(0)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    volume = rng.uniform(1e5, 1e6, n)

    matrix = compute_indicators(high, low, close, volume)
    assert matrix.shape == (n, len(TECHNICAL_INDICATORS))
    assert np.isfinite(matrix).all()

    # 이동평균: 창 구간의 평균과 일치
    ma5 = simple_moving_average(close, 5)
    assert np.isclose(ma5[10], close[6:11].mean())

    # RSI는 0~100 범위
    rsi = relative_strength_index(close)
    assert ((rsi >= 0) & (rsi <= 100)).all()

    # ATR은 항상 양수
    assert (average_true_range(high, low, close) > 0).all()

    # 볼린저 상한선 >= 하한선
    upper_idx = TECHNICAL_INDICATORS.index("bb_upper")
    lower_idx = TECHNICAL_INDICATORS.index("bb_lower")
    assert (matrix[:, upper_idx] >= matrix[:, lower_idx]).all()


def test_scaler_uses_train_statistics_only():
    """정규화: 학습 구간 통계만 사용하고 평가 구간은 [0,1]로 클리핑."""
    train = np.array([[0.0, 10.0], [10.0, 20.0]])
    test = np.array([[-5.0, 30.0], [5.0, 15.0]])

    scaler = MinMaxScaler().fit(train)
    scaled_train = scaler.transform(train)
    assert np.allclose(scaled_train.min(axis=0), 0.0)
    assert np.allclose(scaled_train.max(axis=0), 1.0)

    scaled_test = scaler.transform(test)
    assert ((scaled_test >= 0) & (scaled_test <= 1)).all()
    assert np.isclose(scaled_test[1, 0], 0.5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\n모든 검증을 통과했습니다.")
