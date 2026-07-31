"""전역 설정.

논문 "뉴스 이벤트 기반 주식 가격 예측 모델"에서 명시한 차원/하이퍼파라미터를
한 곳에 모아 둔다. 각 모듈은 이 설정값을 인자로 받아 동작하며, 설정에 대한
의존성을 갖지 않는다(모듈 단독 사용 가능).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 논문에서 사용하는 9종 기술적 지표 (Ⅰ. 서론)
#   종가, 거래량, 5일/20일 이동평균, RSI, MACD, 볼린저 밴드 상·하한선, ATR
# --------------------------------------------------------------------------
TECHNICAL_INDICATORS: tuple[str, ...] = (
    "close",       # 종가
    "volume",      # 거래량
    "ma5",         # 5일 이동평균
    "ma20",        # 20일 이동평균
    "rsi",         # 상대강도지수
    "macd",        # MACD
    "bb_upper",    # 볼린저 밴드 상한선
    "bb_lower",    # 볼린저 밴드 하한선
    "atr",         # Average True Range
)


@dataclass
class NewsEncoderConfig:
    """모듈 1: LLM 기반 주 신호 생성."""

    llm_model: str = "gpt-4o-mini"                  # 요약 + 감성 점수 추출
    embedding_model: str = "text-embedding-3-small"  # 의미 임베딩
    embedding_dim: int = 1536                        # 의미 임베딩 차원
    sentiment_dim: int = 1                           # 감성 점수(스칼라)
    d_model: int = 512                               # 주 신호 벡터 차원
    dropout: float = 0.1

    @property
    def fusion_input_dim(self) -> int:
        """임베딩(1536) + 감성 점수(1) = 1537차원 결합 벡터."""
        return self.embedding_dim + self.sentiment_dim


@dataclass
class TCNConfig:
    """모듈 2: TCN 기반 보조 벡터 생성."""

    in_channels: int = len(TECHNICAL_INDICATORS)  # 9종 기술적 지표
    d_model: int = 512                            # 시점별 보조 표현 벡터 차원
    hidden_channels: int = 128                    # 중간 블록 채널 수
    kernel_size: int = 3
    num_blocks: int = 4                           # dilation 1, 2, 4, 8
    dropout: float = 0.1
    lookback: int = 30                            # 과거 30일


@dataclass
class CrossAttentionConfig:
    """모듈 3: 비대칭 교차 어텐션."""

    d_model: int = 512
    # 논문 본문은 "내적"만 명시한다. 512차원 내적은 값의 크기가 커져
    # softmax가 포화되기 쉬우므로 1/sqrt(d) 스케일링을 기본값으로 둔다.
    # 논문 표기를 문자 그대로 재현하려면 False로 설정한다.
    scaled: bool = True


@dataclass
class GatedFusionConfig:
    """모듈 4: 게이트 기반 잔차 결합."""

    d_model: int = 512
    # 게이트는 [주 신호 ; 보조 문맥] 을 입력받아 차원별 0~1 값을 학습한다.
    gate_hidden_dim: int | None = None  # None이면 단일 선형층


@dataclass
class PredictorConfig:
    """모듈 5: 등락률 예측 (MLP)."""

    d_model: int = 512
    hidden_dims: tuple[int, ...] = (256, 64)  # 512 -> 256 -> 64 -> 1
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """학습 설정."""

    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


@dataclass
class ModelConfig:
    """전체 모델 설정."""

    d_model: int = 512
    news: NewsEncoderConfig = field(default_factory=NewsEncoderConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)
    attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    fusion: GatedFusionConfig = field(default_factory=GatedFusionConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # 모든 모듈이 동일한 표현 차원(512)을 공유하도록 강제한다.
        self.news.d_model = self.d_model
        self.tcn.d_model = self.d_model
        self.attention.d_model = self.d_model
        self.fusion.d_model = self.d_model
        self.predictor.d_model = self.d_model
