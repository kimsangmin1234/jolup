"""전체 모델 조립: 뉴스 주도형 멀티모달 주가 예측 모델 (Ⅱ. 제안하는 모델)

    뉴스 임베딩(1536) + 감성(1)
        └─[1] NewsSignalEncoder ─────────── 주 신호 (B, 512) ──────────┐
                                                                      │
    기술적 지표 (B, 30, 9)                                             │
        └─[2] TCNEncoder ── 보조 표현 (B, 30, 512) ─┐                  │
                                                    │                  │
                       [3] AsymmetricCrossAttention ┴── 보조 문맥 (B, 512)
                                                                      │
                       [4] GatedResidualFusion ────────────────────────┘
                                    │
                            통합 표현 (B, 512)
                                    │
                       [5] ReturnPredictor ── 등락률 (B,)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import ModelConfig
from modules.cross_attention import AsymmetricCrossAttention
from modules.gated_fusion import GatedResidualFusion
from modules.news_encoder import NewsSignalEncoder
from modules.return_predictor import ReturnPredictor
from modules.tcn_encoder import TCNEncoder


@dataclass
class ModelOutput:
    """예측값과 함께 해석에 필요한 중간 표현을 반환한다."""

    prediction: torch.Tensor       # (B,) 등락률
    primary_signal: torch.Tensor   # (B, 512) 주 신호 벡터
    aux_context: torch.Tensor      # (B, 512) 보조 문맥 벡터
    fused: torch.Tensor            # (B, 512) 통합 표현 벡터
    attention_weights: torch.Tensor  # (B, 30) 시점별 어텐션 가중치
    gate: torch.Tensor             # (B, 512) 차원별 게이트 값


class NewsDrivenStockPredictor(nn.Module):
    """논문에서 제안한 5개 모듈을 순차 결합한 전체 모델."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.news_encoder = NewsSignalEncoder(self.config.news)              # 모듈 1
        self.tcn_encoder = TCNEncoder(self.config.tcn)                       # 모듈 2
        self.cross_attention = AsymmetricCrossAttention(self.config.attention)  # 모듈 3
        self.fusion = GatedResidualFusion(self.config.fusion)                # 모듈 4
        self.predictor = ReturnPredictor(self.config.predictor)              # 모듈 5

    def forward(
        self,
        embedding: torch.Tensor,
        sentiment: torch.Tensor,
        indicators: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        Args:
            embedding:  (B, 1536) text-embedding-3-small 의미 임베딩
            sentiment:  (B,) 또는 (B, 1) GPT-4o-mini 감성 점수 (-1 ~ +1)
            indicators: (B, 30, 9) 정규화된 기술적 지표 시계열
            mask:       (B, 30) 선택적 시점 마스크

        Returns:
            ModelOutput
        """
        primary = self.news_encoder(embedding, sentiment)          # (B, 512)
        auxiliary = self.tcn_encoder(indicators)                   # (B, 30, 512)
        context, weights = self.cross_attention(primary, auxiliary, mask)
        fused, gate = self.fusion(primary, context)                # (B, 512)
        prediction = self.predictor(fused)                         # (B,)

        return ModelOutput(
            prediction=prediction,
            primary_signal=primary,
            aux_context=context,
            fused=fused,
            attention_weights=weights,
            gate=gate,
        )

    @torch.no_grad()
    def predict(
        self,
        embedding: torch.Tensor,
        sentiment: torch.Tensor,
        indicators: torch.Tensor,
    ) -> torch.Tensor:
        """추론 전용 경로. 등락률만 반환한다."""
        self.eval()
        return self.forward(embedding, sentiment, indicators).prediction

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )
