"""모듈 3. 비대칭 교차 어텐션 (Ⅱ.3)

뉴스 기반 주 신호 벡터를 **질의(Query)** 로, 30일간의 보조 표현 벡터를
**키(Key)/값(Value)** 로 설정하여, 뉴스 표현이 과거 시장 상태를 선택적으로
참조하도록 한다. 질의가 한 개(뉴스)이고 키/값이 30개(시점)인 비대칭 구조이다.

    score_t = <q, k_t>                    (t = 1..30)
    a       = softmax([score_1, ..., score_30])
    c       = Σ_t a_t · v_t               (512차원 보조 문맥 벡터)

논문 본문은 관련도를 "내적"으로 기술하지만, 512차원 내적은 값의 크기가 커져
softmax가 포화되기 쉽다. 기본값은 1/sqrt(d) 스케일링을 적용하며
(``CrossAttentionConfig.scaled``), False로 두면 논문 표기 그대로 순수 내적이 된다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import CrossAttentionConfig


class AsymmetricCrossAttention(nn.Module):
    """주 신호(질의) × 보조 표현(키/값) → 512차원 보조 문맥 벡터.

    입력  : primary (B, 512), auxiliary (B, T, 512)
    출력  : context (B, 512), weights (B, T)
    """

    def __init__(self, config: CrossAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CrossAttentionConfig()

        d = self.config.d_model
        self.query_proj = nn.Linear(d, d)
        self.key_proj = nn.Linear(d, d)
        self.value_proj = nn.Linear(d, d)

        self.scale = 1.0 / math.sqrt(d) if self.config.scaled else 1.0

    def forward(
        self,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            primary:   (B, 512) 뉴스 기반 주 신호 벡터
            auxiliary: (B, T, 512) 시점별 보조 표현 벡터
            mask:      (B, T) bool. True인 시점만 참조(패딩 구간 제외용).

        Returns:
            context: (B, 512) 보조 문맥 벡터
            weights: (B, T) 시점별 어텐션 가중치 (합 = 1)
        """
        if primary.dim() != 2:
            raise ValueError(f"primary는 (B, D) 형태여야 합니다: {tuple(primary.shape)}")
        if auxiliary.dim() != 3:
            raise ValueError(f"auxiliary는 (B, T, D) 형태여야 합니다: {tuple(auxiliary.shape)}")
        if primary.size(0) != auxiliary.size(0):
            raise ValueError("primary와 auxiliary의 배치 크기가 다릅니다.")
        if primary.size(-1) != auxiliary.size(-1):
            raise ValueError("primary와 auxiliary의 표현 차원이 다릅니다.")

        q = self.query_proj(primary)        # (B, D)
        k = self.key_proj(auxiliary)        # (B, T, D)
        v = self.value_proj(auxiliary)      # (B, T, D)

        # 시점별 관련도 = 질의와 각 시점 키의 내적
        scores = torch.einsum("bd,btd->bt", q, k) * self.scale  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        weights = torch.softmax(scores, dim=-1)                 # (B, T), 합 = 1
        context = torch.einsum("bt,btd->bd", weights, v)        # (B, D)
        return context, weights
