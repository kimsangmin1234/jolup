"""모듈 4. 게이트 기반 잔차 결합 (Ⅱ.4)

교차 어텐션은 가중치 합이 항상 1이 되도록 설계되어 있어, 보조 정보가 예측에
도움이 되지 않는 경우에도 일정 비중을 부여한다. 이를 보정하기 위해 어텐션 출력
직후에 게이트를 두고, 보조 문맥 벡터가 주 신호에 반영되는 정도를 학습한다.

    g = σ(W [h_news ; c_aux] + b)      # 512차원, 각 차원마다 0~1
    h = h_news + g ⊙ c_aux             # 잔차 형태로 결합

주 신호는 항상 그대로 보존되고 보조 정보만 게이트로 조절되므로,
"뉴스 중심(news-driven)" 표현이라는 논문의 설계 의도가 유지된다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import GatedFusionConfig


class GatedResidualFusion(nn.Module):
    """주 신호 + 게이트로 조절된 보조 문맥 → 512차원 통합 표현 벡터.

    입력  : primary (B, 512), context (B, 512)
    출력  : fused (B, 512), gate (B, 512)
    """

    def __init__(self, config: GatedFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or GatedFusionConfig()

        d = self.config.d_model
        hidden = self.config.gate_hidden_dim

        if hidden is None:
            # 논문 기술 그대로: [주 신호 ; 보조 문맥] → 차원별 게이트 값
            self.gate_net = nn.Linear(2 * d, d)
        else:
            self.gate_net = nn.Sequential(
                nn.Linear(2 * d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d),
            )

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        primary: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            primary: (B, 512) 뉴스 기반 주 신호 벡터
            context: (B, 512) 교차 어텐션이 만든 보조 문맥 벡터

        Returns:
            fused: (B, 512) 통합 표현 벡터
            gate:  (B, 512) 차원별 게이트 값 (0~1). 해석/분석용.
        """
        if primary.shape != context.shape:
            raise ValueError(
                f"primary와 context의 형태가 다릅니다: "
                f"{tuple(primary.shape)} vs {tuple(context.shape)}"
            )

        gate = torch.sigmoid(self.gate_net(torch.cat([primary, context], dim=-1)))
        fused = primary + gate * context     # 잔차 결합
        return self.norm(fused), gate
