"""모듈 5. 등락률 예측 (Ⅱ.5)

통합 표현 벡터를 MLP에 입력하여 단일 스칼라 등락률 예측값으로 변환한다.

    512 → 256 → 64 → 1

각 은닉층 사이에는 ReLU를 두고, 마지막 출력층에는 활성함수를 두지 않아
실수 값이 그대로 산출되도록 한다(회귀).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import PredictorConfig


class ReturnPredictor(nn.Module):
    """통합 표현 벡터 → 등락률(스칼라).

    입력  : (B, 512)
    출력  : (B,)
    """

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()

        layers: list[nn.Module] = []
        in_dim = self.config.d_model
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden_dim

        # 출력층: 활성함수 없음 → 실수 값 그대로
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.dim() != 2:
            raise ValueError(f"fused는 (B, D) 형태여야 합니다: {tuple(fused.shape)}")
        if fused.size(-1) != self.config.d_model:
            raise ValueError(
                f"표현 차원 불일치: {fused.size(-1)} != {self.config.d_model}"
            )
        return self.mlp(fused).squeeze(-1)   # (B, 1) -> (B,)
