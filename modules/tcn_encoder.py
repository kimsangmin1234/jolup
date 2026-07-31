"""모듈 2. TCN 기반 보조 벡터 생성 (Ⅱ.2)

뉴스와 관련된 종목의 과거 30일간 9가지 기술적 지표 시계열을 입력으로 받아
시점별 512차원 보조 표현 벡터를 생성한다.

핵심 설계:
* **인과적(causal) 합성곱** — 합성곱 수행 전에 시계열의 앞쪽만 0으로 채우고
  뒤쪽 잉여분을 잘라낸다. 각 시점의 출력은 자기 자신과 과거 시점만 참조하므로
  미래 정보 누수가 구조적으로 차단된다.
* **팽창(dilation) 스택** — dilation을 1, 2, 4, 8로 늘려 30일 구간 전체를
  적은 층수로 커버한다.
* **가중치 정규화** — BatchNorm은 시간 축을 포함해 통계를 산출하므로 학습 모드에서
  미래 시점 정보가 통계에 섞인다. 인과성을 유지하기 위해 weight norm을 사용한다.
* 출력 형태는 (B, 30, 512).

입력 정규화(학습 구간 통계만 사용하는 최소-최대 정규화)는
``data/technical_indicators.py`` 의 ``MinMaxScaler`` 가 담당한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # PyTorch 2.1+
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # 구버전 호환
    from torch.nn.utils import weight_norm

from config import TCNConfig


class CausalConv1d(nn.Conv1d):
    """왼쪽만 패딩하는 1차원 팽창 합성곱.

    ``padding = (kernel_size - 1) * dilation`` 만큼 양쪽에 패딩이 생기지 않도록
    직접 왼쪽에만 0을 채운 뒤 합성곱을 수행한다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        # 앞쪽에만 0을 채워 미래 시점 참조를 차단한다.
        x = nn.functional.pad(x, (self.left_padding, 0))
        return super().forward(x)


class TemporalBlock(nn.Module):
    """TCN 잔차 블록: (인과 합성곱 → ReLU → Dropout) × 2 + 잔차 연결."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # 가중치 정규화를 쓴다. BatchNorm은 시간 축을 포함해 통계를 내므로
        # 학습 모드에서 미래 시점 정보가 통계에 섞여 인과성이 깨진다.
        self.conv1 = weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation), name="weight"
        )
        self.conv2 = weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation), name="weight"
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 채널 수가 바뀌는 첫 블록에서는 1x1 합성곱으로 잔차 경로를 맞춘다.
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        out = self.dropout(self.activation(self.conv1(x)))
        out = self.dropout(self.activation(self.conv2(out)))
        return self.activation(out + self.downsample(x))


class TCNEncoder(nn.Module):
    """9종 기술적 지표 시계열 → 시점별 512차원 보조 표현 벡터.

    입력  : (B, T, 9)   T = 30 (과거 30일)
    출력  : (B, T, 512)
    """

    def __init__(self, config: TCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConfig()

        blocks: list[nn.Module] = []
        in_channels = self.config.in_channels
        for i in range(self.config.num_blocks):
            # 마지막 블록의 출력 채널을 표현 차원(512)에 맞춘다.
            is_last = i == self.config.num_blocks - 1
            out_channels = self.config.d_model if is_last else self.config.hidden_channels
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.config.kernel_size,
                    dilation=2 ** i,          # 1, 2, 4, 8, ...
                    dropout=self.config.dropout,
                )
            )
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(self.config.d_model)

    @property
    def receptive_field(self) -> int:
        """수용 영역(일 단위). lookback(30일)을 덮는지 확인용."""
        k = self.config.kernel_size
        # 블록당 인과 합성곱 2개
        return 1 + sum(2 * (k - 1) * (2 ** i) for i in range(self.config.num_blocks))

    def forward(self, indicators: torch.Tensor) -> torch.Tensor:
        if indicators.dim() != 3:
            raise ValueError(
                f"기술적 지표는 (B, T, C) 형태여야 합니다: {tuple(indicators.shape)}"
            )
        if indicators.size(-1) != self.config.in_channels:
            raise ValueError(
                f"지표 개수 불일치: {indicators.size(-1)} != {self.config.in_channels}"
            )

        x = indicators.transpose(1, 2)   # (B, T, C) -> (B, C, T)
        x = self.blocks(x)               # (B, 512, T)
        x = x.transpose(1, 2)            # (B, T, 512)
        return self.norm(x)
