"""9종 기술적 지표 계산 및 최소-최대 정규화 (Ⅰ. 서론, Ⅱ.2)

논문이 사용하는 지표는 다음 9종이다.

    종가, 거래량, 5일 이동평균, 20일 이동평균, 상대강도지수(RSI),
    MACD, 볼린저 밴드 상한선, 볼린저 밴드 하한선, ATR

정규화는 **학습 구간에서 산출한 통계량만** 사용하는 최소-최대 정규화를 적용한다
(검증/평가 구간 통계를 쓰면 미래 정보 누수가 발생한다).

pandas 없이 numpy만으로 구현하여 의존성을 최소화했다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import TECHNICAL_INDICATORS


# --------------------------------------------------------------------------
# 개별 지표
# --------------------------------------------------------------------------

def simple_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """단순 이동평균. 앞쪽 (window-1)개 구간은 누적 평균으로 채운다."""
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values)
    full = (cumsum[window:] - cumsum[:-window]) / window
    out[window - 1 :] = full
    counts = np.arange(1, window)
    out[: window - 1] = cumsum[1:window] / counts
    return out


def exponential_moving_average(values: np.ndarray, span: int) -> np.ndarray:
    """지수 이동평균 (pandas의 ewm(adjust=False)와 동일한 재귀식)."""
    values = np.asarray(values, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def relative_strength_index(close: np.ndarray, period: int = 14) -> np.ndarray:
    """상대강도지수(RSI). Wilder 평활 방식, 0~100 범위."""
    close = np.asarray(close, dtype=np.float64)
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # 손실이 전혀 없는 구간은 RSI = 100
    rsi[avg_loss == 0] = 100.0
    # 이득과 손실이 모두 없으면 중립값
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return rsi


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder 평활: 첫 period개는 단순 평균, 이후 재귀 갱신."""
    out = np.empty_like(values)
    if len(values) < period:
        return np.full_like(values, values.mean() if len(values) else 0.0)
    seed = values[:period].mean()
    out[:period] = seed
    for i in range(period, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def macd(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """MACD = 12일 EMA - 26일 EMA."""
    return exponential_moving_average(close, fast) - exponential_moving_average(close, slow)


def bollinger_bands(
    close: np.ndarray, window: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """볼린저 밴드 (상한선, 하한선)."""
    close = np.asarray(close, dtype=np.float64)
    middle = simple_moving_average(close, window)

    std = np.empty_like(close)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        std[i] = close[start : i + 1].std()

    return middle + num_std * std, middle - num_std * std


def average_true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """ATR: True Range의 Wilder 평활."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    true_range = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return _wilder_smooth(true_range, period)


# --------------------------------------------------------------------------
# 9종 지표 일괄 계산
# --------------------------------------------------------------------------

def compute_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """OHLCV 시계열 → (T, 9) 지표 행렬.

    열 순서는 ``config.TECHNICAL_INDICATORS`` 와 동일하다.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    lengths = {len(high), len(low), len(close), len(volume)}
    if len(lengths) != 1:
        raise ValueError("high/low/close/volume의 길이가 서로 다릅니다.")

    bb_upper, bb_lower = bollinger_bands(close)

    columns = {
        "close": close,
        "volume": volume,
        "ma5": simple_moving_average(close, 5),
        "ma20": simple_moving_average(close, 20),
        "rsi": relative_strength_index(close),
        "macd": macd(close),
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "atr": average_true_range(high, low, close),
    }
    return np.stack([columns[name] for name in TECHNICAL_INDICATORS], axis=1)


# --------------------------------------------------------------------------
# 최소-최대 정규화 (학습 구간 통계만 사용)
# --------------------------------------------------------------------------

@dataclass
class MinMaxScaler:
    """열별 최소-최대 정규화.

    ``fit`` 은 반드시 학습 구간 데이터로만 호출한다. 검증/평가 데이터는
    학습 구간에서 얻은 min/max로 ``transform`` 만 수행한다.
    """

    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        x = np.asarray(x, dtype=np.float64)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("fit()을 먼저 호출하십시오.")
        x = np.asarray(x, dtype=np.float64)
        span = np.maximum(self.maximum - self.minimum, self.eps)
        # 학습 구간을 벗어난 값은 [0, 1] 밖으로 나갈 수 있으므로 클리핑한다.
        return np.clip((x - self.minimum) / span, 0.0, 1.0)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    # -- 저장/복원 ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("fit()을 먼저 호출하십시오.")
        Path(path).write_text(
            json.dumps({"min": self.minimum.tolist(), "max": self.maximum.tolist()}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            minimum=np.asarray(payload["min"], dtype=np.float64),
            maximum=np.asarray(payload["max"], dtype=np.float64),
        )
