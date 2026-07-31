"""학습용 데이터셋.

한 개의 샘플 = (뉴스 1건, 해당 종목의 과거 30일 기술적 지표, 다음 거래일 등락률)

LLM 전처리(요약/감성/임베딩)는 비용이 크므로 학습 루프 밖에서 한 번만 수행하고
JSONL 캐시로 저장한 뒤, 이 데이터셋이 캐시를 읽어 쓰는 구조를 전제로 한다.

캐시 한 줄의 형식 (``news_cache.jsonl``)::

    {"news_id": "...", "ticker": "005930", "date": "2026-03-04",
     "sentiment": 0.42, "embedding": [1536개 실수], "label": 0.0132}

기술적 지표는 종목별 (T, 9) 배열을 담은 ``.npz`` 파일에서 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TECHNICAL_INDICATORS
from data.technical_indicators import MinMaxScaler


class NewsStockDataset(Dataset):
    """뉴스 + 과거 30일 지표 → 등락률 회귀 데이터셋.

    Args:
        records:     캐시 레코드 목록.
        indicators:  {ticker: (T, 9) 지표 행렬}.
        date_index:  {ticker: {날짜 문자열: 행 인덱스}}.
        scaler:      학습 구간 통계로 fit된 MinMaxScaler.
        lookback:    과거 참조 일수(기본 30).
    """

    def __init__(
        self,
        records: Sequence[dict],
        indicators: dict[str, np.ndarray],
        date_index: dict[str, dict[str, int]],
        scaler: MinMaxScaler,
        lookback: int = 30,
        embedding_dim: int = 1536,
    ) -> None:
        self.indicators = indicators
        self.date_index = date_index
        self.scaler = scaler
        self.lookback = lookback
        self.embedding_dim = embedding_dim

        # 30일 구간을 온전히 확보할 수 있는 레코드만 남긴다.
        self.records = [r for r in records if self._window_available(r)]

    def _anchor(self, record: dict) -> str:
        """지표 윈도우가 끝나는 거래일.

        ``anchor`` 가 없으면 뉴스 발생일을 쓴다(next_day 모드와 동일).
        same_day 모드에서는 뉴스 당일 종가가 입력에 들어가지 않도록
        prepare_fnspid.py 가 전 거래일을 anchor로 기록한다.
        """
        return record.get("anchor") or record["date"]

    def _window_available(self, record: dict) -> bool:
        ticker = record["ticker"]
        if ticker not in self.date_index:
            return False
        row = self.date_index[ticker].get(self._anchor(record))
        # anchor 당일까지 포함하여 lookback일이 필요하다.
        return row is not None and row + 1 >= self.lookback

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        ticker = record["ticker"]
        row = self.date_index[ticker][self._anchor(record)]

        window = self.indicators[ticker][row + 1 - self.lookback : row + 1]  # (30, 9)
        window = self.scaler.transform(window)

        embedding = np.asarray(record["embedding"], dtype=np.float32)
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"임베딩 차원 불일치({record.get('news_id')}): "
                f"{embedding.shape[0]} != {self.embedding_dim}"
            )

        return {
            "embedding": torch.from_numpy(embedding),
            "sentiment": torch.tensor(float(record["sentiment"]), dtype=torch.float32),
            "indicators": torch.from_numpy(window.astype(np.float32)),
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
        }


# --------------------------------------------------------------------------
# 로딩 유틸리티
# --------------------------------------------------------------------------

def load_records(path: str | Path) -> list[dict]:
    """JSONL 캐시를 읽는다."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_indicators(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """npz에서 종목별 지표 행렬과 날짜 인덱스를 읽는다.

    npz 키 규약: ``{ticker}__values`` (T, 9) / ``{ticker}__dates`` (T,)
    """
    archive = np.load(path, allow_pickle=True)
    indicators: dict[str, np.ndarray] = {}
    date_index: dict[str, dict[str, int]] = {}

    for key in archive.files:
        if not key.endswith("__values"):
            continue
        ticker = key[: -len("__values")]
        values = archive[key]
        if values.shape[1] != len(TECHNICAL_INDICATORS):
            raise ValueError(
                f"{ticker}: 지표 개수 불일치 {values.shape[1]} != {len(TECHNICAL_INDICATORS)}"
            )
        dates = archive[f"{ticker}__dates"]
        indicators[ticker] = values.astype(np.float64)
        date_index[ticker] = {str(d): i for i, d in enumerate(dates)}

    return indicators, date_index


def split_by_date(
    records: Sequence[dict], train_end: str, valid_end: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """시간 순 분할. 무작위 분할은 미래 정보 누수를 일으키므로 사용하지 않는다."""
    train = [r for r in records if r["date"] <= train_end]
    valid = [r for r in records if train_end < r["date"] <= valid_end]
    test = [r for r in records if r["date"] > valid_end]
    return train, valid, test


def fit_scaler_on_train(
    train_records: Sequence[dict],
    indicators: dict[str, np.ndarray],
    date_index: dict[str, dict[str, int]],
) -> MinMaxScaler:
    """학습 구간에 실제로 등장하는 시점까지의 지표만으로 스케일러를 fit한다."""
    chunks = []
    for ticker, rows in _last_train_row_per_ticker(train_records, date_index).items():
        chunks.append(indicators[ticker][: rows + 1])
    if not chunks:
        raise ValueError("학습 구간에 해당하는 지표 데이터가 없습니다.")
    return MinMaxScaler().fit(np.concatenate(chunks, axis=0))


def _last_train_row_per_ticker(
    train_records: Sequence[dict], date_index: dict[str, dict[str, int]]
) -> dict[str, int]:
    last: dict[str, int] = {}
    for record in train_records:
        ticker = record["ticker"]
        row = date_index.get(ticker, {}).get(record["date"])
        if row is not None:
            last[ticker] = max(last.get(ticker, 0), row)
    return last
