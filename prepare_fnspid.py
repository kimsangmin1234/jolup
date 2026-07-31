"""FNSPID 데이터셋 → 본 모델 학습 형식 변환.

FNSPID (Financial News and Stock Price Integration Dataset, KDD 2024)
    https://huggingface.co/datasets/Zihan1004/FNSPID

    wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
    wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
    unzip full_history.zip

입력 형식
---------
뉴스 ``nasdaq_exteral_data.csv`` (약 5GB, 청크 단위로 스트리밍 처리):
    Date, Article_title, Stock_symbol, Url, Publisher, Author, Article,
    Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary

주가 ``full_history/{ticker}.csv``:
    Date, Open, High, Low, Close, Adj Close, Volume

출력 형식
---------
* ``news_cache.jsonl`` : data/dataset.py 가 읽는 레코드
                         (news_id, ticker, date, summary, sentiment, embedding, label)
* ``indicators.npz``   : 종목별 9종 지표 행렬 + 날짜 인덱스

감성 점수
---------
FNSPID의 ``Sentiment_gpt`` 는 1~5 척도이므로 논문의 -1~+1 범위로 선형 변환한다.
해당 열이 없으면 ``--sentiment llm`` 으로 GPT-4o-mini를 호출해 직접 산출한다.

사용 예::

    python prepare_fnspid.py \\
        --news nasdaq_exteral_data.csv \\
        --price-dir full_history \\
        --tickers AAPL,MSFT,NVDA,AMZN,GOOGL \\
        --start 2015-01-01 --end 2023-12-31 \\
        --out-records data/news_cache.jsonl \\
        --out-indicators data/indicators.npz \\
        --embedding openai
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from config import NewsEncoderConfig
from data.technical_indicators import compute_indicators

logger = logging.getLogger(__name__)

# FNSPID 요약 열 우선순위. 앞쪽 열이 있으면 그것을 요약으로 사용한다.
SUMMARY_COLUMNS = ("Textrank_summary", "Luhn_summary", "Lsa_summary",
                   "Lexrank_summary", "Article_title")


# --------------------------------------------------------------------------
# 주가 → 9종 지표 + 등락률 라벨
# --------------------------------------------------------------------------

def load_price_csv(path: Path) -> dict[str, np.ndarray]:
    """FNSPID 주가 CSV를 읽어 날짜순 정렬된 배열로 반환한다."""
    dates: list[str] = []
    rows: list[tuple[float, float, float, float]] = []

    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                high = float(row["High"])
                low = float(row["Low"])
                close = float(row["Close"])
                volume = float(row["Volume"])
            except (KeyError, TypeError, ValueError):
                continue  # 결측/손상 행은 건너뛴다
            if not all(np.isfinite([high, low, close, volume])) or close <= 0:
                continue
            dates.append(str(row["Date"])[:10])
            rows.append((high, low, close, volume))

    if not rows:
        raise ValueError(f"{path}: 유효한 주가 행이 없습니다.")

    order = np.argsort(np.array(dates))
    arr = np.array(rows, dtype=np.float64)[order]
    return {
        "dates": np.array(dates)[order],
        "high": arr[:, 0],
        "low": arr[:, 1],
        "close": arr[:, 2],
        "volume": arr[:, 3],
    }


def build_indicator_table(
    price_dir: Path, tickers: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """종목별 9종 지표 행렬, 날짜 인덱스, 다음 거래일 등락률을 만든다."""
    arrays: dict[str, np.ndarray] = {}
    index: dict[str, dict[str, int]] = {}
    labels: dict[str, dict[str, float]] = {}

    for ticker in tickers:
        path = _find_price_file(price_dir, ticker)
        if path is None:
            logger.warning("%s: 주가 파일을 찾지 못해 건너뜁니다.", ticker)
            continue

        series = load_price_csv(path)
        arrays[ticker] = compute_indicators(
            series["high"], series["low"], series["close"], series["volume"]
        )
        index[ticker] = {d: i for i, d in enumerate(series["dates"])}

        # 라벨: 당일 종가 → 다음 거래일 종가 등락률
        close = series["close"]
        ret = np.full_like(close, np.nan)
        ret[:-1] = (close[1:] - close[:-1]) / close[:-1]
        labels[ticker] = {
            d: float(r) for d, r in zip(series["dates"], ret) if np.isfinite(r)
        }

        logger.info("%s: %d 거래일 처리 완료", ticker, len(close))

    return arrays, index, labels


def _find_price_file(price_dir: Path, ticker: str) -> Path | None:
    """FNSPID는 종목 파일명이 대소문자 혼재이므로 두 형태를 모두 시도한다."""
    for name in (f"{ticker}.csv", f"{ticker.lower()}.csv", f"{ticker.upper()}.csv"):
        candidate = price_dir / name
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# 뉴스 스트리밍 처리
# --------------------------------------------------------------------------

def stream_news(
    news_path: Path,
    tickers: set[str],
    start: str,
    end: str,
    labels: dict[str, dict[str, float]],
    max_per_ticker: int,
) -> list[dict]:
    """대용량 뉴스 CSV를 한 줄씩 읽어 대상 종목·기간만 추출한다."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    collected: list[dict] = []
    per_ticker: dict[str, int] = {t: 0 for t in tickers}
    seen = 0

    with news_path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        summary_col = _pick_summary_column(reader.fieldnames or [])
        has_sentiment = "Sentiment_gpt" in (reader.fieldnames or [])
        logger.info("요약 열: %s / Sentiment_gpt 존재: %s", summary_col, has_sentiment)

        for row in reader:
            seen += 1
            if seen % 1_000_000 == 0:
                logger.info("뉴스 %d행 스캔, %d건 수집", seen, len(collected))

            ticker = (row.get("Stock_symbol") or "").strip().upper()
            if ticker not in tickers or per_ticker[ticker] >= max_per_ticker:
                continue

            date = str(row.get("Date") or "")[:10]
            if not (start <= date <= end):
                continue

            # 라벨(다음 거래일 등락률)이 있는 날짜만 학습에 쓸 수 있다.
            label = labels.get(ticker, {}).get(date)
            if label is None:
                continue

            summary = (row.get(summary_col) or "").strip()
            if not summary:
                continue

            record = {
                "news_id": f"{ticker}-{date}-{per_ticker[ticker]}",
                "ticker": ticker,
                "date": date,
                "summary": summary,
                "label": label,
            }
            if has_sentiment:
                sentiment = _scale_sentiment(row.get("Sentiment_gpt"))
                if sentiment is not None:
                    record["sentiment"] = sentiment

            collected.append(record)
            per_ticker[ticker] += 1

    logger.info("뉴스 총 %d행 스캔, %d건 수집", seen, len(collected))
    return collected


def _pick_summary_column(fieldnames: list[str]) -> str:
    for name in SUMMARY_COLUMNS:
        if name in fieldnames:
            return name
    if "Article" in fieldnames:
        return "Article"
    raise ValueError(f"요약으로 쓸 열을 찾지 못했습니다. 열 목록: {fieldnames}")


def _scale_sentiment(raw) -> float | None:
    """FNSPID의 1~5 척도를 논문의 -1~+1 범위로 변환한다."""
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return max(-1.0, min(1.0, (score - 3.0) / 2.0))


# --------------------------------------------------------------------------
# 임베딩 백엔드
# --------------------------------------------------------------------------

def embed_records(records: list[dict], backend: str, batch_size: int) -> None:
    """각 레코드에 ``embedding`` 키를 채운다 (제자리 수정)."""
    summaries = [r["summary"] for r in records]

    if backend == "openai":
        from modules.news_encoder import NewsLLMExtractor

        extractor = NewsLLMExtractor(NewsEncoderConfig())
        vectors: list[list[float]] = []
        for start in range(0, len(summaries), batch_size):
            vectors.extend(extractor.embed(summaries[start : start + batch_size]))
            logger.info("임베딩 %d / %d", len(vectors), len(summaries))

    elif backend == "local":
        # 논문은 text-embedding-3-small을 쓰지만, 외부 API를 못 쓰는 환경을 위한 대안.
        # 차원이 다르므로 NewsEncoderConfig.embedding_dim 도 함께 맞춰야 한다.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(
            summaries, batch_size=batch_size, show_progress_bar=True
        ).tolist()
        logger.warning(
            "로컬 임베딩(%d차원)은 논문의 text-embedding-3-small(1536차원)과 다릅니다. "
            "NewsEncoderConfig.embedding_dim 을 %d 로 설정하십시오.",
            len(vectors[0]), len(vectors[0]),
        )

    else:
        raise ValueError(f"알 수 없는 임베딩 백엔드: {backend}")

    for record, vector in zip(records, vectors):
        record["embedding"] = vector


def fill_missing_sentiment(records: list[dict], batch_size: int) -> None:
    """Sentiment_gpt 열이 없을 때 GPT-4o-mini로 감성 점수를 산출한다."""
    from modules.news_encoder import NewsLLMExtractor

    extractor = NewsLLMExtractor(NewsEncoderConfig())
    todo = [r for r in records if "sentiment" not in r]
    logger.info("감성 점수 산출 대상 %d건", len(todo))

    for i, record in enumerate(todo, 1):
        try:
            _, sentiment = extractor.summarize_and_score(record["summary"])
        except Exception:
            logger.exception("감성 산출 실패(%s), 0.0으로 대체", record["news_id"])
            sentiment = 0.0
        record["sentiment"] = sentiment
        if i % batch_size == 0:
            logger.info("감성 %d / %d", i, len(todo))


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FNSPID → 학습 형식 변환")
    parser.add_argument("--news", required=True, help="nasdaq_exteral_data.csv 경로")
    parser.add_argument("--price-dir", required=True, help="full_history 디렉터리")
    parser.add_argument("--tickers", required=True, help="쉼표 구분 종목 코드")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--max-per-ticker", type=int, default=5000)
    parser.add_argument("--out-records", default="data/news_cache.jsonl")
    parser.add_argument("--out-indicators", default="data/indicators.npz")
    parser.add_argument("--embedding", choices=("openai", "local"), default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logger.info("대상 종목: %s", ", ".join(tickers))

    # 1) 주가 → 지표 + 라벨
    arrays, index, labels = build_indicator_table(Path(args.price_dir), tickers)
    if not arrays:
        raise SystemExit("처리된 종목이 없습니다. --price-dir 경로를 확인하십시오.")

    # 2) 뉴스 추출
    records = stream_news(
        Path(args.news), set(arrays), args.start, args.end, labels, args.max_per_ticker
    )
    if not records:
        raise SystemExit("조건에 맞는 뉴스가 없습니다. 기간·종목을 확인하십시오.")

    # 3) 감성 점수 보완 + 임베딩
    if any("sentiment" not in r for r in records):
        fill_missing_sentiment(records, args.batch_size)
    embed_records(records, args.embedding, args.batch_size)

    # 4) 저장
    out_records = Path(args.out_records)
    out_records.parent.mkdir(parents=True, exist_ok=True)
    with out_records.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_indicators = Path(args.out_indicators)
    out_indicators.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for ticker, matrix in arrays.items():
        payload[f"{ticker}__values"] = matrix
        ordered = sorted(index[ticker], key=lambda d: index[ticker][d])
        payload[f"{ticker}__dates"] = np.array(ordered)
    np.savez_compressed(out_indicators, **payload)

    logger.info("완료 — 레코드 %d건 → %s", len(records), out_records)
    logger.info("완료 — 종목 %d개 → %s", len(arrays), out_indicators)


if __name__ == "__main__":
    main()
