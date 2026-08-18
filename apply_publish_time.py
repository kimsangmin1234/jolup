"""복구한 발행 시각으로 레코드를 다시 라벨링한다.

``crawl_publish_time.py`` 가 모은 발행 시각을 레코드에 결합하고, 뉴스가
장 마감(동부시간 16:00) 전인지 후인지에 따라 예측 대상을 나눈다.

    장 마감 전 뉴스  →  당일  등락률  close[d]/close[d-1] - 1,  윈도우 d-1 까지
    장 마감 후 뉴스  →  다음날 등락률  close[d+1]/close[d] - 1,  윈도우 d 까지

이렇게 하면 "당일 뉴스가 당일 주가에 미치는 영향"이라는 논문의 설정을
지키면서도, 이미 실현된 등락률을 맞히는 누수를 피할 수 있다. 시각을 복구하지
못한 레코드(기사 삭제 등)는 ``--fallback`` 정책에 따라 처리한다.

사용 예::

    python apply_publish_time.py \\
        --records data/fnspid/records_article.jsonl \\
        --times times.jsonl \\
        --price-dir full_history --price-start 2020-07-06 \\
        --out data/fnspid/records_timed.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

import numpy as np

from prepare_fnspid import (MARKET_CLOSE_HOUR, _find_price_file, load_price_csv)

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_price_index(price_dir: Path, tickers: set[str], price_start: str,
                      alias: dict[str, str]) -> dict[str, dict]:
    """종목별 날짜 배열·종가·행 인덱스."""
    table: dict[str, dict] = {}
    for ticker in sorted(tickers):
        path = _find_price_file(price_dir, alias.get(ticker, ticker))
        if path is None:
            continue
        series = load_price_csv(path)
        if price_start:
            keep = series["dates"] >= price_start
            series = {k: v[keep] for k, v in series.items()}
        if len(series["dates"]) < 60:
            continue
        table[ticker] = {
            "dates": series["dates"],
            "close": series["close"],
            "row": {d: i for i, d in enumerate(series["dates"])},
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="발행 시각 기반 재라벨링")
    parser.add_argument("--records", required=True, help="url 필드를 가진 레코드 JSONL")
    parser.add_argument("--times", required=True, help="crawl_publish_time.py 출력")
    parser.add_argument("--price-dir", required=True)
    parser.add_argument("--price-start", default="2020-07-06")
    parser.add_argument("--price-alias", default="GOOGL=GOOG")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fallback", choices=("next_day", "drop"), default="next_day",
                        help="시각을 복구하지 못한 레코드 처리 방식")
    parser.add_argument("--close-hour", type=int, default=MARKET_CLOSE_HOUR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = load_jsonl(Path(args.records))
    times = {t["url"]: t["published_et"]
             for t in load_jsonl(Path(args.times)) if t.get("published_et")}
    logger.info("레코드 %d건 / 복구된 시각 %d건", len(records), len(times))

    alias = {}
    for pair in args.price_alias.split(","):
        if "=" in pair:
            a, b = pair.split("=", 1)
            alias[a.strip().upper()] = b.strip().upper()

    prices = build_price_index(Path(args.price_dir),
                               {r["ticker"] for r in records}, args.price_start, alias)

    stats: collections.Counter = collections.Counter()
    out_records: list[dict] = []

    for record in records:
        ticker = record["ticker"]
        table = prices.get(ticker)
        if table is None:
            stats["주가없음"] += 1
            continue

        row = table["row"].get(record["date"])
        if row is None:
            stats["거래일아님"] += 1
            continue

        published = times.get(record.get("url", ""))
        if published:
            hour = int(published[11:13])
            before_close = hour < args.close_hour
            stats["마감전" if before_close else "마감후"] += 1
        else:
            if args.fallback == "drop":
                stats["시각없음_제외"] += 1
                continue
            before_close = False           # 보수적으로 다음날 라벨을 쓴다
            stats["시각없음_다음날"] += 1

        close, dates = table["close"], table["dates"]
        if before_close:
            # 당일 등락률. 뉴스 당일 종가가 입력에 들어가지 않도록 윈도우를 앞당긴다.
            if row < 1:
                stats["구간부족"] += 1
                continue
            label = (close[row] - close[row - 1]) / close[row - 1]
            anchor = dates[row - 1]
        else:
            # 다음 거래일 등락률.
            if row + 1 >= len(close):
                stats["구간부족"] += 1
                continue
            label = (close[row + 1] - close[row]) / close[row]
            anchor = dates[row]

        if not np.isfinite(label):
            stats["라벨무효"] += 1
            continue

        updated = dict(record)
        updated["published_et"] = published
        updated["horizon"] = "same_day" if before_close else "next_day"
        updated["anchor"] = str(anchor)
        updated["label"] = float(label)
        out_records.append(updated)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in out_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("저장 %d건 → %s", len(out_records), out_path)
    for key, value in stats.most_common():
        logger.info("  %-16s %7d", key, value)

    covered = stats["마감전"] + stats["마감후"]
    if covered:
        logger.info("시각 복구분 중 마감 전 비율: %.1f%%", stats["마감전"] / covered * 100)


if __name__ == "__main__":
    main()
