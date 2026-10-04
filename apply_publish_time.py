"""복구한 발행 시각으로 레코드를 다시 라벨링한다.

``crawl_publish_time.py`` 가 모은 발행 시각을 레코드에 결합하고, 뉴스가
장 마감(동부시간 16:00) 전인지 후인지에 따라 예측 대상을 나눈다.

    장 마감 전 뉴스  →  당일 시가→종가  close[d]/open[d] - 1,    윈도우 d-1 까지
    장 마감 후 뉴스  →  다음날 등락률    close[d+1]/close[d] - 1, 윈도우 d 까지

장 마감 전 뉴스의 라벨을 전일 종가가 아닌 당일 시가에서 시작하는 이유:
전일 종가→시가 갭은 장중 뉴스가 나오기 **전**에 끝난 움직임이다. 이를 라벨에
넣으면 모델이 미래가 아니라 기사가 이미 보도한 움직임을 맞히게 된다
(experiments/diagnosis/SENTIMENT_SIGNIFICANCE.md). 예전 방식(전일 종가→당일 종가)은
``--intraday-label prev_close`` 로 재현할 수 있다.

시각을 복구하지 못한 레코드(기사 삭제 등)는 ``--fallback`` 정책에 따라 처리한다.
레코드에 ``published_et`` 가 이미 있으면 그 값을 쓴다(``--times`` 생략 가능).
크롤링한 발행 날짜가 데이터셋 날짜와 다르면(약 3.5%) 실제 발행 시점을 확신할 수 없어
기본으로 제외한다(``--date-mismatch``).

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
            "open": series["open"],
            "row": {d: i for i, d in enumerate(series["dates"])},
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="발행 시각 기반 재라벨링")
    parser.add_argument("--records", required=True, help="url 필드를 가진 레코드 JSONL")
    parser.add_argument("--times", default="", help="crawl_publish_time.py 출력(쉼표로 여러 개)")
    parser.add_argument("--price-dir", required=True)
    parser.add_argument("--price-start", default="2020-07-06")
    parser.add_argument("--price-alias", default="GOOGL=GOOG")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fallback", choices=("next_day", "drop"), default="drop",
                        help="시각을 복구하지 못한 레코드 처리 방식. 기본은 학습에서 제외(drop). "
                             "당일/다음날 중 어느 라벨이 맞는지 알 수 없기 때문이다.")
    parser.add_argument("--close-hour", type=int, default=MARKET_CLOSE_HOUR)
    parser.add_argument("--intraday-label", choices=("open_close", "prev_close"), default="open_close",
                        help="장 마감 전 뉴스 라벨. open_close: 당일 시가→종가(기본), "
                             "prev_close: 전일 종가→당일 종가(예전 방식)")
    parser.add_argument("--date-mismatch", choices=("drop", "keep"), default="drop",
                        help="크롤링한 발행 날짜가 데이터셋 날짜와 다른 레코드 처리(약 3.5%%). "
                             "어느 쪽이 맞는지 알 수 없어 기본은 제외한다.")
    parser.add_argument("--labels-only", action="store_true",
                        help="news_id·라벨·앵커·horizon 만 저장한다(임베딩을 다시 쓰지 않음)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = load_jsonl(Path(args.records))
    # --times 는 쉼표로 여러 파일을 받는다(서버 수집분 + 로컬 PC 수집분).
    times = {}
    for path in args.times.split(","):
        if path.strip() and Path(path.strip()).exists():
            times.update({t["url"]: t["published_et"]
                          for t in load_jsonl(Path(path.strip())) if t.get("published_et")})
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

        published = record.get("published_et") or times.get(record.get("url", ""))
        if published and published[:10] != record["date"] and args.date_mismatch == "drop":
            stats["날짜불일치_제외"] += 1
            continue
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
            if args.intraday_label == "open_close":
                opened = table["open"][row]
                if not np.isfinite(opened) or opened <= 0:
                    stats["시가없음"] += 1
                    continue
                label = (close[row] - opened) / opened
            else:
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
        # 본문은 캐시 용량을 키우므로 남기지 않는다(원본은 articles/ 에 있다).
        updated.pop("text", None)
        out_records.append(updated)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in out_records:
            if args.labels_only:
                record = {k: record.get(k) for k in
                          ("news_id", "ticker", "date", "published_et", "horizon", "anchor", "label")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("저장 %d건 → %s", len(out_records), out_path)
    for key, value in stats.most_common():
        logger.info("  %-16s %7d", key, value)

    covered = stats["마감전"] + stats["마감후"]
    if covered:
        logger.info("시각 복구분 중 마감 전 비율: %.1f%%", stats["마감전"] / covered * 100)

    # 라벨 분포를 horizon 별로 보여준다. 두 집단의 성격이 다르므로
    # 학습 결과를 해석할 때 이 차이를 알고 있어야 한다.
    by_horizon: dict[str, list[float]] = collections.defaultdict(list)
    for record in out_records:
        by_horizon[record["horizon"]].append(record["label"])
    for horizon, values in sorted(by_horizon.items()):
        arr = np.asarray(values)
        logger.info("  %-9s %6d건  라벨 평균 %+.3f%%  표준편차 %.2f%%",
                    horizon, len(arr), arr.mean() * 100, arr.std() * 100)


if __name__ == "__main__":
    main()
