"""FNSPID 뉴스 CSV 스트리밍 필터.

``nasdaq_exteral_data.csv`` 는 약 22GB이므로 디스크에 통째로 내려받기 어렵다.
이 스크립트는 표준입력으로 들어오는 CSV 스트림을 한 행씩 읽어 대상 종목·기간에
해당하는 행만 작은 CSV로 기록한다. 원본은 디스크에 남지 않는다.

사용 예::

    curl -sSL https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv \\
      | python stream_fnspid_news.py \\
            --tickers AAPL,MSFT,NVDA,AMZN,GOOGL \\
            --start 2015-01-01 --end 2023-12-31 \\
            --out data/fnspid_news_subset.csv

출력 CSV는 ``prepare_fnspid.py --news`` 가 그대로 읽을 수 있는 형식이다.

기사 본문(``Article``)에 줄바꿈이 포함되어 있으므로 반드시 csv 모듈로 파싱해야
한다. 줄 단위 grep은 행을 깨뜨린다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time

# 출력에 남길 열. 본문 전체(Article)는 용량이 크므로 기본적으로 제외한다.
OUTPUT_COLUMNS = (
    "Date", "Article_title", "Stock_symbol", "Url",
    "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="FNSPID 뉴스 CSV 스트리밍 필터")
    parser.add_argument("--tickers", required=True, help="쉼표 구분 종목 코드")
    parser.add_argument("--start", default="1999-01-01")
    parser.add_argument("--end", default="2099-12-31")
    parser.add_argument("--out", required=True, help="필터 결과 CSV 경로")
    parser.add_argument("--max-per-ticker", type=int, default=0,
                        help="종목당 최대 행 수 (0이면 제한 없음)")
    parser.add_argument("--keep-article", action="store_true",
                        help="기사 본문(Article) 열도 함께 저장")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--count-all", default="",
                        help="지정하면 전 종목의 뉴스 건수를 이 CSV에 함께 기록한다. "
                             "종목 선정 근거를 데이터로 잡을 때 쓴다.")
    args = parser.parse_args()

    tickers = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    columns = list(OUTPUT_COLUMNS)
    if args.keep_article:
        columns.insert(4, "Article")

    # 기사 본문이 매우 길 수 있으므로 필드 크기 제한을 올린다.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace",
                              newline="")
    reader = csv.DictReader(stream)

    per_ticker: dict[str, int] = {t: 0 for t in tickers}
    all_counts: dict[str, int] = {}   # --count-all 용: 전 종목 뉴스 건수
    seen = 0
    written = 0
    started = time.time()

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            seen += 1
            if seen % args.progress_every == 0:
                elapsed = time.time() - started
                print(f"  {seen:,}행 스캔 / {written:,}행 기록 "
                      f"({elapsed:.0f}초)", file=sys.stderr, flush=True)

            symbol = (row.get("Stock_symbol") or "").strip().upper()

            if args.count_all and symbol:
                date_all = str(row.get("Date") or "")[:10]
                if args.start <= date_all <= args.end:
                    all_counts[symbol] = all_counts.get(symbol, 0) + 1

            if symbol not in tickers:
                continue
            if args.max_per_ticker and per_ticker[symbol] >= args.max_per_ticker:
                continue

            date = str(row.get("Date") or "")[:10]
            if not (args.start <= date <= args.end):
                continue

            writer.writerow(row)
            per_ticker[symbol] += 1
            written += 1

    if args.count_all:
        with open(args.count_all, "w", encoding="utf-8", newline="") as f:
            counter = csv.writer(f)
            counter.writerow(["Stock_symbol", "news_count"])
            for symbol, count in sorted(all_counts.items(),
                                        key=lambda kv: -kv[1]):
                counter.writerow([symbol, count])
        print(f"전 종목 뉴스 건수 {len(all_counts):,}개 → {args.count_all}",
              file=sys.stderr)

    elapsed = time.time() - started
    print(f"완료 — {seen:,}행 스캔, {written:,}행 기록 ({elapsed:.0f}초)",
          file=sys.stderr)
    for ticker in sorted(per_ticker):
        print(f"  {ticker}: {per_ticker[ticker]:,}건", file=sys.stderr)


if __name__ == "__main__":
    main()
