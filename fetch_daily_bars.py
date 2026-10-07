"""Alpaca 로 여러 종목의 일봉을 한꺼번에 받는다 (분할 조정).

EDT 종목(4천여 개)의 과거 30일 기술적 지표를 만들기 위해 쓴다. 한 요청에 종목 100개씩 묶는다.

    export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
    python fetch_daily_bars.py --tickers-file tickers.txt --start 2019-11-01 --end 2021-06-10 \\
        --out data/edt/daily_bars.csv.gz
"""
import argparse
import csv
import gzip
import logging
import os
import time

from fetch_minute_bars import request

URL = "https://data.alpaca.markets/v2/stocks/bars"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    key, secret = os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    tickers = [t for t in open(a.tickers_file).read().split(",") if t]
    rows = 0
    with gzip.open(a.out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "date", "o", "h", "l", "c", "v"])
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i + 100]
            token = None
            while True:
                params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": f"{a.start}T00:00:00Z",
                          "end": f"{a.end}T23:59:59Z", "limit": 10000, "adjustment": "split", "feed": "sip"}
                if token:
                    params["page_token"] = token
                data = request(params, key, secret)
                for sym, bars in (data.get("bars") or {}).items():
                    for b in bars:
                        w.writerow([sym, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"]])
                        rows += 1
                token = data.get("next_page_token")
                time.sleep(0.31)
                if not token:
                    break
            logging.info("%d/%d 종목, 누적 %d봉", min(i + 100, len(tickers)), len(tickers), rows)


if __name__ == "__main__":
    main()
