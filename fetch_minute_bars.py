"""Alpaca Market Data API 로 분 단위 주가를 받는다.

장중 뉴스의 라벨을 '발행 시각의 주가 → 당일 종가'로 만들려면 분 단위 주가가
필요하다. FNSPID 는 일 단위 주가만 제공한다.

Alpaca 무료(Basic) 플랜은 2016년 이후 전체 시장(SIP) 분봉을 제공한다(최근
15분 제외). 요청 한도는 분당 200회, 요청당 최대 10,000개 봉이다. 22종목 ×
약 3.3년이면 수백 회 요청으로 끝난다. 무료 계정의 데이터 권한이 IEX 로
제한된 경우 ``--feed iex`` 로 받는다(거래소 한 곳의 체결만 담겨 거래가 적은
종목은 빈 분이 생긴다).

키는 https://alpaca.markets 가입 후 Paper 계정 대시보드에서 발급한다.

    export APCA_API_KEY_ID=...
    export APCA_API_SECRET_KEY=...
    python fetch_minute_bars.py --tickers AAPL,MSFT --start 2020-08-31 --end 2023-12-29

결과: data/fnspid/minute/{TICKER}_{YEAR}.csv.gz
    열: t(UTC, ISO), o, h, l, c, v  — 시간외(04:00~20:00 ET) 포함
종목·연도 파일이 이미 있으면 건너뛰므로 중단 후 재실행해도 된다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

URL = "https://data.alpaca.markets/v2/stocks/bars"
DEFAULT_TICKERS = ("AAPL,AMC,AMD,AMZN,BA,BLNK,CVX,DIS,F,FCEL,GE,GM,GME,INTC,KO,MRK,"
                   "MSFT,MU,NKLA,NVDA,TSLA,WMT")


def request(params: dict, key: str, secret: str, retries: int = 6) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{URL}?{query}", headers={
        "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300]
            if exc.code == 429 or exc.code >= 500:
                logger.warning("  %d 응답, %.0f초 후 재시도", exc.code, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise SystemExit(f"요청 실패 {exc.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("  연결 오류(%s), %.0f초 후 재시도", exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise SystemExit("재시도 소진")


def fetch(ticker: str, start: str, end: str, feed: str, key: str, secret: str) -> list[dict]:
    """[start, end] 구간의 1분봉 전부. 페이지를 따라가며 모은다."""
    bars, token = [], None
    while True:
        params = {"symbols": ticker, "timeframe": "1Min", "start": f"{start}T00:00:00Z",
                  "end": f"{end}T23:59:59Z", "limit": 10000, "adjustment": "raw",
                  "feed": feed, "sort": "asc"}
        if token:
            params["page_token"] = token
        data = request(params, key, secret)
        bars.extend((data.get("bars") or {}).get(ticker, []))
        token = data.get("next_page_token")
        time.sleep(0.31)                      # 분당 200회 한도 아래로
        if not token:
            return bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2020-08-31")
    ap.add_argument("--end", default="2023-12-29")
    ap.add_argument("--feed", choices=("sip", "iex"), default="sip")
    ap.add_argument("--out", default="data/fnspid/minute")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("APCA_API_KEY_ID / APCA_API_SECRET_KEY 환경변수가 필요합니다.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    years = range(int(args.start[:4]), int(args.end[:4]) + 1)
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        for year in years:
            path = out / f"{ticker}_{year}.csv.gz"
            if path.exists():
                continue
            start = max(args.start, f"{year}-01-01")
            end = min(args.end, f"{year}-12-31")
            t0 = time.time()
            bars = fetch(ticker, start, end, args.feed, key, secret)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "o", "h", "l", "c", "v"])
                for b in bars:
                    w.writerow([b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]])
            tmp.rename(path)
            logger.info("%s %d: %d봉 (%.0fs)", ticker, year, len(bars), time.time() - t0)


if __name__ == "__main__":
    main()
