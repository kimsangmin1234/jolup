"""Dukascopy 공개 시세 서버에서 미국 주식 1분봉을 받는다(키 불필요).

Dukascopy 는 미국 주식 CFD 의 과거 분봉을 공개 데이터 피드로 제공한다.
CFD 매수호가(BID) 기준이라 공식 체결가와 0.1~0.3% 차이가 나지만, 라벨은
같은 소스 안에서 '발행 직후 시가 → 마지막 분봉 종가' 비율로 계산하므로
수준 차이는 상쇄된다. 가격 단위(1/1000)와 액면분할 미조정도 비율에서는
영향이 없다. 정규장(09:30~16:00 ET) 분봉만 들어 있다.

서버가 요청 속도를 강하게 제한하므로(429) 장중 뉴스가 있는 종목·날짜만
받는다. 받은 날짜는 .work/dukascopy/ 에 남겨 두어 중단 후 재실행하면
이어서 받는다.

    python fetch_dukascopy.py --labels data/fnspid/labels_open_close.jsonl.gz --commit

결과: data/fnspid/minute/{TICKER}_{YEAR}.csv.gz  (열: t(UTC), o, h, l, c, v)
      apply_minute_labels.py 가 그대로 읽는다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import logging
import lzma
import struct
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
URL = "https://datafeed.dukascopy.com/datafeed/{sym}USUSD/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


class Fetcher:
    """요청 간격을 스스로 조절한다. 429 가 나면 간격을 늘리고 길게 쉰다."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def get(self, url: str) -> bytes | None:
        """본문(빈 바이트 가능) 또는 None(404, 데이터 없음)."""
        wait = 30.0
        for _ in range(12):
            time.sleep(self.delay)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code == 429:
                    self.delay = min(self.delay + 0.5, 6.0)
                    logger.info("  429 — %.0f초 대기, 간격 %.1f초", wait, self.delay)
                    time.sleep(wait)
                    wait = min(wait * 2, 600)
                    continue
                logger.warning("  HTTP %d — %.0f초 대기", exc.code, wait)
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                logger.warning("  연결 오류(%s) — %.0f초 대기", type(exc).__name__, wait)
                time.sleep(wait)
                wait = min(wait * 2, 600)
        raise RuntimeError(f"재시도 소진: {url}")


def decode(body: bytes, day: str) -> list[tuple]:
    """bi5(LZMA) → [(UTC ISO, o, h, l, c, v)], 거래량 0 인 분은 뺀다."""
    if not body:
        return []
    raw = lzma.decompress(body)
    base = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    rows = []
    for i in range(len(raw) // 24):
        sec, o, c, lo, hi, vol = struct.unpack(">5if", raw[i * 24:(i + 1) * 24])
        if vol <= 0:
            continue
        ts = (base + timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append((ts, o / 1000, hi / 1000, lo / 1000, c / 1000, round(vol, 4)))
    return rows


def git_commit(paths: list[str], message: str) -> None:
    subprocess.run(["git", "add", "-f", *paths], check=False)
    msg = (f"{message}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK")
    subprocess.run(["git", "-c", "user.email=sunshine31885@gmail.com", "-c", "user.name=Claude",
                    "commit", "-q", "-m", msg], check=False)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for delay in (2, 4, 8, 16):
        subprocess.run(["git", "pull", "-q", "--no-rebase", "origin", branch], check=False)
        if subprocess.run(["git", "push", "-q", "origin", branch]).returncode == 0:
            return
        time.sleep(delay)


def write_year_files(ticker: str, cache: Path, out: Path) -> list[str]:
    """캐시의 날짜별 파일을 종목·연도 파일로 모은다."""
    by_year: dict[str, list[Path]] = collections.defaultdict(list)
    for p in sorted(cache.glob("*.csv")):
        by_year[p.stem[:4]].append(p)
    written = []
    for year, files in by_year.items():
        path = out / f"{ticker}_{year}.csv.gz"
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "o", "h", "l", "c", "v"])
            for p in files:
                f.write(p.read_text())
        written.append(str(path))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/fnspid/labels_open_close.jsonl.gz")
    ap.add_argument("--out", default="data/fnspid/minute")
    ap.add_argument("--cache", default=".work/dukascopy")
    ap.add_argument("--delay", type=float, default=1.5, help="요청 간 기본 간격(초)")
    ap.add_argument("--tickers", default="", help="쉼표 구분, 비우면 전체")
    ap.add_argument("--commit", action="store_true", help="종목이 끝날 때마다 깃에 올린다")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # 장중(09:30~16:00) 뉴스가 있는 종목·날짜만 받는다.
    need: dict[str, set[str]] = collections.defaultdict(set)
    with gzip.open(args.labels, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["horizon"] == "same_day" and r["published_et"][11:16] >= "09:30":
                need[r["ticker"]].add(r["date"])
    only = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    tickers = sorted((t for t in need if not only or t in only), key=lambda t: -len(need[t]))
    logger.info("대상 %d종목 / %d종목·일", len(tickers), sum(len(need[t]) for t in tickers))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(args.delay)
    summary_path = out / "dukascopy_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for ticker in tickers:
        cache = Path(args.cache) / ticker
        cache.mkdir(parents=True, exist_ok=True)
        missing_file = cache / "_missing.txt"
        missing = set(missing_file.read_text().split()) if missing_file.exists() else set()
        days = sorted(need[ticker])
        todo = [d for d in days if not (cache / f"{d}.csv").exists() and d not in missing]
        logger.info("%s: %d일 중 남은 %d일", ticker, len(days), len(todo))
        consecutive_404 = 0
        t0 = time.time()
        for n, day in enumerate(todo, 1):
            y, m, d = int(day[:4]), int(day[5:7]), int(day[8:10])
            body = fetcher.get(URL.format(sym=ticker, y=y, m=m - 1, d=d))   # 월은 0부터
            if body is None:
                missing.add(day)
                missing_file.write_text("\n".join(sorted(missing)))
                consecutive_404 += 1
                if consecutive_404 >= 8 and not any(cache.glob("*.csv")):
                    logger.info("  %s: Dukascopy 에 없는 종목으로 판단 — 건너뜀", ticker)
                    break
                continue
            consecutive_404 = 0
            rows = decode(body, day)
            tmp = cache / f"{day}.tmp"
            with tmp.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            tmp.rename(cache / f"{day}.csv")
            if n % 50 == 0:
                logger.info("  %s %d/%d (%.1f초/일)", ticker, n, len(todo), (time.time() - t0) / n)
            if args.commit and n % 100 == 0:
                # 오래 걸리므로 중간 진행분도 올린다(컨테이너 회수 대비).
                git_commit(write_year_files(ticker, cache, out),
                           f"분봉 수집(Dukascopy) 진행: {ticker} {n}/{len(todo)}일")
        got = len(list(cache.glob("*.csv")))
        summary[ticker] = {"need": len(days), "got": got, "missing": len(missing)}
        logger.info("%s 완료: %d/%d일 확보, 없음 %d일", ticker, got, len(days), len(missing))
        files = write_year_files(ticker, cache, out) if got else []
        summary_path.write_text(json.dumps(summary, indent=2))
        if args.commit:
            git_commit(files + [str(summary_path)],
                       f"분봉 수집(Dukascopy): {ticker} {got}/{len(days)}일")


if __name__ == "__main__":
    main()
