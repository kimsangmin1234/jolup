"""분 단위 주가로 장중 뉴스의 라벨을 '발행 직후 → 당일 종가'로 다시 만든다.

시가→종가 라벨은 장중 뉴스에 대해 시가부터 발행 시각까지의 움직임(기사가 이미
보도한 움직임)을 포함한다. 분봉이 있으면 이 구간을 뺄 수 있다.

    장 시작 전(<09:30) 뉴스  →  당일 시가 → 종가            (변경 없음)
    장중(09:30~16:00) 뉴스   →  발행 다음 분봉 시가 → 종가    (분봉 사용)
    장 마감 후(≥16:00) 뉴스  →  당일 종가 → 다음날 종가        (변경 없음)

장중 진입가는 발행 시각이 속한 분의 **다음** 분봉 시가다. 14:00:xx 발행이면 14:01
봉 시가를 쓴다. 발행 분 안에서 이미 일어난 반응을 라벨에 넣지 않기 위해서다.
종가는 15:59 분봉 종가(정규장 마지막 체결)다. 진입할 분봉이 없으면(15:59 이후
발행 등) 제외한다.

    python apply_minute_labels.py \\
        --labels data/fnspid/labels_open_close.jsonl.gz \\
        --minute-dir data/fnspid/minute \\
        --out data/fnspid/labels_minute.jsonl.gz
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
OPEN, CLOSE = "09:30", "16:00"


def load_session(minute_dir: Path, ticker: str) -> dict[str, tuple[list[str], list[float], list[float]]]:
    """날짜 → (정규장 분 'HH:MM' 목록, 시가 목록, 종가 목록). 시각은 동부시간."""
    days: dict[str, tuple[list, list, list]] = {}
    for path in sorted(minute_dir.glob(f"{ticker}_*.csv.gz")):
        with gzip.open(path, "rt", newline="") as f:
            for row in csv.DictReader(f):
                ts = datetime.fromisoformat(row["t"].replace("Z", "+00:00")).astimezone(ET)
                hm = ts.strftime("%H:%M")
                if not (OPEN <= hm < CLOSE):
                    continue
                d = ts.strftime("%Y-%m-%d")
                mins, opens, closes = days.setdefault(d, ([], [], []))
                mins.append(hm)
                opens.append(float(row["o"]))
                closes.append(float(row["c"]))
    return days


def intraday_label(session, published_hm: str):
    """(라벨, 진입 분) 또는 사유 문자열."""
    mins, opens, closes = session
    if not mins or mins[-1] < "15:55":
        return "정규장불완전"                   # 조기 폐장·데이터 누락
    i = bisect.bisect_right(mins, published_hm)  # 발행 분보다 뒤인 첫 봉
    if i >= len(mins):
        return "진입봉없음"
    entry = opens[i]
    if entry <= 0:
        return "진입가무효"
    return closes[-1] / entry - 1.0, mins[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/fnspid/labels_open_close.jsonl.gz")
    ap.add_argument("--minute-dir", default="data/fnspid/minute")
    ap.add_argument("--out", default="data/fnspid/labels_minute.jsonl.gz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with gzip.open(args.labels, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    minute_dir = Path(args.minute_dir)
    sessions = {t: load_session(minute_dir, t) for t in sorted({r["ticker"] for r in records})}

    stats: collections.Counter = collections.Counter()
    out = []
    for r in records:
        hm = r["published_et"][11:16]
        if r["horizon"] != "same_day" or hm < OPEN:
            stats["변경없음(장 시작 전·마감 후)"] += 1
            out.append(r)
            continue
        session = sessions.get(r["ticker"], {}).get(r["date"])
        if session is None:
            stats["분봉없음_제외"] += 1
            continue
        result = intraday_label(session, hm)
        if isinstance(result, str):
            stats[result + "_제외"] += 1
            continue
        label, entry_hm = result
        out.append({**r, "label": float(label), "entry_et": f"{r['date']} {entry_hm}",
                    "label_def": "entry_to_close"})
        stats["장중_발행후→종가"] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("저장 %d건 → %s", len(out), args.out)
    for k, v in stats.most_common():
        logger.info("  %-24s %7d", k, v)
    lab = np.array([r["label"] for r in out if r.get("label_def") == "entry_to_close"])
    if len(lab):
        logger.info("장중 라벨 평균 %+.3f%% 표준편차 %.2f%%", lab.mean() * 100, lab.std() * 100)


if __name__ == "__main__":
    main()
