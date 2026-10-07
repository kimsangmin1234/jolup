"""뉴스 이벤트 단위 특징과 목표값을 만든다 (선행 연구 기반 설계).

각 뉴스에 대해 '진입 시점'(라벨 구간 시작)을 정하고, 그 시점에 알 수 있는
정보만으로 특징을 만든다.

    장 시작 전(<09:30) 뉴스  진입 = 당일 시가          청산 = 당일 종가
    장중 뉴스               진입 = 발행 다음 분봉 시가  청산 = 당일 종가
    장 마감 후(≥16:00) 뉴스  진입 = 당일 종가          청산 = 다음 거래일 종가

목표값 (이벤트 연구 방법론, MacKinlay 1997)
    y_raw    청산/진입 − 1
    y_abn    y_raw − β·(같은 구간 SPY 수익률). β 는 직전 60거래일 일간 수익률로
             추정해 1 쪽으로 절반 수축한다. 시장 전체 움직임을 빼 종목 고유 반응만 남긴다.
    y_abn_z  y_abn / 직전 20거래일 일간 변동성. 종목 간 크기를 맞추고 극단값 영향을 줄인다.

진입 전 반응 특징 (Chan 2003: 뉴스가 있는 움직임은 지속, 뉴스 없는 움직임은 반전.
Lou·Polk·Skouras 2019: 장중·야간 수익률의 성격 차이)
    gap      전일 종가 → 당일 시가
    pre      당일 시가 → 진입가 (장 시작 전 0, 장 마감 후는 당일 시가→종가)
    prev1    전일 수익률, mom5 직전 5일 수익률
    *_abn    같은 구간 SPY 를 β 만큼 뺀 값, *_z 는 20일 변동성으로 나눈 값
    volr     진입 시점까지 당일 누적 거래량 / 20일 평균 일 거래량

뉴스 특징
    sentiment                 GPT-4o-mini 감성 점수
    novelty                   1 − 같은 종목의 직전 72시간 기사와의 최대 코사인 유사도
                              (Tetlock 2011: 반복 보도는 과잉 반응 후 반전)
    n_prior24, sent_prior24   직전 24시간 같은 종목 기사 수·평균 감성 (뉴스 집중도)
    tod                       진입 시각(정규장 시작 후 분)

    python build_event_features.py --out data/fnspid/event_features.jsonl.gz
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import glob
import gzip
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60


class Bars:
    """종목별 날짜 → (분 배열, 시가, 종가, 거래량). 시각은 동부시간 자정 기준 분."""

    def __init__(self, minute_dir: Path, ticker: str) -> None:
        offset_cache: dict[str, int] = {}
        raw: dict[str, list] = collections.defaultdict(list)
        for path in sorted(minute_dir.glob(f"{ticker}_*.csv.gz")):
            with gzip.open(path, "rt", newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for t, o, h, l, c, v in reader:
                    day = t[:10]
                    off = offset_cache.get(day)
                    if off is None:
                        y, m, d = int(day[:4]), int(day[5:7]), int(day[8:10])
                        off = int(datetime(y, m, d, 12, tzinfo=timezone.utc).astimezone(ET)
                                  .utcoffset().total_seconds() // 60)
                        offset_cache[day] = off
                    minute = int(t[11:13]) * 60 + int(t[14:16]) + off
                    if OPEN_MIN <= minute < CLOSE_MIN:
                        raw[day].append((minute, float(o), float(c), float(v)))
        self.days: dict[str, tuple] = {}
        for day, rows in raw.items():
            rows.sort()
            arr = np.array(rows)
            self.days[day] = (arr[:, 0].astype(int), arr[:, 1], arr[:, 2], arr[:, 3])
        self.dates = sorted(self.days)

    def open(self, day):
        d = self.days.get(day)
        return d[1][0] if d is not None and d[0][0] <= OPEN_MIN + 5 else None

    def close(self, day):
        d = self.days.get(day)
        return d[2][-1] if d is not None and d[0][-1] >= CLOSE_MIN - 5 else None

    def entry(self, day, minute):
        """minute 이후 첫 분봉의 (시가, 그 분, 그때까지 누적 거래량)."""
        d = self.days.get(day)
        if d is None:
            return None
        i = bisect.bisect_right(d[0], minute)
        if i >= len(d[0]):
            return None
        return d[1][i], int(d[0][i]), float(d[3][:i].sum())

    def volume(self, day):
        d = self.days.get(day)
        return float(d[3].sum()) if d is not None else None


def daily_stats(bars: Bars, spy: Bars, cal: list[str]):
    """날짜 → (전일 종가, 20일 변동성, 60일 β, 전일 수익률, 5일 수익률, 20일 평균 거래량). 모두 d−1 까지."""
    closes = np.array([bars.close(d) or np.nan for d in cal])
    spy_c = np.array([spy.close(d) or np.nan for d in cal])
    vols = np.array([bars.volume(d) or np.nan for d in cal])
    r = closes[1:] / closes[:-1] - 1
    rm = spy_c[1:] / spy_c[:-1] - 1
    out = {}
    for i in range(61, len(cal)):
        # r[k] 는 cal[k] → cal[k+1] 수익률. d=cal[i] 기준 과거는 r[:i-1] (cal[i-1] 종가까지)
        past, pm = r[i - 61:i - 1], rm[i - 61:i - 1]
        ok = np.isfinite(past) & np.isfinite(pm)
        if ok.sum() < 40 or not np.isfinite(closes[i - 1]):
            continue
        p, q = past[ok], pm[ok]
        beta = 0.5 * (np.cov(p, q)[0, 1] / q.var()) + 0.5
        vol20 = np.nanstd(r[i - 21:i - 1])
        out[cal[i]] = {
            "prev_close": closes[i - 1], "spy_prev_close": spy_c[i - 1],
            "vol20": vol20, "beta": beta, "prev1": r[i - 2],
            "mom5": closes[i - 1] / closes[i - 6] - 1 if np.isfinite(closes[i - 6]) else np.nan,
            "avgvol20": np.nanmean(vols[i - 20:i]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/fnspid/labels_minute.jsonl.gz")
    ap.add_argument("--cache", default="experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz")
    ap.add_argument("--minute-dir", default="data/fnspid/minute")
    ap.add_argument("--out", default="data/fnspid/event_features.jsonl.gz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with gzip.open(args.labels, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    tickers = sorted({r["ticker"] for r in records})
    minute_dir = Path(args.minute_dir)
    spy = Bars(minute_dir, "SPY")
    cal = spy.dates
    nxt = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}
    logger.info("SPY 거래일 %d일", len(cal))
    bars, stats = {}, {}
    for t in tickers:
        bars[t] = Bars(minute_dir, t)
        stats[t] = daily_stats(bars[t], spy, cal)
        logger.info("  %s 분봉 %d일", t, len(bars[t].dates))

    # 뉴스 특징: 감성, 새로움, 집중도 ------------------------------------------
    emb, sent = {}, {}
    for p in sorted(glob.glob(args.cache)):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                v = np.asarray(r["embedding"], dtype=np.float32)
                emb[r["news_id"]] = v / (np.linalg.norm(v) + 1e-8)
                sent[r["news_id"]] = float(r["sentiment"])
    news_feat = {}
    by_ticker = collections.defaultdict(list)
    for r in records:
        by_ticker[r["ticker"]].append((datetime.fromisoformat(r["published_et"]), r["news_id"]))
    for t, items in by_ticker.items():
        items.sort()
        times = [x[0] for x in items]
        for i, (ts, nid) in enumerate(items):
            lo72 = bisect.bisect_left(times, ts - timedelta(hours=72))
            lo24 = bisect.bisect_left(times, ts - timedelta(hours=24))
            prev72 = [items[j][1] for j in range(lo72, i)]
            prev24 = [items[j][1] for j in range(lo24, i)]
            sim = max((float(emb[nid] @ emb[p]) for p in prev72), default=0.0)
            news_feat[nid] = {"novelty": 1.0 - sim, "n_prior24": len(prev24),
                              "sent_prior24": float(np.mean([sent[p] for p in prev24])) if prev24 else 0.0}

    # 이벤트별 진입·청산과 특징 ------------------------------------------------
    out, drop = [], collections.Counter()
    for r in records:
        t, day = r["ticker"], r["date"]
        b, st = bars[t], stats[t].get(day)
        hm = r["published_et"][11:16]
        pub = int(hm[:2]) * 60 + int(hm[3:])
        if st is None or day not in spy.days:
            drop["통계없음"] += 1
            continue
        o, so = b.open(day), spy.open(day)
        if o is None or so is None:
            drop["시가없음"] += 1
            continue
        if pub < OPEN_MIN:
            kind, e_px, s_e, e_min, cumv = "pre", o, so, OPEN_MIN, 0.0
            x_px, s_x = b.close(day), spy.close(day)
        elif pub < CLOSE_MIN:
            ent, sent_ = b.entry(day, pub), spy.entry(day, pub)
            if ent is None or sent_ is None:
                drop["진입봉없음"] += 1
                continue
            kind, (e_px, e_min, cumv), s_e = "intra", ent, sent_[0]
            x_px, s_x = b.close(day), spy.close(day)
        else:
            nd = nxt.get(day)
            kind, e_px, s_e, e_min, cumv = "after", b.close(day), spy.close(day), CLOSE_MIN, b.volume(day) or 0.0
            x_px, s_x = (b.close(nd), spy.close(nd)) if nd else (None, None)
        if not all(v is not None and v > 0 for v in (e_px, s_e, x_px, s_x)):
            drop["가격없음"] += 1
            continue
        beta, vol = st["beta"], max(st["vol20"], 1e-4)
        y_raw = x_px / e_px - 1
        y_abn = y_raw - beta * (s_x / s_e - 1)
        gap = o / st["prev_close"] - 1
        gap_m = so / st["spy_prev_close"] - 1
        pre = e_px / o - 1
        pre_m = s_e / so - 1
        f = {
            "news_id": r["news_id"], "ticker": t, "date": day, "published_et": r["published_et"],
            "horizon": r["horizon"], "anchor": r["anchor"], "kind": kind,
            "label": y_abn / vol,                         # 기본 목표값(apply_label_file 호환)
            "y_raw": y_raw, "y_abn": y_abn, "y_abn_z": y_abn / vol,
            "beta": beta, "vol20": vol, "tod": e_min - OPEN_MIN,
            "gap": gap, "pre": pre, "gap_abn": gap - beta * gap_m, "pre_abn": pre - beta * pre_m,
            "prev1": st["prev1"], "mom5": st["mom5"],
            "volr": cumv / st["avgvol20"] if st["avgvol20"] > 0 else 0.0,
            "sentiment": sent.get(r["news_id"], 0.0), **news_feat[r["news_id"]],
        }
        for k in ("gap_abn", "pre_abn", "prev1", "mom5"):
            f[k + "_z"] = f[k] / vol
        if not all(np.isfinite(v) for v in f.values() if isinstance(v, float)):
            drop["비유한값"] += 1
            continue
        out.append(f)

    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for f in out:
            fh.write(json.dumps(f) + "\n")
    logger.info("저장 %d건 → %s (제외 %s)", len(out), args.out, dict(drop))
    for kind in ("pre", "intra", "after"):
        ys = np.array([f["y_abn_z"] for f in out if f["kind"] == kind])
        logger.info("  %-5s %6d건  y_abn_z 평균 %+.3f 표준편차 %.3f", kind, len(ys), ys.mean(), ys.std())


if __name__ == "__main__":
    main()
