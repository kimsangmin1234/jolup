"""최종 모델의 거래비용·유동성 검증.

롱숏 수익은 거래비용 전이고 저가주 비중이 높다. 다음을 본다.
  1) 주가·거래대금 필터: 주가 $5 이상·$20 이상, 직전 20일 평균 거래대금 $1M·$10M 이상 집단에서도
     순위 IC 와 롱숏이 유지되는가.
  2) 비용 차감: Corwin & Schultz (2012, JF) 의 고가·저가 기반 호가 스프레드 추정치(직전 20일 평균)를
     종목별 왕복 비용으로 보고, 롱숏의 각 이벤트 수익에서 뺀다. 매수·매도 양쪽 모두 비용을 낸다.
  3) 손익분기 비용: 롱숏 총수익이 0 이 되는 이벤트당 왕복 비용.

    python analysis/edt_robustness.py --spy <SPY 분봉 폴더> --out experiments/edt_suite3/ROBUSTNESS.md
"""
import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_edt_suite2 as s2  # noqa: E402
from run_edt_suite import EDT  # noqa: E402
from edt_study import ric  # noqa: E402
from edt_ablation import group_ric  # noqa: E402

MEMBERS = ["G", "G2", "G3", "G4", "P0", "P1", "R", "V3", "V4", "V5", "V6", "V7"]


def cs_spread(h, l):
    """Corwin–Schultz 스프레드(이틀 단위 추정의 평균). 음수는 0 으로 둔다."""
    k = 3 - 2 * np.sqrt(2)
    out = []
    for t in range(len(h) - 1):
        beta = np.log(h[t] / l[t]) ** 2 + np.log(h[t + 1] / l[t + 1]) ** 2
        gamma = np.log(max(h[t], h[t + 1]) / min(l[t], l[t + 1])) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        out.append(max(s, 0.0))
    return float(np.mean(out)) if out else np.nan


def daily_stats(rows):
    bars = defaultdict(list)
    with gzip.open("data/edt/daily_bars.csv.gz", "rt", newline="") as f:
        r = csv.reader(f)
        next(r)
        for t, d, o, h, l, c, v in r:
            bars[t].append((d, float(h), float(l), float(c), float(v)))
    for t in bars:
        bars[t].sort()
    adv, spr = [], []
    for row in rows:
        b = bars.get(row["ticker"], [])
        dates = [x[0] for x in b]
        j = np.searchsorted(dates, row["date"])          # 발행일 전 거래일까지
        win = b[max(0, j - 20):j]
        if len(win) < 10:
            adv.append(np.nan); spr.append(np.nan)
            continue
        adv.append(np.mean([x[3] * x[4] for x in win]))
        hl = [(x[1], x[2]) for x in win if x[1] > 0 and x[2] > 0]
        spr.append(cs_spread([x[0] for x in hl], [x[1] for x in hl]))
    return np.array(adv), np.array(spr)


def ls_detail(pred, y, cost, days):
    """날짜별 예측 상위 1/3 매수 − 하위 1/3 매도. (총수익 bp, t, 순수익 bp, t, 이벤트당 평균 비용 bp)."""
    gross, net, costs = defaultdict(list), defaultdict(list), []
    for day in np.unique(days):
        m = days == day
        if m.sum() < 3:
            continue
        p, r, c = pred[m], y[m], cost[m]
        lo, hi = np.quantile(p, [1 / 3, 2 / 3])
        sgn = np.where(p >= hi, 1, np.where(p <= lo, -1, 0))
        sel = sgn != 0
        if not sel.any():
            continue
        gross[day] = (sgn[sel] * r[sel]).mean()
        net[day] = (sgn[sel] * r[sel] - c[sel]).mean()
        costs += list(c[sel])

    def mt(v):
        v = np.array(list(v.values()))
        return v.mean() * 1e4, v.mean() / v.std(ddof=1) * np.sqrt(len(v))
    g, gt = mt(gross)
    n, nt = mt(net)
    return g, gt, n, nt, np.mean(costs) * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = EDT(a.spy)
    P = {m: np.load(f"experiments/edt_suite3/preds/{m}.npz") for m in MEMBERS}
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    idx, pred = [], []
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        w = np.array([max(ric(P[m][f"valid{k}"], d.y[va]), 0) for m in MEMBERS])
        w = w / w.sum()
        pred.append(sum(wi * rk(P[m][f"test{k}"]) for wi, m in zip(w, MEMBERS)))
        idx.append(np.where(te)[0])
    idx = np.concatenate(idx)
    pred = np.concatenate(pred)
    rows = [d.rows[i] for i in idx]
    y, days, grp = d.y[idx], d.D[idx], d.grp[idx]
    px = np.array([r["start_price"] for r in rows])
    adv, spr = daily_stats(rows)
    spr = np.where(np.isnan(spr), np.nanmedian(spr), spr)

    L = ["# 최종 모델의 거래비용·유동성 검증", "",
         f"대상: 세 평가 구간의 핵심 이벤트 {len(y):,}건, 최종 모델(12개 모델 검증 비례 앙상블) 예측.",
         "비용: Corwin & Schultz(2012) 고가·저가 스프레드 추정치(발행 전 20거래일 평균)를 종목별 왕복 비용으로 보고",
         "롱숏의 매수·매도 이벤트 모두에서 뺐다. 손익분기 비용은 총수익을 0 으로 만드는 이벤트당 왕복 비용이다.", "",
         "| 집단 | 이벤트 | 순위IC | 집단 내 | 롱숏 총수익 bp/일 | t | 평균 추정 비용 bp | 비용 차감 후 bp/일 | t | 손익분기 비용 bp |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    groups = [("전체", np.ones(len(y), bool)), ("주가 $5 이상", px >= 5), ("주가 $20 이상", px >= 20),
              ("거래대금 $1M 이상", adv >= 1e6), ("거래대금 $10M 이상", adv >= 1e7),
              ("주가 $5 이상 + 거래대금 $10M 이상", (px >= 5) & (adv >= 1e7)),
              ("주가 $5 미만", px < 5)]
    for name, m in groups:
        g, gt, n, nt, c = ls_detail(pred[m], y[m], spr[m], days[m])
        L.append(f"| {name} | {m.sum():,} | {ric(pred[m], y[m]):+.4f} | {group_ric(pred[m], y[m], grp[m]):+.4f} | "
                 f"{g:+.1f} | {gt:+.2f} | {c:.1f} | {n:+.1f} | {nt:+.2f} | {g:.1f} |")
        print(L[-1])
    L += ["", "주: 롱숏은 매일 이벤트가 발생한 종목만으로 구성한 이벤트 포트폴리오이며, 같은 날 매수·매도 이벤트 수가 달라",
          "시장 중립이 아닐 수 있다. 수익은 SPY 를 뺀 초과수익률 기준이다."]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
