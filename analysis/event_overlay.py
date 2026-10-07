"""지표 기준선 + 핵심 이벤트 오버레이 (Trade the Event 식 이벤트 조건부 조정).

각 폴드에서
  1) 지표 30일 Ridge(λ=1e6)로 기본 예측을 만들고,
  2) 학습 구간의 핵심 이벤트(주인공 + 새 정보)를 발행 시점(장 시작 전·장중·마감 후) × 방향(좋은·나쁜)
     6개 집단으로 나눠, 기본 예측의 잔차 평균을 n/(n+50) 으로 수축해 집단 효과로 추정한 뒤,
  3) 검증·평가 구간에서 해당 집단 이벤트의 예측에 그 효과를 더한다.
이벤트 연구(EVENT_STUDY.md)에서 확인한 '나쁜 뉴스 후 지속 하락'을 모델에 넣는 가장 단순한 방법이다.

    python analysis/event_overlay.py --out experiments/eventdriven_suite/OVERLAY.md
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_suite as B
import run_eventdriven_suite as ED
import run_event_suite as EV

GROUPS = [(k, d) for k in ("pre", "intra", "after") for d in (-1, 1)]


def group_ids(data):
    g = np.full(len(data.Y), -1)
    for i, (k, d) in enumerate(GROUPS):
        g[(data.kind == k) & data.core & (data.A["direction"] == d)] = i
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    data = ED.EDData("experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz", "data/fnspid/indicators.npz",
                     "data/fnspid/event_features.jsonl.gz", "data/fnspid/events_llm.jsonl.gz")
    g = group_ids(data)
    rows, effects_test = [], None
    for name, tr, ev_ in data.splits():
        x, y = data.features(tr, {"ind": "flat"}), EV.winsor(data.Y, tr)
        base = B.ridge_fit_predict(x[tr], y[tr], x, 1e6)
        resid = y - base
        eff = np.zeros(len(GROUPS))
        for i in range(len(GROUPS)):
            m = tr & (g == i)
            if m.sum():
                eff[i] = resid[m].mean() * m.sum() / (m.sum() + 50)
        over = base + np.where(g >= 0, eff[np.maximum(g, 0)], 0.0)
        idx = np.where(ev_)[0]
        r0, r1 = ED.evaluate(data, idx, base[idx]), ED.evaluate(data, idx, over[idx])
        rows.append((name, r0, r1))
        if name == "test":
            effects_test = eff
    L = ["# 지표 기준선 + 핵심 이벤트 오버레이", "",
         "| 구간 | 기준선 순위IC | 오버레이 순위IC | 기준선 핵심 이벤트 | 오버레이 핵심 이벤트 | 기준선 롱숏 bp (t) | 오버레이 롱숏 bp (t) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r0, r1 in rows:
        L.append(f"| {name} | {r0['rank_ic']:+.4f} | {r1['rank_ic']:+.4f} | {r0['rank_ic_core']:+.3f} | {r1['rank_ic_core']:+.3f} | "
                 f"{r0['ls_mean_bp']:+.1f} ({r0['ls_t']:+.2f}) | {r1['ls_mean_bp']:+.1f} ({r1['ls_t']:+.2f}) |")
    v0 = np.mean([r0["rank_ic"] for n, r0, _ in rows[:-1]]); v1 = np.mean([r1["rank_ic"] for n, _, r1 in rows[:-1]])
    L += ["", f"검증 6개 분기 평균 순위IC: 기준선 {v0:+.4f} → 오버레이 {v1:+.4f}", "",
          "평가용 학습(2023-06-27 까지)에서 추정한 집단 효과 (y_abn_z 단위, 수축 후):", "",
          "| 발행 시점 | 방향 | 효과 | 학습 건수 |", "|---|---:|---:|---:|"]
    _, tr, _ = data.splits()[-1]
    for i, (k, d) in enumerate(GROUPS):
        L.append(f"| {k} | {d:+d} | {effects_test[i]:+.3f} | {int((tr & (g == i)).sum())} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
