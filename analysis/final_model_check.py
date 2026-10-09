"""최종 모델(E6: 12개 모델 검증 비례 앙상블)의 평가 성능 신뢰구간.

세 평가 구간의 예측을 이어 붙여, 같은 거래일 이벤트를 한 묶음으로 복원 추출(2,000회)하는
날짜 블록 부트스트랩으로 순위 IC·집단 내 순위 IC 의 95% 신뢰구간을 낸다. 같은 방식으로
논문 구조(P0)·개선 구조(P1)와의 차이도 본다.

    python analysis/final_model_check.py --spy <SPY 분봉 폴더> --out experiments/edt_suite3/FINAL.md
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_edt_suite2 as s2
from run_edt_suite import EDT
from edt_study import ric
from edt_ablation import group_ric

MEMBERS = ["G", "G2", "G3", "G4", "P0", "P1", "R", "V3", "V4", "V5", "V6", "V7"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spy", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = EDT(a.spy)
    P = {m: np.load(f"experiments/edt_suite3/preds/{m}.npz") for m in MEMBERS}
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    idx, final, p0, p1 = [], [], [], []
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        w = np.array([max(ric(P[m][f"valid{k}"], d.y[va]), 0) for m in MEMBERS]); w = w / w.sum()
        final.append(sum(wi * rk(P[m][f"test{k}"]) for wi, m in zip(w, MEMBERS)))
        p0.append(rk(P["P0"][f"test{k}"])); p1.append(rk(P["P1"][f"test{k}"]))
        idx.append(np.where(te)[0])
    idx = np.concatenate(idx); f, q0, q1 = (np.concatenate(x) for x in (final, p0, p1))
    y, days, grp = d.y[idx], d.D[idx], d.grp[idx]
    ud, inv = np.unique(days, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    rng = np.random.default_rng(0)
    stats = {"E6 순위IC": [], "E6 집단 내": [], "E6 − P0 순위IC": [], "E6 − P1 순위IC": [], "E6 − P0 집단 내": [], "E6 − P1 집단 내": []}
    point = {"E6 순위IC": ric(f, y), "E6 집단 내": group_ric(f, y, grp),
             "E6 − P0 순위IC": ric(f, y) - ric(q0, y), "E6 − P1 순위IC": ric(f, y) - ric(q1, y),
             "E6 − P0 집단 내": group_ric(f, y, grp) - group_ric(q0, y, grp),
             "E6 − P1 집단 내": group_ric(f, y, grp) - group_ric(q1, y, grp)}
    for _ in range(2000):
        s = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        a_, g_ = ric(f[s], y[s]), group_ric(f[s], y[s], grp[s])
        stats["E6 순위IC"].append(a_); stats["E6 집단 내"].append(g_)
        stats["E6 − P0 순위IC"].append(a_ - ric(q0[s], y[s])); stats["E6 − P1 순위IC"].append(a_ - ric(q1[s], y[s]))
        stats["E6 − P0 집단 내"].append(g_ - group_ric(q0[s], y[s], grp[s]))
        stats["E6 − P1 집단 내"].append(g_ - group_ric(q1[s], y[s], grp[s]))
    L = ["# 최종 모델 성능 신뢰구간", "", f"세 평가 구간 이어 붙임: 이벤트 {len(y):,}건, 거래일 {len(ud)}일. 날짜 블록 부트스트랩 2,000회.", "",
         "| 지표 | 값 | 95% 신뢰구간 | 0 이하 확률 |", "|---|---:|---|---:|"]
    for k_, v in stats.items():
        v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5])
        L.append(f"| {k_} | {point[k_]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(v <= 0).mean():.3f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
