"""최종 모델 해석.

1) LightGBM 특징 기여(gain): 문맥(발행 시점·주가 수준) / LLM 이벤트 내용 / 상호작용 / 임베딩 묶음별 비중과 상위 특징.
2) 이벤트 유형별: 최종 앙상블의 평균 예측 백분위와 실제 평균 초과수익률(평가 구간).
3) 개선 구조(P1) 내부: 비대칭 교차 어텐션이 과거 30일 중 어느 날을 보는지(시차별 평균 가중치),
   게이트가 TCN 문맥을 얼마나 받아들이는지(게이트 평균값)를 집단별로 본다. 구간 3 학습 모델(시드 0, 에폭 4).

    python analysis/edt_interpret.py --spy <SPY 분봉 폴더> --out experiments/edt_suite3/INTERPRET.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_edt_suite2 as s2  # noqa: E402
import run_event_suite as ev  # noqa: E402
from run_edt_suite import EDT  # noqa: E402
from edt_study import FAMILY, ric  # noqa: E402
from extract_events import EVENT_TYPES  # noqa: E402

MEMBERS = ["G", "G2", "G3", "G4", "P0", "P1", "R", "V3", "V4", "V5", "V6", "V7"]
CTX = ["장전", "장중", "장후", "주가<5", "주가5~20", "주가≥20", "log주가", "발행시각"]


def names():
    base = ["방향", "중요도", "방향×중요도", "나쁜뉴스", "핵심", "주가재보도", "새정보", "주인공"]
    fam = list(FAMILY)
    content = base + [f"유형군:{f}" for f in fam] + [f"방향×{f}" for f in fam] + \
        [f"유형:{t}" for t in EVENT_TYPES] + [f"방향×{t}" for t in EVENT_TYPES]
    inter = [f"방향×{k}" for k in CTX[:3]] + [f"방향×{k}" for k in CTX[3:6]] + \
        [f"나쁜뉴스×{k}" for k in CTX[:3]] + [f"나쁜뉴스×{k}" for k in CTX[3:6]] + [f"중요도×{k}" for k in CTX[3:6]]
    return content, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = EDT(a.spy)
    folds = s2.fold_masks(d)
    tr, va, tr2, te = folds[-1]
    L = ["# 최종 모델 해석", ""]

    # 1) LightGBM 특징 기여
    import lightgbm as lgb
    content_n, inter_n = names()
    X = np.hstack([d.ctx, d.content, d.inter, d.pca(tr2, 32)])
    cols = CTX + content_n + inter_n + [f"임베딩PC{i + 1}" for i in range(32)]
    assert len(cols) == X.shape[1], (len(cols), X.shape)
    params = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
              "seed": 0, "num_threads": 4}
    bst = lgb.train(params, lgb.Dataset(X[tr2], d.yz[tr2]), num_boost_round=200)
    gain = bst.feature_importance("gain")
    gain = gain / gain.sum()
    blocks = {"문맥 (발행 시점·주가 수준)": slice(0, len(CTX)),
              "LLM 이벤트 내용": slice(len(CTX), len(CTX) + len(content_n)),
              "상호작용 (방향 × 문맥)": slice(len(CTX) + len(content_n), len(CTX) + len(content_n) + len(inter_n)),
              "임베딩 PCA32": slice(len(CTX) + len(content_n) + len(inter_n), len(cols))}
    L += ["## 1. LightGBM 특징 기여 (gain 비중, 구간 3 학습 모델)", "", "| 묶음 | 비중 |", "|---|---:|"]
    for k, sl in blocks.items():
        L.append(f"| {k} | {gain[sl].sum():.1%} |")
    L += ["", "상위 15개 특징:", "", "| 순위 | 특징 | 비중 |", "|---:|---|---:|"]
    for i, j in enumerate(np.argsort(gain)[::-1][:15], 1):
        L.append(f"| {i} | {cols[j]} | {gain[j]:.1%} |")

    # 2) 이벤트 유형별 최종 예측과 실제 수익
    P = {m: np.load(f"experiments/edt_suite3/preds/{m}.npz") for m in MEMBERS}
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    idx, pred = [], []
    for k, (ftr, fva, ftr2, fte) in enumerate(folds):
        w = np.array([max(ric(P[m][f"valid{k}"], d.y[fva]), 0) for m in MEMBERS])
        w = w / w.sum()
        pred.append(sum(wi * rk(P[m][f"test{k}"]) for wi, m in zip(w, MEMBERS)))
        idx.append(np.where(fte)[0])
    idx, pred = np.concatenate(idx), np.concatenate(pred)
    et = np.array([d.rows[i]["event_type"] for i in idx])
    dr = np.array([d.rows[i]["direction"] for i in idx])
    y = d.y[idx]
    L += ["", "## 2. 이벤트 유형·방향별 최종 예측과 실제 초과수익률 (평가 구간 3개, 100건 이상)", "",
          "예측 백분위: 0 = 가장 낮게(하락) 예측, 1 = 가장 높게(상승) 예측. 실제는 1일 초과수익률 평균.", "",
          "| 유형 | 방향 | 건수 | 평균 예측 백분위 | 실제 평균 bp |", "|---|---:|---:|---:|---:|"]
    combos = sorted({(t, s) for t, s in zip(et, dr)}, key=lambda c: -np.mean(pred[(et == c[0]) & (dr == c[1])]))
    for t, s in combos:
        m = (et == t) & (dr == s)
        if m.sum() >= 100:
            L.append(f"| {t} | {s:+d} | {m.sum():,} | {pred[m].mean():.3f} | {y[m].mean() * 1e4:+.1f} |")
    L.append(f"\n유형·방향 평균 예측과 평균 실제 수익의 상관(100건 이상 집단): "
             f"{np.corrcoef([pred[(et == t) & (dr == s)].mean() for t, s in combos if ((et == t) & (dr == s)).sum() >= 100], [y[(et == t) & (dr == s)].mean() for t, s in combos if ((et == t) & (dr == s)).sum() >= 100])[0, 1]:+.2f}")

    # 3) 개선 구조 내부: 어텐션·게이트
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spec = dict(s2.SMALL)
    x, c, w = s2.nn_inputs(d, tr2, spec)
    yt = (d.yz / d.yz[tr2].std()).astype(np.float32)
    model = s2.EDTNet(x.shape[1], c.shape[1], spec["d"])
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["wd"])
    Xt, S, W, Y = (torch.from_numpy(v) for v in (x, d.S, w, yt))
    _, day_codes = np.unique(d.D, return_inverse=True)
    Dc = torch.from_numpy(day_codes)
    for _ in range(4):
        model.train()
        for b in ev.date_batches(day_codes, np.where(tr2)[0], 256, rng):
            bi = torch.from_numpy(b)
            p, _ = model(Xt[bi], S[bi], W[bi])
            loss = nn.functional.mse_loss(p, Y[bi]) + ev.pairwise_rank_loss(p, Y[bi], Dc[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    ti = torch.from_numpy(np.where(te)[0])
    with torch.no_grad():
        prim = model.news(Xt[ti], S[ti])
        ctxv, attw = model.attention(prim, model.tcn(W[ti]))
        fused, gate = model.fusion(prim, ctxv)
        out = model.predictor(fused).numpy()
    attw, gate = attw.numpy(), gate.numpy()
    rows = [d.rows[i] for i in ti.numpy()]
    kind = np.array([r["kind"] for r in rows])
    px = np.array([r["start_price"] for r in rows])
    L += ["", "## 3. 개선 구조 내부 (구간 3 평가 이벤트, 시드 0)", "",
          f"이 모델의 평가 순위 IC: {ric(out, d.y[ti.numpy()]):+.4f}", "",
          "### 비대칭 교차 어텐션: 과거 30일 중 어느 날을 보는가", "",
          "| 시차 (발행 전 거래일) | 1 | 2 | 3 | 4 | 5 | 6~10 | 11~20 | 21~30 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lag = attw[:, ::-1]                       # 0 = 직전 거래일
    cells = [lag[:, i].mean() for i in range(5)] + [lag[:, 5:10].sum(1).mean(), lag[:, 10:20].sum(1).mean(), lag[:, 20:].sum(1).mean()]
    L.append("| 평균 가중치 | " + " | ".join(f"{v:.3f}" for v in cells) + " |")
    ent = -(attw * np.log(attw + 1e-12)).sum(1)
    L.append(f"\n가중치가 고르게 퍼졌을 때(30일 균등)의 엔트로피는 {np.log(30):.2f}, 실제 평균 {ent.mean():.2f}.")
    L += ["", "### 게이트: TCN 문맥(기술적 지표)을 얼마나 받아들이는가", "",
          "게이트 값 0 = 뉴스 주 신호만, 1 = 지표 문맥을 그대로 더함.", "",
          "| 집단 | 건수 | 평균 게이트 |", "|---|---:|---:|"]
    g = gate.mean(1)
    for name, m in (("전체", np.ones(len(g), bool)), ("장 시작 전", kind == "pre"), ("장중", kind == "intra"),
                    ("장 마감 후", kind == "after"), ("주가 $5 미만", px < 5), ("주가 $5~20", (px >= 5) & (px < 20)),
                    ("주가 $20 이상", px >= 20)):
        L.append(f"| {name} | {m.sum():,} | {g[m].mean():.3f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
