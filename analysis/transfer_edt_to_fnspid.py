"""전이 검증: EDT(보도자료, 2020-03~2021-05)로 학습한 모델을 FNSPID(뉴스 기사, 2020-12~2023-12)에 그대로 적용한다.

다른 기간·다른 출처·다른 종목 구성에서도 예측력이 남는지 보는 가장 강한 검증이다. FNSPID 쪽에서는 아무것도 학습하지 않는다.

입력을 맞춘 방법
    문맥      발행 시점(장 시작 전·장중·마감 후), 주가 수준(직전 거래일 종가), 발행 시각
    LLM 이벤트 같은 프롬프트로 뽑은 유형·방향·중요도·주인공·새 정보·재보도 (extract_events.py)
    임베딩    본문 앞 800자 text-embedding-3-small (EDT 는 제목 + 본문 앞 800자). PCA 는 EDT 로 맞춘다
    지표 창   발행일 전 거래일까지 30일, 9종 지표를 창 기준 정규화 후 EDT 통계로 표준화
목표값: FNSPID 이벤트의 발행 이후 시장 조정 초과수익률 / 20일 변동성 (event_features 의 y_abn_z).

모델 (EDT 전체로 학습)
    R   Ridge (문맥 + LLM + 상호작용 + 임베딩 PCA32)
    G   LightGBM 시드 5개 평균 (같은 특징)
    P1  개선 구조 신경망 (시드 3개)
    E   세 모델 예측 순위의 같은 가중치 평균
비교 기준(학습 없음): LLM 방향 규칙, 나쁜 뉴스만 매도 규칙.
평가: FNSPID 핵심 이벤트(주인공 + 새 정보). 겹치는 기간(~2021-05)과 이후 기간(2021-06~)을 나눠 본다.

    python analysis/transfer_edt_to_fnspid.py --spy <SPY 분봉 폴더> --out experiments/transfer/RESULTS.md
"""
import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_edt_suite2 as s2  # noqa: E402
import run_event_suite as ev  # noqa: E402
import run_suite as base  # noqa: E402
from data.dataset import load_indicators  # noqa: E402
from edt_ablation import blocks, group_ric  # noqa: E402
from edt_study import ric, ridge  # noqa: E402
from run_edt_suite import EDT  # noqa: E402

SPLIT = "2021-06-01"


def fnspid_rows():
    with gzip.open("data/fnspid/events_llm.jsonl.gz", "rt") as f:
        llm = {r["news_id"]: r for r in map(json.loads, f)}
    ids = json.loads(Path("data/fnspid/article_emb_ids.json").read_text())
    vec = np.vstack([np.load(p).astype(np.float32) for p in sorted(Path("data/fnspid").glob("article_emb.part*.npy"))])
    epos = {n: i for i, n in enumerate(ids)}
    ind, idx = load_indicators("data/fnspid/indicators.npz")
    rows, E, W = [], [], []
    with gzip.open("data/fnspid/event_features.jsonl.gz", "rt") as f:
        for r in map(json.loads, f):
            e = llm.get(r["news_id"])
            j = idx.get(r["ticker"], {}).get(r["date"])
            if e is None or j is None or j < 31 or r["news_id"] not in epos:
                continue
            v = vec[epos[r["news_id"]]]
            if not np.abs(v).sum():
                continue
            core = int(e["about_company"] and e["new_info"])
            if not core:
                continue
            row = {**r, **e, "core": core, "pub_time": r["published_et"],
                   "start_price": float(ind[r["ticker"]][j - 1, 0])}      # 직전 거래일 종가
            rows.append(row)
            E.append(v)
            W.append(ind[r["ticker"]][j - 30:j])                          # 발행일 전 거래일까지 30일
    return rows, np.array(E), base.window_normalize(np.stack(W))


def ls(pred, y, days):
    by = {}
    for day in np.unique(days):
        m = days == day
        if m.sum() < 3:
            continue
        lo, hi = np.quantile(pred[m], [1 / 3, 2 / 3])
        s = np.where(pred[m] >= hi, 1, np.where(pred[m] <= lo, -1, 0))
        if hi == lo:
            s = np.sign(pred[m])
        if (s != 0).any():
            by[day] = (s[s != 0] * y[m][s != 0]).mean()
    v = np.array(list(by.values()))
    return v.mean() * 1e4, v.mean() / v.std(ddof=1) * np.sqrt(len(v)), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    d = EDT(a.spy)
    tr = np.ones(len(d.y), bool)
    frows, FE, FW = fnspid_rows()
    fctx, fcont, finter = blocks(frows)
    fy = np.array([r["y_abn_z"] for r in frows])
    fya = np.array([r["y_abn"] for r in frows])
    fdays = np.array([r["date"] for r in frows])
    fkind = np.array([r["kind"] for r in frows])
    fpx = np.array([r["start_price"] for r in frows])
    fgrp = np.array([f"{k}|{0 if p < 5 else (1 if p < 20 else 2)}" for k, p in zip(fkind, fpx)])
    fS = np.array([r["direction"] * r["materiality"] / 3 for r in frows], dtype=np.float32)
    fdir = np.array([r["direction"] for r in frows], float)

    # EDT 로 PCA
    mu = d.E.mean(0)
    _, _, vt = np.linalg.svd(d.E - mu, full_matrices=False)
    pca = lambda E, k: (E - mu) @ vt[:k].T

    # R, G
    Xe = np.hstack([d.ctx, d.content, d.inter, pca(d.E, 32)])
    Xf = np.hstack([fctx, fcont, finter, pca(FE, 32)])
    preds = {"R  Ridge": ridge(Xe, d.yz, Xf, 1000.0)}
    import lightgbm as lgb
    gp = []
    for seed in range(5):
        params = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 100,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
                  "seed": seed, "num_threads": 4}
        gp.append(lgb.train(params, lgb.Dataset(Xe, d.yz), num_boost_round=200).predict(Xf))
    preds["G  LightGBM (시드 5)"] = np.mean(gp, 0)

    # P1: EDT 통계로 표준화
    xe = np.hstack([pca(d.E, 64), d.content, d.inter, d.ctx])
    xf = np.hstack([pca(FE, 64), fcont, finter, fctx])
    m, s = xe.mean(0), xe.std(0) + 1e-8
    xe, xf = ((xe - m) / s).astype(np.float32), ((xf - m) / s).astype(np.float32)
    wm, ws = d.W.mean((0, 1)), d.W.std((0, 1)) + 1e-8
    we, wf = ((d.W - wm) / ws).astype(np.float32), ((FW - wm) / ws).astype(np.float32)
    ye = (d.yz / d.yz.std()).astype(np.float32)
    _, day_codes = np.unique(d.D, return_inverse=True)
    pp = []
    for seed in range(3):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = s2.EDTNet(xe.shape[1], d.ctx.shape[1], 64)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        X, S, Wt, Y, Dc = (torch.from_numpy(v) for v in (xe, d.S, we, ye, day_codes))
        for _ in range(4):
            model.train()
            for b in ev.date_batches(day_codes, np.arange(len(ye)), 256, rng):
                bi = torch.from_numpy(b)
                p, _ = model(X[bi], S[bi], Wt[bi])
                loss = nn.functional.mse_loss(p, Y[bi]) + ev.pairwise_rank_loss(p, Y[bi], Dc[bi])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pp.append(model(torch.from_numpy(xf), torch.from_numpy(fS), torch.from_numpy(wf))[0].numpy())
    preds["P1 개선 구조 (시드 3)"] = np.mean(pp, 0)
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    preds["E  앙상블 (R+G+P1 같은 가중치)"] = (rk(preds["R  Ridge"]) + rk(preds["G  LightGBM (시드 5)"]) + rk(preds["P1 개선 구조 (시드 3)"])) / 3
    preds["규칙: LLM 방향"] = fdir
    preds["규칙: 나쁜 뉴스만 매도"] = -(fdir < 0).astype(float)

    L = ["# 전이 검증: EDT 로 학습 → FNSPID 에 그대로 적용", "",
         f"FNSPID 핵심 이벤트 {len(fy):,}건 (겹치는 기간 ~2021-05 {(fdays < SPLIT).sum():,}건 / 이후 2021-06~ {(fdays >= SPLIT).sum():,}건).",
         "FNSPID 에서는 아무것도 학습하지 않았다. 목표값은 발행 이후 시장 조정 초과수익률 / 20일 변동성.",
         "롱숏: 날짜별 예측 상위 1/3 매수 − 하위 1/3 매도(이벤트 3건 이상인 날), 시장 조정 초과수익률, 거래비용 미반영.", ""]
    for title, mask in (("이후 기간 (2021-06 ~ 2023-12): 시간상 완전한 표본 외", fdays >= SPLIT),
                        ("겹치는 기간 (2020-12 ~ 2021-05)", fdays < SPLIT), ("전체", np.ones(len(fy), bool))):
        L += [f"## {title}", "",
              "| 모델 | 순위IC | 집단 내 | 장 시작 전 | 장중 | 장 마감 후 | 롱숏 bp/일 | t |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, p in preds.items():
            pm, ym = p[mask], fy[mask]
            cells = [ric(pm[fkind[mask] == k], ym[fkind[mask] == k]) for k in ("pre", "intra", "after")]
            bp, t, _ = ls(pm, fya[mask], fdays[mask])
            L.append(f"| {name} | {ric(pm, ym):+.4f} | {group_ric(pm, ym, fgrp[mask]):+.4f} | "
                     + " | ".join(f"{c:+.3f}" for c in cells) + f" | {bp:+.1f} | {t:+.2f} |")
        L.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
