"""EDT 학습 모델의 성능이 뉴스 내용에서 오는지 가른다.

같은 분할(학습 ~2020-12 → 검증 2021-01~02 → 학습 ~2021-02 → 평가 2021-03~05)로
특징 묶음을 바꿔 가며 Ridge·LightGBM 을 비교한다.

    A0  발행 시점 + 주가 수준만 (뉴스 내용 없음)
    A1  LLM 이벤트 내용만 (유형·방향·중요도, 시점·주가 수준 없음)
    A2  A0 + A1 (상호작용 포함, edt_study 의 'LLM 이벤트 특징')
    A3  A2 + 임베딩 PCA32
또 평가 구간을 주가 수준·발행 시점 집단 안에서만 순위를 매겨 본다(집단 내 순위 IC 평균):
집단 간 평균 차이를 맞힌 것이 아니라 같은 집단 안에서 뉴스 내용으로 구분하는지를 본다.

    python analysis/edt_ablation.py --spy <SPY 분봉 폴더> --out experiments/edt/ABLATION.md
"""
import argparse
import glob
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edt_study import FAMILY, Spy, load, ls, ric, ridge, TRAIN_END, VALID_END  # noqa: E402
from extract_events import EVENT_TYPES  # noqa: E402


def blocks(rows):
    tid = {t: i for i, t in enumerate(EVENT_TYPES)}
    n = len(rows)
    ctx, content, inter = [], [], []
    for r in rows:
        kind = [r["kind"] == k for k in ("pre", "intra", "after")]
        px = r["start_price"]
        size = [px < 5, 5 <= px < 20, px >= 20]
        hour = datetime.fromisoformat(r["pub_time"]).hour
        ctx.append(kind + size + [np.log(px), hour / 24])
        d, m = r["direction"], r["materiality"]
        oh = np.zeros(len(EVENT_TYPES))
        oh[tid[r["event_type"]]] = 1
        fam = [r["event_type"] in v for v in FAMILY.values()]
        content.append(np.concatenate([[d, m, d * m, d < 0, r["core"], r["price_recap"], r["new_info"], r["about_company"]],
                                       fam, [d * f for f in fam], oh, oh * d]))
        inter.append([d * k for k in kind] + [d * s for s in size] + [(d < 0) * k for k in kind] + [(d < 0) * s for s in size]
                     + [m * s for s in size])
    return np.array(ctx, float), np.array(content, float), np.array(inter, float)


def group_ric(p, y, g):
    vals = [ric(p[g == k], y[g == k]) for k in np.unique(g) if (g == k).sum() > 50]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(Spy(a.spy))
    rows.sort(key=lambda r: r["pub_time"])
    y = np.array([r["a_1d"] if r["a_1d"] is not None else np.nan for r in rows])
    lo, hi = np.nanquantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    days = np.array([r["date"] for r in rows])
    ok = ~np.isnan(y)
    core = np.array([r["core"] for r in rows], bool) & ok
    px = np.array([r["start_price"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    grp = np.array([f"{k}|{0 if p < 5 else (1 if p < 20 else 2)}" for k, p in zip(kind, px)])
    cut = lambda d: (datetime.fromisoformat(d) + timedelta(days=3)).strftime("%Y-%m-%d")
    tr = (days <= TRAIN_END) & core
    va = (days > cut(TRAIN_END)) & (days <= VALID_END) & core
    tr2 = (days <= VALID_END) & core
    te = (days > cut(VALID_END)) & core

    ids = np.load("data/edt/edt_emb_ids.npy")
    vec = np.vstack([np.load(p).astype(np.float32) for p in sorted(glob.glob("data/edt/edt_emb.part*.npy"))])
    pos = {int(i): k for k, i in enumerate(ids)}
    E = np.array([vec[pos[r["edt_id"]]] for r in rows])

    def pca(mask, k=32):
        mu = E[mask].mean(0)
        _, _, vt = np.linalg.svd(E[mask] - mu, full_matrices=False)
        return (E - mu) @ vt[:k].T

    ctx, content, inter = blocks(rows)
    sets = {
        "A0 발행 시점 + 주가 수준만 (뉴스 내용 없음)": lambda m: ctx,
        "A1 LLM 이벤트 내용만": lambda m: content,
        "A2 시점·주가 수준 + LLM 이벤트 + 상호작용": lambda m: np.hstack([ctx, content, inter]),
        "A3 A2 + 임베딩 PCA32": lambda m: np.hstack([ctx, content, inter, pca(m)]),
        "A4 임베딩 PCA32 만": lambda m: pca(m),
    }
    import lightgbm as lgb
    params = {"objective": "huber", "alpha": 0.02, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
              "seed": 0, "num_threads": 4}
    L = ["# EDT 학습 모델 성능의 출처 (제거 실험)", "",
         "핵심 이벤트, 1일 초과수익률. '집단 내 순위IC'는 발행 시점(3) × 주가 수준(3) 9개 집단 안에서 매긴 순위 IC 의 평균이다.",
         "집단 간 평균 차이(예: 저가주가 대체로 하락)가 아니라 같은 집단 안에서 뉴스 내용으로 구분하는 힘을 본다.", "",
         "| 모델 | 검증 순위IC | 평가 순위IC | 평가 집단 내 순위IC | 평가 롱숏 bp | t |", "|---|---:|---:|---:|---:|---:|"]
    for name, make in sets.items():
        for algo in ("Ridge", "LightGBM"):
            Xv, Xt = make(tr), make(tr2)
            if algo == "Ridge":
                lam = max((1, 10, 100, 1e3, 1e4, 1e5), key=lambda l: ric(ridge(Xv[tr], y[tr], Xv[va], l), y[va]))
                pv, pt = ridge(Xv[tr], y[tr], Xv[va], lam), ridge(Xt[tr2], y[tr2], Xt[te], lam)
                tag = f"λ={lam:g}"
            else:
                bst = lgb.train(params, lgb.Dataset(Xv[tr], y[tr]), num_boost_round=600)
                best = max(range(50, 601, 50), key=lambda r: ric(bst.predict(Xv[va], num_iteration=r), y[va]))
                pv = bst.predict(Xv[va], num_iteration=best)
                pt = lgb.train(params, lgb.Dataset(Xt[tr2], y[tr2]), num_boost_round=best).predict(Xt[te])
                tag = f"반복 {best}"
            bp, t, _ = ls(pt, y[te], days[te])
            L.append(f"| {name} · {algo} ({tag}) | {ric(pv, y[va]):+.4f} | {ric(pt, y[te]):+.4f} | "
                     f"{group_ric(pt, y[te], grp[te]):+.4f} | {bp:+.1f} | {t:+.2f} |")
            print(L[-1])
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
