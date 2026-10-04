"""walk-forward 교차검증으로 임베딩 축소 차원을 고른다.

폴드마다 학습 구간에서만 PCA 를 맞추고(미래 정보 누수 방지), 축소한 임베딩에
감성 점수를 붙여 Ridge 회귀로 다음 분기를 예측한다. 검증 6개 분기의 평균 IC
(예측과 실제 등락률의 상관계수)로 차원과 규제 강도를 고른 뒤, 한 번도 쓰지
않은 2023 하반기에서 1회 평가한다.

    python analysis/select_embedding_dim.py \\
        --cache "experiments/fnspid-server/news_cache_timed.part*.jsonl.gz" \\
        --out experiments/diagnosis/EMBED_DIM.md
"""
from __future__ import annotations

import argparse, glob, gzip, json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

FOLDS = [("2022-01-01", "2022-03-31"), ("2022-04-01", "2022-06-30"),
         ("2022-07-01", "2022-09-30"), ("2022-10-01", "2022-12-31"),
         ("2023-01-01", "2023-03-31"), ("2023-04-01", "2023-06-30")]
TEST = ("2023-07-01", "2023-12-31")
EMBARGO_DAYS = 3          # 검증 시작 직전 데이터는 학습에서 뺀다(다음날 라벨 누수 방지)
DIMS = [0, 4, 8, 16, 32, 64, 128, 256, 1536]
LAMBDAS = [1, 10, 100, 1e3, 1e4, 1e5]


def load(pattern, subset):
    meta, emb = [], []
    for p in sorted(glob.glob(pattern)):
        for line in gzip.open(p, "rt", encoding="utf-8"):
            r = json.loads(line)
            if subset == "same_day" and r.get("horizon") != "same_day":
                continue
            e = r.pop("embedding")
            meta.append((r["date"], float(r["sentiment"]), float(r["label"])))
            emb.append(np.asarray(e, dtype=np.float32))
    d = np.array([m[0] for m in meta]); s = np.array([m[1] for m in meta]); y = np.array([m[2] for m in meta])
    return d, s, y, np.vstack(emb)


def ic(p, t):
    return float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-12 else 0.0


def pca_basis(Etr):
    """학습 구간 임베딩의 평균과 주성분(분산 큰 순)."""
    mu = Etr.mean(0)
    w, V = np.linalg.eigh(np.cov((Etr - mu).T))
    order = np.argsort(w)[::-1]
    return mu, V[:, order], w[order]


def fit_predict(Etr, str_, ytr, Ete, ste, k, lam, basis=None):
    """학습 구간으로 PCA·표준화·Ridge 를 맞추고 평가 구간을 예측한다."""
    if k > 0:
        mu, V, _ = basis if basis is not None else pca_basis(Etr)
        P = V[:, :k]
        Xtr = np.hstack([(Etr - mu) @ P, str_[:, None]])
        Xte = np.hstack([(Ete - mu) @ P, ste[:, None]])
    else:
        Xtr, Xte = str_[:, None], ste[:, None]
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - m) / sd, (Xte - m) / sd
    ym = ytr.mean()
    wv = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
    return Xte @ wv + ym


def run(d, s, y, E):
    folds = []
    for vs, ve in FOLDS:
        cut = (date.fromisoformat(vs) - timedelta(days=EMBARGO_DAYS)).isoformat()
        tr, va = d < cut, (d >= vs) & (d <= ve)
        if tr.sum() < 500 or va.sum() < 100:
            continue
        folds.append((tr, va, pca_basis(E[tr])))
    res = {}
    for k in DIMS:
        best = None
        for lam in LAMBDAS:
            ics = [ic(fit_predict(E[tr], s[tr], y[tr], E[va], s[va], k, lam, b), y[va])
                   for tr, va, b in folds]
            score = float(np.mean(ics))
            if best is None or score > best[0]:
                best = (score, lam, ics)
        res[k] = best
    return res, folds[-1][2][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["# 임베딩 축소 차원 선택 (walk-forward)", "",
             "검증 6개 분기(2022 1분기 ~ 2023 2분기)의 평균 IC 로 고른다. 폴드마다 학습 구간에서만",
             f"PCA 를 맞추고, 검증 시작 {EMBARGO_DAYS}일 전 데이터는 학습에서 뺐다. 차원 0 은 감성만 쓴 경우다.", ""]
    for subset, title in (("same_day", "장중 뉴스 → 당일 등락률"), ("all", "전체")):
        d, s, y, E = load(a.cache, subset)
        E = E.astype(np.float64)
        res, eigval = run(d, s, y, E)
        cum = np.cumsum(eigval) / eigval.sum()
        lines += [f"## {title} ({len(y):,}건)", "",
                  "| 차원 | 설명 분산 | 검증 평균 IC | 분기별 IC | 최적 λ |", "|---:|---:|---:|---|---:|"]
        for k in DIMS:
            sc, lam, ics = res[k]
            ev = f"{cum[k-1]:.1%}" if k > 0 else "-"
            lines.append(f"| {k} | {ev} | {sc:+.4f} | {' '.join(f'{x:+.3f}' for x in ics)} | {lam:g} |")
        kbest = max(DIMS, key=lambda k: res[k][0])
        lam = res[kbest][1]
        tr = d < (date.fromisoformat(TEST[0]) - timedelta(days=EMBARGO_DAYS)).isoformat()
        te = (d >= TEST[0]) & (d <= TEST[1])
        p = fit_predict(E[tr], s[tr], y[tr], E[te], s[te], kbest, lam)
        t = y[te]
        rk = lambda v: np.argsort(np.argsort(v))
        lines += ["", f"**선택: {kbest}차원** (λ={lam:g}) → 2023 하반기 평가 {te.sum():,}건: "
                  f"IC {ic(p,t):+.4f}, 순위 IC {ic(rk(p).astype(float),rk(t).astype(float)):+.4f}, "
                  f"방향 적중 {(np.sign(p)==np.sign(t)).mean():.3f}", ""]
        print("\n".join(lines[-len(DIMS)-6:]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
