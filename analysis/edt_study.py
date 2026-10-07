"""EDT 보도자료로 이벤트 신호를 재현·학습한다.

1. 재현: FNSPID 에서 찾은 규칙(LLM 방향, 나쁜 뉴스)을 전혀 다른 데이터(보도자료, 2020-03 ~ 2021-05,
   7천여 종목)에 학습 없이 그대로 적용한다. 규칙은 EDT 를 보기 전에 정했으므로 완전한 표본 외 검증이다.
2. 이질성: 발행 시점(장 시작 전·장중·장 마감 후), 주가 수준(소형주 대용), 보유 기간(1·2·3일).
3. 학습: 표본이 충분하므로 LLM 이벤트 특징·임베딩으로 Ridge·LightGBM·MLP 를 학습해 규칙과 비교한다.
   분할: 학습 ~2020-12 → 검증 2021-01~02 로 설정 선택 → 학습 ~2021-02 로 다시 맞춰 2021-03~05 평가(3일 엠바고).

초과수익률 = 종목 수익률 − SPY 수익률(같은 시각 구간, 시간외 포함 분봉). 1·99% 로 윈저화.

    python analysis/edt_study.py --spy <SPY 분봉 폴더> --out experiments/edt/RESULTS.md
"""
import argparse
import bisect
import csv
import glob
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_events import EVENT_TYPES  # noqa: E402

FAMILY = {
    "earnings": ("earnings_beat", "earnings_miss", "earnings_inline"),
    "guidance": ("guidance_raise", "guidance_cut"),
    "analyst": ("analyst_upgrade", "analyst_downgrade", "price_target_change"),
    "deal": ("acquisition_merger", "new_contract_partnership"),
    "product": ("product_launch", "regulatory_approval_clinical"),
    "risk": ("regulatory_legal_risk", "lawsuit_settlement", "operational_issue", "layoffs_restructuring"),
    "capital": ("dividend_increase", "dividend_cut", "share_buyback", "stock_split", "equity_offering",
                "insider_institutional_trade"),
    "people": ("management_change",),
}
TRAIN_END, VALID_END, TEST_END = "2020-12-31", "2021-02-28", "2021-06-01"


def utc(s):
    return datetime.fromisoformat(s).astimezone(timezone.utc)


class Spy:
    def __init__(self, folder):
        rows = []
        for p in sorted(glob.glob(f"{folder}/SPY_*.csv.gz")):
            with gzip.open(p, "rt", newline="") as f:
                r = csv.reader(f)
                next(r)
                rows += [(datetime.fromisoformat(t.replace("Z", "+00:00")), float(c)) for t, o, h, l, c, v in r]
        rows.sort()
        self.t = [x[0] for x in rows]
        self.c = [x[1] for x in rows]

    def at(self, ts):
        i = bisect.bisect_right(self.t, ts) - 1
        return self.c[i] if i >= 0 and ts - self.t[i] < timedelta(hours=72) else None


def load(spy):
    ev = {}
    with gzip.open("data/edt/edt_events.jsonl.gz", "rt") as f:
        for r in map(json.loads, f):
            ev[r["edt_id"]] = r
    with gzip.open("data/edt/edt_llm.jsonl.gz", "rt") as f:
        llm = {r["edt_id"]: r for r in map(json.loads, f)}
    rows = []
    for eid, r in ev.items():
        if eid not in llm:
            continue
        st = utc(r["start_time"])
        s0 = spy.at(st)
        rec = {**r, **llm[eid]}
        ok = s0 is not None
        for h in (1, 2, 3):
            et = r.get(f"end_time_{h}d")
            s1 = spy.at(utc(et)) if et else None
            if r.get(f"r_{h}d") is None or s1 is None or not ok:
                rec[f"a_{h}d"] = None
            else:
                rec[f"a_{h}d"] = r[f"r_{h}d"] - (s1 / s0 - 1)
        et_local = datetime.fromisoformat(r["pub_time"])
        hm = et_local.strftime("%H:%M")
        rec["kind"] = "pre" if hm < "09:30" else ("intra" if hm < "16:00" else "after")
        rec["date"] = et_local.strftime("%Y-%m-%d")
        rec["core"] = int(rec["about_company"] and rec["new_info"])
        rows.append(rec)
    return rows


def tstat_by_day(vals, days):
    by = {}
    for v, d in zip(vals, days):
        by.setdefault(d, []).append(v)
    x = np.array([np.mean(v) for v in by.values()])
    return x.mean(), (x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan), len(x)


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def ric(p, y):
    return float(np.corrcoef(rank(p), rank(y))[0, 1]) if len(p) > 10 and np.std(p) > 0 else float("nan")


def ls(pred, y, days):
    """예측 상위 1/3 매수 − 하위 1/3 매도, 날짜별 평균 → 평균(bp)·t."""
    lo, hi = np.quantile(pred, [1 / 3, 2 / 3])
    sign = np.sign(pred) if hi == lo else np.where(pred >= hi, 1, np.where(pred <= lo, -1, 0))
    m = sign != 0
    mean, t, n = tstat_by_day(sign[m] * y[m], days[m])
    return mean * 1e4, t, n


def features(rows, emb=None, use_emb=False):
    tid = {t: i for i, t in enumerate(EVENT_TYPES)}
    X = []
    for r in rows:
        d, m = r["direction"], r["materiality"]
        kind = [r["kind"] == k for k in ("pre", "intra", "after")]
        fam = [r["event_type"] in v for v in FAMILY.values()]
        px = r["start_price"]
        size = [px < 5, 5 <= px < 20, px >= 20]
        onehot = np.zeros(len(EVENT_TYPES))
        onehot[tid[r["event_type"]]] = 1
        base = [d, m, d * m, d < 0, r["core"], r["price_recap"], d * r["core"], (d < 0) * r["core"]]
        base += kind + [d * k for k in kind] + fam + [d * f for f in fam] + size + [d * s for s in size]
        X.append(np.concatenate([np.array(base, dtype=float), onehot, onehot * d]))
    X = np.array(X)
    if use_emb:
        X = np.hstack([X, emb])
    return X


def ridge(xtr, ytr, xte, lam):
    m, s = xtr.mean(0), xtr.std(0) + 1e-8
    a, b = (xtr - m) / s, (xte - m) / s
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (ytr - ytr.mean()))
    return b @ w + ytr.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(Spy(a.spy))
    rows.sort(key=lambda r: r["pub_time"])
    Y = {h: np.array([r[f"a_{h}d"] if r[f"a_{h}d"] is not None else np.nan for r in rows]) for h in (1, 2, 3)}
    for h in Y:
        lo, hi = np.nanquantile(Y[h], [0.01, 0.99])
        Y[h] = np.clip(Y[h], lo, hi)
    days = np.array([r["date"] for r in rows])
    core = np.array([r["core"] for r in rows], bool)
    dirn = np.array([r["direction"] for r in rows], float)
    kind = np.array([r["kind"] for r in rows])
    px = np.array([r["start_price"] for r in rows])
    ok = ~np.isnan(Y[1])
    L = ["# EDT 보도자료로 이벤트 신호 재현과 학습", "",
         f"대상: 가격·SPY·LLM 결과가 모두 있는 {ok.sum():,}건 (2020-03 ~ 2021-05). 핵심 이벤트 {(core & ok).sum():,}건.",
         "초과수익률 = 종목 − SPY (같은 시각 구간). 값은 날짜별 평균 후 평균(bp), t 는 날짜 묶음 기준. 거래비용 미반영.", ""]

    # 1. 규칙 재현
    L += ["## 1. FNSPID 에서 찾은 규칙을 그대로 적용 (학습 없음)", "",
          "| 집단 | 규칙 | 1일 bp | t | 2일 bp | t | 3일 bp | t | 이벤트 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for gname, gm in (("전체", np.ones(len(rows), bool)), ("핵심 이벤트", core)):
        for rname, sig in (("LLM 방향 (+1 매수, −1 매도)", dirn), ("나쁜 뉴스만 매도", -(dirn < 0).astype(float))):
            cells = []
            for h in (1, 2, 3):
                m = gm & (sig != 0) & ~np.isnan(Y[h])
                mean, t, _ = tstat_by_day(sig[m] * Y[h][m], days[m])
                cells += [f"{mean * 1e4:+.1f}", f"{t:+.2f}"]
            L.append(f"| {gname} | {rname} | " + " | ".join(cells) + f" | {int((gm & (sig != 0)).sum()):,} |")

    # 2. 이질성 (핵심 이벤트, 1일)
    L += ["", "## 2. 어디서 효과가 나는가 (핵심 이벤트, LLM 방향 규칙, 1일)", "",
          "| 구분 | 방향 반영 초과수익률 bp | t | 나쁜 뉴스 평균 bp | t | 좋은 뉴스 평균 bp | t |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    groups = [("장 시작 전 발행", kind == "pre"), ("장중 발행", kind == "intra"), ("장 마감 후 발행", kind == "after"),
              ("주가 $5 미만", px < 5), ("주가 $5~20", (px >= 5) & (px < 20)), ("주가 $20 이상", px >= 20)]
    for gname, gm in groups:
        m = core & gm & ok
        c = []
        for sel, sgn in ((dirn != 0, dirn), (dirn < 0, np.ones_like(dirn)), (dirn > 0, np.ones_like(dirn))):
            mm = m & sel
            mean, t, _ = tstat_by_day(sgn[mm] * Y[1][mm], days[mm]) if mm.sum() > 5 else (np.nan, np.nan, 0)
            c += [f"{mean * 1e4:+.1f}", f"{t:+.2f}"]
        L.append(f"| {gname} ({m.sum():,}) | " + " | ".join(c) + " |")

    # 3. 학습 모델 (핵심 이벤트, 1일)
    emb = None
    try:
        ids = np.load("data/edt/edt_emb_ids.npy")
        vec = np.vstack([np.load(p).astype(np.float32) for p in sorted(glob.glob("data/edt/edt_emb.part*.npy"))])
        pos = {int(i): k for k, i in enumerate(ids)}
        emb_full = np.array([vec[pos[r["edt_id"]]] for r in rows])
    except FileNotFoundError:
        emb_full = None
    tr = (days <= TRAIN_END) & ok & core
    va = (days > (datetime.fromisoformat(TRAIN_END) + timedelta(days=3)).strftime("%Y-%m-%d")) & (days <= VALID_END) & ok & core
    tr2 = (days <= VALID_END) & ok & core
    te = (days > (datetime.fromisoformat(VALID_END) + timedelta(days=3)).strftime("%Y-%m-%d")) & ok & core
    y = Y[1]
    L += ["", "## 3. 학습 모델 vs 규칙 (핵심 이벤트, 1일 초과수익률)", "",
          f"학습 ~2020-12 ({tr.sum():,}) → 검증 2021-01~02 ({va.sum():,}) 로 설정 선택 → 학습 ~2021-02 ({tr2.sum():,}) → 평가 2021-03~05 ({te.sum():,}).", "",
          "| 모델 | 검증 순위IC | 평가 순위IC | 평가 롱숏 bp | t |", "|---|---:|---:|---:|---:|"]

    def report(name, pv, pt):
        bp, t, _ = ls(pt, y[te], days[te])
        L.append(f"| {name} | {ric(pv, y[va]):+.4f} | {ric(pt, y[te]):+.4f} | {bp:+.1f} | {t:+.2f} |")

    report("규칙: LLM 방향", dirn[va], dirn[te])
    report("규칙: 나쁜 뉴스만 매도", -(dirn[va] < 0).astype(float), -(dirn[te] < 0).astype(float))
    X = features(rows)
    if emb_full is not None:
        # 임베딩 PCA 는 학습 구간으로만 맞춘다
        def pca(fit_mask, k):
            mu = emb_full[fit_mask].mean(0)
            _, _, vt = np.linalg.svd(emb_full[fit_mask] - mu, full_matrices=False)
            return (emb_full - mu) @ vt[:k].T
    variants = [("Ridge: LLM 이벤트 특징", False, 0)]
    if emb_full is not None:
        variants += [("Ridge: LLM 이벤트 + 임베딩 PCA32", True, 32), ("Ridge: LLM 이벤트 + 임베딩 PCA128", True, 128)]
    for name, use_emb, k in variants:
        best, bestv = None, -9
        for lam in (1, 10, 100, 1e3, 1e4, 1e5):
            Xv = np.hstack([X, pca(tr, k)]) if use_emb else X
            v = ric(ridge(Xv[tr], y[tr], Xv[va], lam), y[va])
            if v > bestv:
                best, bestv = lam, v
        Xv = np.hstack([X, pca(tr, k)]) if use_emb else X
        pv = ridge(Xv[tr], y[tr], Xv[va], best)
        Xt = np.hstack([X, pca(tr2, k)]) if use_emb else X
        report(f"{name} (λ={best:g})", pv, ridge(Xt[tr2], y[tr2], Xt[te], best))
    import lightgbm as lgb
    for name, use_emb, k in [("LightGBM: LLM 이벤트 특징", False, 0)] + (
            [("LightGBM: LLM 이벤트 + 임베딩 PCA32", True, 32)] if emb_full is not None else []):
        params = {"objective": "huber", "alpha": 0.02, "learning_rate": 0.03, "num_leaves": 15,
                  "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "lambda_l2": 10.0, "verbose": -1, "seed": 0, "num_threads": 4}
        Xv = np.hstack([X, pca(tr, k)]) if use_emb else X
        bst = lgb.train(params, lgb.Dataset(Xv[tr], y[tr]), num_boost_round=600)
        rounds = list(range(50, 601, 50))
        best = max(rounds, key=lambda r: ric(bst.predict(Xv[va], num_iteration=r), y[va]))
        pv = bst.predict(Xv[va], num_iteration=best)
        Xt = np.hstack([X, pca(tr2, k)]) if use_emb else X
        bst2 = lgb.train(params, lgb.Dataset(Xt[tr2], y[tr2]), num_boost_round=best)
        report(f"{name} (반복 {best})", pv, bst2.predict(Xt[te]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
