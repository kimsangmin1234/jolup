"""핵심 이벤트 전용 모델 (Trade the Event 방식의 조건부 예측).

모든 기사를 점수화하는 대신, LLM 이 '대상 종목이 주인공 + 새 정보'로 판정한 핵심 이벤트만으로
학습·평가한다. 이벤트 연구에서 확인한 '나쁜 뉴스 후 지속 하락'과 Chan(2003)의
'뉴스와 같은 방향의 기존 움직임은 이어진다'를 특징으로 넣는다.

모델
    R0 규칙: 예측 = LLM 방향 (−1/0/+1)
    R1 규칙: 예측 = 나쁜 뉴스면 −1, 아니면 0 (비대칭)
    C1 Ridge: 방향·중요도·발행 시점·유형군 + 상호작용
    C2 C1 + 진입 전 반응(갭·장중·전일·5일·거래량) + 방향×반응
    C3 C2 + 최근 지표(창 마지막 날 9종)
    CG LightGBM (C3 특징)
평가: 같은 walk-forward. 핵심 이벤트 안에서의 순위 IC, 그리고 예측 상위 1/3 매수 − 하위 1/3 매도의
      이벤트 포트폴리오 초과수익률(같은 종목·날짜는 묶은 뒤 평균, t 는 날짜 묶음 기준).

    python analysis/core_event_model.py --out experiments/eventdriven_suite/CORE_MODEL.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_event_suite as EV
import run_eventdriven_suite as ED
import run_suite as B

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


def features(d, tr, level):
    from extract_events import EVENT_TYPES
    n = len(d.Y)
    dirn, mat = d.A["direction"], d.A["materiality"]
    kinds = [(d.kind == k).astype(float) for k in ("pre", "intra", "after")]
    names = np.array(EVENT_TYPES)[d.etype]
    fam = [np.isin(names, v).astype(float) for v in FAMILY.values()]
    cols = [dirn, mat, dirn * mat, (dirn < 0).astype(float), d.A["price_recap"]]
    cols += kinds + [dirn * k for k in kinds] + [(dirn < 0) * k for k in kinds]
    cols += fam + [dirn * f for f in fam]
    if level >= 2:
        R = [d.F[k] for k in ("gap_abn_z", "pre_abn_z", "prev1_z", "mom5_z", "volr")]
        cols += R + [dirn * r for r in R]
    X = np.stack(cols, 1)
    if level >= 3:
        X = np.hstack([X, d.windows(tr, "window")[:, -1, :]])
    return X


def portfolio(pred, y_abn, key, day):
    """예측 상위 1/3 매수 − 하위 1/3 매도 이벤트 포트폴리오(같은 종목·날짜 묶음, 날짜별 평균)."""
    lo, hi = np.quantile(pred, [1 / 3, 2 / 3])
    sign = np.where(pred >= hi, 1, np.where(pred <= lo, -1, 0))
    if hi == lo:                                   # 동률이 많을 때(규칙 모델): 부호로 판정
        sign = np.sign(pred).astype(int)
    by_day = {}
    seen = set()
    for s, r, k, dd in zip(sign, y_abn, key, day):
        if s == 0 or k in seen:
            continue
        seen.add(k)
        by_day.setdefault(dd, []).append(s * r)
    v = np.array([np.mean(x) for x in by_day.values()])
    return v.mean() * 1e4, v.mean() / v.std(ddof=1) * np.sqrt(len(v)) if len(v) > 2 else np.nan, len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = ED.EDData("experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz", "data/fnspid/indicators.npz",
                  "data/fnspid/event_features.jsonl.gz", "data/fnspid/events_llm.jsonl.gz")
    core = d.core
    key = np.array([f"{t}|{x}|{k}" for t, x, k in zip(d.T, d.D, d.kind)])
    splits = d.splits()
    lams = [1, 10, 100, 1e3, 1e4, 1e5]
    results = {}

    def run_rule(fn):
        out = []
        for name, tr, ev_ in splits:
            m = ev_ & core
            out.append((name, fn()[m], m))
        return out

    specs = [("R0 규칙: LLM 방향", "rule0"), ("R1 규칙: 나쁜 뉴스만 −1", "rule1"),
             ("C1 Ridge: 방향·중요도·시점·유형군", 1), ("C2 C1 + 진입 전 반응·방향×반응", 2),
             ("C3 C2 + 최근 지표", 3), ("CG LightGBM (C3 특징)", "gbm")]
    for title, kind in specs:
        rows = []
        if kind == "rule0":
            rows = run_rule(lambda: d.A["direction"].astype(float))
        elif kind == "rule1":
            rows = run_rule(lambda: -(d.A["direction"] < 0).astype(float))
        elif kind == "gbm":
            import lightgbm as lgb
            params = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.02, "num_leaves": 7,
                      "min_data_in_leaf": 100, "feature_fraction": 0.7, "bagging_fraction": 0.7,
                      "bagging_freq": 1, "lambda_l2": 50.0, "verbose": -1, "seed": 0, "num_threads": 4}
            rounds = list(range(25, 401, 25))
            fold = {r: [] for r in rounds}
            for name, tr, ev_ in splits[:-1]:
                X, y = features(d, tr, 3), EV.winsor(d.Y, tr)
                t, v = tr & core, ev_ & core
                bst = lgb.train(params, lgb.Dataset(X[t], y[t]), num_boost_round=rounds[-1])
                for r in rounds:
                    fold[r].append(EV._rank_ic(bst.predict(X[v], num_iteration=r), d.Y[v]))
            best = max(rounds, key=lambda r: np.mean(fold[r]))
            for name, tr, ev_ in splits:
                X, y = features(d, tr, 3), EV.winsor(d.Y, tr)
                t, v = tr & core, ev_ & core
                bst = lgb.train(params, lgb.Dataset(X[t], y[t]), num_boost_round=best)
                rows.append((name, bst.predict(X[v]), v))
        else:
            fold = {l: [] for l in lams}
            for name, tr, ev_ in splits[:-1]:
                X, y = features(d, tr, kind), EV.winsor(d.Y, tr)
                t, v = tr & core, ev_ & core
                for l in lams:
                    fold[l].append(EV._rank_ic(B.ridge_fit_predict(X[t], y[t], X[v], l), d.Y[v]))
            best = max(lams, key=lambda l: np.mean(fold[l]))
            for name, tr, ev_ in splits:
                X, y = features(d, tr, kind), EV.winsor(d.Y, tr)
                t, v = tr & core, ev_ & core
                rows.append((name, B.ridge_fit_predict(X[t], y[t], X[v], best), v))
        res = []
        for name, p, m in rows:
            ric = EV._rank_ic(p, d.Y[m]) if np.std(p) > 0 else 0.0
            bp, t, nd = portfolio(p, d.y_abn[m], key[m], d.D[m])
            res.append((name, ric, bp, t, int(m.sum())))
        results[title] = res
        print(title, " | ".join(f"{n} {r:+.3f}" for n, r, *_ in res), f"| test LS {res[-1][2]:+.1f}bp t={res[-1][3]:+.2f}")

    L = ["# 핵심 이벤트 전용 모델", "",
         "대상: LLM 이 '대상 종목이 주인공 + 새 정보'로 판정한 핵심 이벤트만. 학습도 평가도 이 집단 안에서 한다.",
         "롱숏: 예측 상위 1/3 매수 − 하위 1/3 매도 (규칙 모델은 부호), 같은 종목·날짜 묶음 후 날짜별 평균, 시장 조정 초과수익률, 거래비용 미반영.", "",
         "| 모델 | 검증 평균 순위IC | 분기별 순위IC | 평가 순위IC | 평가 롱숏 bp | t | 평가 이벤트 |",
         "|---|---:|---|---:|---:|---:|---:|"]
    for title, res in results.items():
        val = [r for _, r, *_ in res[:-1]]
        n, r, bp, t, cnt = res[-1]
        L.append(f"| {title} | {np.mean(val):+.4f} | {' '.join(f'{x:+.3f}' for x in val)} | {r:+.4f} | {bp:+.1f} | {t:+.2f} | {cnt:,} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
