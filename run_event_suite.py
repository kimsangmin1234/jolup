"""선행 연구를 반영한 이벤트 기반 구조들을 같은 walk-forward 기준으로 비교한다.

설계 근거
    - 목표값: 시장(SPY) 조정 초과수익률을 20일 변동성으로 나눈 값(y_abn_z).
      이벤트 연구(MacKinlay 1997). 시장 전체 움직임과 종목 간 크기 차이를 제거한다.
    - 진입 전 반응(갭·장중 움직임·전일/5일 수익률·거래량): 뉴스가 있는 움직임은
      지속되고 뉴스 없는 움직임은 반전된다(Chan 2003). 단기 반전(Jegadeesh 1990).
    - 뉴스 새로움·집중도: 반복 보도는 과잉 반응 후 반전(Tetlock 2011).
    - LLM 감성·임베딩(Lopez-Lira & Tang 2023; Chen·Kelly·Xiu 2023).
    - 같은 날 이벤트끼리의 순위 학습(Feng et al. 2019, Relational Stock Ranking).
    - 신경망은 논문 모듈(주 신호 FC → TCN → 비대칭 교차 어텐션 → 게이트 결합 → MLP)을
      그대로 쓰되, 주 신호 입력을 [임베딩 PCA, 뉴스 특징, 진입 전 반응]으로 확장한다.

평가 (run_suite.py 와 같은 분할)
    검증 6개 분기 평균 순위 IC 로 설정 선택 → 2023 하반기 1회 평가.
    지표: 순위 IC·피어슨 IC(y_abn_z), 유형별 순위 IC, 방향 적중,
          날짜별 롱숏(예측 상위 1/3 매수 − 하위 1/3 매도, 초과수익률 평균)의 일평균·t값·연환산 샤프.

    python run_event_suite.py --out experiments/event_suite --commit
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import run_suite as base
from config import ModelConfig
from model import NewsDrivenStockPredictor

logger = logging.getLogger("event_suite")

REACT = ["gap_abn_z", "pre_abn_z", "prev1_z", "mom5_z", "volr", "tod", "is_pre", "is_intra", "is_after"]
NEWS = ["sentiment", "abs_sent", "novelty", "n_prior24", "sent_prior24",
        "sent_x_novelty", "sent_x_pre", "sent_x_gap"]


# --------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------

class EventData(base.Data):
    def __init__(self, cache: str, npz: str, events: str) -> None:
        super().__init__(cache, npz, events)
        with gzip.open(events, "rt", encoding="utf-8") as f:
            feat = {r["news_id"]: r for r in map(json.loads, f)}
        rows = [feat[r["news_id"]] for r in self.meta]
        for r in rows:
            r["is_pre"], r["is_intra"], r["is_after"] = (float(r["kind"] == k) for k in ("pre", "intra", "after"))
            r["abs_sent"] = abs(r["sentiment"])
            r["sent_x_novelty"] = r["sentiment"] * r["novelty"]
            r["sent_x_pre"] = r["sentiment"] * r["pre_abn_z"]
            r["sent_x_gap"] = r["sentiment"] * r["gap_abn_z"]
        self.F = {k: np.clip(np.array([r[k] for r in rows], dtype=np.float64), -10, 10)
                  for k in REACT + NEWS}
        self.Y = np.array([r["y_abn_z"] for r in rows])
        self.y_abn = np.array([r["y_abn"] for r in rows])
        self.y_raw = np.array([r["y_raw"] for r in rows])
        self.kind = np.array([r["kind"] for r in rows])
        logger.info("이벤트 %d건 (장 시작 전 %d / 장중 %d / 장 마감 후 %d)", len(rows),
                    (self.kind == "pre").sum(), (self.kind == "intra").sum(), (self.kind == "after").sum())

    def features(self, tr: np.ndarray, spec: dict) -> np.ndarray:
        cols = []
        if spec.get("sent_only"):
            cols.append(self.F["sentiment"][:, None])
        if spec.get("react"):
            cols += [self.F[k][:, None] for k in REACT]
        if spec.get("news"):
            cols += [self.F[k][:, None] for k in NEWS]
        if spec.get("k"):
            cols.append(self.embedding(tr, spec["k"]).astype(np.float64))
        if spec.get("ind") == "flat":
            w = self.windows(tr, "window")
            cols.append(w.reshape(len(w), -1))
        elif spec.get("ind") == "last":
            cols.append(self.windows(tr, "window")[:, -1, :])
        return np.hstack(cols)


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------

def _rank_ic(a, b):
    return base._corr(base._rank(a), base._rank(b))


def evaluate(data: EventData, idx: np.ndarray, pred: np.ndarray) -> dict:
    y, ya, yr, kind, day = data.Y[idx], data.y_abn[idx], data.y_raw[idx], data.kind[idx], data.D[idx]
    out = {"n": int(len(idx)), "rank_ic": _rank_ic(pred, y), "ic": base._corr(pred, y),
           "rank_ic_raw": _rank_ic(pred, yr),
           "dir": float(((pred >= np.median(pred)) == (ya > 0)).mean())}
    for k in ("pre", "intra", "after"):
        m = kind == k
        out[f"rank_ic_{k}"] = _rank_ic(pred[m], y[m]) if m.sum() > 20 else float("nan")
        out[f"n_{k}"] = int(m.sum())
    # 날짜별 롱숏: 예측 상위 1/3 − 하위 1/3 의 초과수익률 평균
    ls = []
    for d in np.unique(day):
        m = day == d
        if m.sum() < 6:
            continue
        p, r = pred[m], ya[m]
        q1, q2 = np.quantile(p, [1 / 3, 2 / 3])
        if (p >= q2).sum() and (p <= q1).sum():
            ls.append(r[p >= q2].mean() - r[p <= q1].mean())
    ls = np.array(ls)
    out.update({"ls_days": int(len(ls)), "ls_mean_bp": float(ls.mean() * 1e4) if len(ls) else float("nan"),
                "ls_t": float(ls.mean() / ls.std() * np.sqrt(len(ls))) if len(ls) > 2 else float("nan"),
                "ls_sharpe": float(ls.mean() / ls.std() * np.sqrt(252)) if len(ls) > 2 else float("nan")})
    return out


def winsor(y: np.ndarray, tr: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(y[tr], [0.01, 0.99])
    return np.clip(y, lo, hi)


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------

def run_linear(data: EventData, spec: dict) -> dict:
    lams = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
    splits = data.splits()
    fold = {l: [] for l in lams}
    for _, tr, va in splits[:-1]:
        x, y = data.features(tr, spec), winsor(data.Y, tr)
        for l in lams:
            fold[l].append(_rank_ic(base.ridge_fit_predict(x[tr], y[tr], x[va], l), data.Y[va]))
    best = max(lams, key=lambda l: np.mean(fold[l]))
    _, tr, te = splits[-1]
    x, y = data.features(tr, spec), winsor(data.Y, tr)
    p = base.ridge_fit_predict(x[tr], y[tr], x[te], best)
    return {"choice": {"lambda": best}, "fold": fold[best],
            "test": evaluate(data, np.where(te)[0], p)}


def run_gbm(data: EventData, spec: dict) -> dict:
    import lightgbm as lgb
    params = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.02, "num_leaves": 7,
              "min_data_in_leaf": 400, "feature_fraction": 0.7, "bagging_fraction": 0.7,
              "bagging_freq": 1, "lambda_l2": 50.0, "verbose": -1, "seed": 0,
              "num_threads": torch.get_num_threads()}
    rounds = list(range(25, 601, 25))
    splits = data.splits()
    fold = {r: [] for r in rounds}
    for _, tr, va in splits[:-1]:
        x, y = data.features(tr, spec), winsor(data.Y, tr)
        bst = lgb.train(params, lgb.Dataset(x[tr], y[tr]), num_boost_round=rounds[-1])
        for r in rounds:
            fold[r].append(_rank_ic(bst.predict(x[va], num_iteration=r), data.Y[va]))
    best = max(rounds, key=lambda r: np.mean(fold[r]))
    _, tr, te = splits[-1]
    x, y = data.features(tr, spec), winsor(data.Y, tr)
    bst = lgb.train(params, lgb.Dataset(x[tr], y[tr]), num_boost_round=best)
    return {"choice": {"rounds": best}, "fold": fold[best],
            "test": evaluate(data, np.where(te)[0], bst.predict(x[te]))}


class EventNet(nn.Module):
    """논문 모듈 그대로, 주 신호 입력만 [임베딩 PCA ‖ 뉴스 특징 ‖ 진입 전 반응]으로 확장."""

    def __init__(self, in_dim: int, d: int, use_tcn: bool) -> None:
        super().__init__()
        config = ModelConfig(d_model=d)
        config.news.embedding_dim = in_dim
        config.tcn.hidden_channels = min(128, d)
        config.predictor.hidden_dims = (d, d // 2)
        self.use_tcn = use_tcn
        self.model = NewsDrivenStockPredictor(config)
        if not use_tcn:
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x, s, w):
        if self.use_tcn:
            return self.model(x, s, w).prediction
        return self.head(self.model.news_encoder(x, s)).squeeze(-1)


def pairwise_rank_loss(pred, y, day):
    """같은 날짜 이벤트 쌍에 대한 로지스틱 순위 손실 (Feng et al. 2019 의 순위 학습 아이디어)."""
    same = day[:, None] == day[None, :]
    better = (y[:, None] > y[None, :]) & same
    if better.sum() == 0:
        return pred.sum() * 0
    diff = pred[:, None] - pred[None, :]
    return nn.functional.softplus(-diff[better]).mean()


def date_batches(day: np.ndarray, idx: np.ndarray, bs: int, rng) -> list[np.ndarray]:
    """같은 날짜 이벤트가 한 배치에 모이도록 날짜 단위로 묶는다."""
    groups = {}
    for i in idx:
        groups.setdefault(day[i], []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    batches, cur = [], []
    for k in keys:
        cur += groups[k]
        if len(cur) >= bs:
            batches.append(np.array(cur))
            cur = []
    if cur:
        batches.append(np.array(cur))
    return batches


def train_eventnet(spec, x, s, w, y, day_codes, tr_idx, ev_idx, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EventNet(x.shape[1], spec["d"], spec.get("tcn", True))
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec.get("wd", 1e-4))
    xt, st, wt, yt = (torch.from_numpy(a.astype(np.float32)) for a in (x, s, w, y))
    dt = torch.from_numpy(day_codes)
    preds = []
    for _ in range(epochs):
        model.train()
        for b in date_batches(day_codes, tr_idx, spec.get("bs", 256), rng):
            bi = torch.from_numpy(b)
            p = model(xt[bi], st[bi], wt[bi])
            loss = nn.functional.mse_loss(p, yt[bi])
            if spec.get("rank", 0):
                loss = loss + spec["rank"] * pairwise_rank_loss(p, yt[bi], dt[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ei = torch.from_numpy(ev_idx)
            preds.append(torch.cat([model(xt[ei[i:i + 4096]], st[ei[i:i + 4096]], wt[ei[i:i + 4096]])
                                    for i in range(0, len(ei), 4096)]).numpy().astype(np.float64))
    return preds


def nn_inputs(data: EventData, tr, spec):
    x = data.features(tr, spec)
    m, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    x = (x - m) / sd
    w = data.windows(tr, "window")
    y = winsor(data.Y, tr)
    return x, w, y / y[tr].std()


def run_nn(data: EventData, spec: dict) -> dict:
    epochs = spec.get("epochs", 15)
    splits = data.splits()
    _, day_codes = np.unique(data.D, return_inverse=True)
    curves = []
    for name, tr, va in splits[:-1]:
        x, w, y = nn_inputs(data, tr, spec)
        per_seed = []
        for seed in spec.get("fold_seeds", [0, 1]):
            t0 = time.time()
            preds = train_eventnet(spec, x, data.S, w, y, day_codes, np.where(tr)[0], np.where(va)[0], seed, epochs)
            per_seed.append([_rank_ic(p, data.Y[va]) for p in preds])
            logger.info("    %s seed %d  최고 순위IC %+.4f (에폭 %d) %.0fs", name, seed,
                        max(per_seed[-1]), int(np.argmax(per_seed[-1])) + 1, time.time() - t0)
        curves.append(np.mean(per_seed, 0))
    curves = np.array(curves)
    best = int(np.argmax(curves.mean(0)))
    _, tr, te = splits[-1]
    x, w, y = nn_inputs(data, tr, spec)
    test_preds = [train_eventnet(spec, x, data.S, w, y, day_codes, np.where(tr)[0], np.where(te)[0],
                                 seed, best + 1)[best] for seed in spec.get("test_seeds", [0, 1, 2])]
    ens = np.mean(test_preds, 0)
    return {"choice": {"epochs": best + 1}, "fold": curves[:, best].tolist(),
            "valid_curve": curves.mean(0).tolist(), "test": evaluate(data, np.where(te)[0], ens),
            "test_seeds": [evaluate(data, np.where(te)[0], p) for p in test_preds]}


NN = {"d": 64, "lr": 1e-3, "wd": 1e-4, "epochs": 15, "bs": 256, "fold_seeds": [0, 1], "test_seeds": [0, 1, 2]}

EXPERIMENTS = {
    "linear": [
        ("E0 감성 점수만", {"sent_only": True}),
        ("E1 진입 전 반응만 (갭·장중·단기 수익률·거래량)", {"react": True}),
        ("E2 뉴스 특징만 (감성·새로움·집중도·상호작용)", {"news": True}),
        ("E3 반응 + 뉴스", {"react": True, "news": True}),
        ("E4 반응 + 뉴스 + 임베딩 PCA8 + 지표 30일", {"react": True, "news": True, "k": 8, "ind": "flat"}),
        ("E5 지표 30일만", {"ind": "flat"}),
        ("E6 반응 + 지표 30일", {"react": True, "ind": "flat"}),
        ("E7 뉴스 + 임베딩 PCA8 + 지표 30일", {"news": True, "k": 8, "ind": "flat"}),
    ],
    "gbm": [
        ("G3 LightGBM 반응 + 뉴스", {"react": True, "news": True}),
        ("G4 LightGBM 반응 + 뉴스 + PCA8 + 최근 지표", {"react": True, "news": True, "k": 8, "ind": "last"}),
    ],
    "nn": [
        ("M1 제안: 반응 인식 논문 구조 + 순위 손실", {**NN, "react": True, "news": True, "k": 8, "rank": 1.0}),
        ("M2 └ 순위 손실 제거", {**NN, "react": True, "news": True, "k": 8}),
        ("M3 └ 진입 전 반응 제거 (논문 구조 + 새 목표값)", {**NN, "news": True, "k": 8, "rank": 1.0}),
        ("M4 └ 뉴스·임베딩 제거 (반응 + 지표만)", {**NN, "react": True, "rank": 1.0}),
        ("M5 └ TCN·어텐션 제거 (주 신호 MLP 만)", {**NN, "react": True, "news": True, "k": 8, "rank": 1.0, "tcn": False}),
    ],
}
RUNNERS = {"linear": run_linear, "gbm": run_gbm, "nn": run_nn}


def write_report(out: Path) -> None:
    results = {}
    for p in sorted(out.glob("results_*.json")):
        results.update(json.loads(p.read_text()))
    rows = sorted(results.items(), key=lambda kv: -np.mean(kv[1]["fold"]))
    L = ["# 이벤트 기반 구조 비교 (walk-forward)", "",
         "- 목표값: 시장(SPY) 조정 초과수익률 / 20일 변동성 (`y_abn_z`). 진입은 발행 이후, 청산은 당일 또는 다음날 종가.",
         "- 선택: 검증 6개 분기(2022 1분기 ~ 2023 2분기) 평균 순위 IC. 평가: 2023 하반기 1회.",
         "- 롱숏: 날짜별로 예측 상위 1/3 매수 − 하위 1/3 매도(초과수익률, 동일가중). 일평균은 bp(0.01%), 샤프는 연환산. 거래비용 미반영.",
         "- 신경망은 시드 3개 예측 평균(앙상블)으로 평가했다.", "",
         "| 모델 | 검증 순위IC | 분기별 | 선택 | 평가 순위IC | 평가 IC | 장 시작 전 | 장중 | 장 마감 후 | 방향 | 롱숏 bp/일 | t | 샤프 |",
         "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows:
        t = r["test"]
        L.append(f"| {name} | {np.mean(r['fold']):+.4f} | {' '.join(f'{x:+.3f}' for x in r['fold'])} | "
                 f"{', '.join(f'{k}={v:g}' for k, v in r['choice'].items())} | {t['rank_ic']:+.4f} | {t['ic']:+.4f} | "
                 f"{t['rank_ic_pre']:+.3f} | {t['rank_ic_intra']:+.3f} | {t['rank_ic_after']:+.3f} | {t['dir']:.3f} | "
                 f"{t['ls_mean_bp']:+.1f} | {t['ls_t']:+.2f} | {t['ls_sharpe']:+.2f} |")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz")
    ap.add_argument("--indicators", default="data/fnspid/indicators.npz")
    ap.add_argument("--events", default="data/fnspid/event_features.jsonl.gz")
    ap.add_argument("--groups", default="linear,gbm,nn")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="experiments/event_suite")
    ap.add_argument("--worker", default="main")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / f"suite_{args.worker}.log", encoding="utf-8")])
    torch.set_num_threads(args.threads)
    data = EventData(args.cache, args.indicators, args.events)
    path = out / f"results_{args.worker}.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    for group in args.groups.split(","):
        for name, spec in EXPERIMENTS[group]:
            if (only and not any(name.startswith(o + " ") for o in only)) or name in results:
                continue
            logger.info("▶ %s  %s", name, json.dumps(spec, ensure_ascii=False))
            t0 = time.time()
            r = RUNNERS[group](data, spec)
            r.update(spec=spec, group=group, sec=round(time.time() - t0))
            results[name] = r
            t = r["test"]
            logger.info("  검증 순위IC %+.4f | 평가 순위IC %+.4f IC %+.4f | 장전 %+.3f 장중 %+.3f 마감후 %+.3f | 롱숏 %+.1fbp t=%+.2f (%.0fs)",
                        np.mean(r["fold"]), t["rank_ic"], t["ic"], t["rank_ic_pre"], t["rank_ic_intra"],
                        t["rank_ic_after"], t["ls_mean_bp"], t["ls_t"], time.time() - t0)
            path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(out)
            if args.commit:
                base.commit(out, f"[event_suite] {name}")


if __name__ == "__main__":
    main()
