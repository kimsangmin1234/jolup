"""여러 모델·설정을 같은 walk-forward 기준으로 비교한다.

모든 모델을 같은 데이터(발행 시각 확인분), 같은 검증·평가 구간으로 비교해
어떤 구성이 예측을 잘하는지 가린다.

* 검증: 2022 1분기 ~ 2023 2분기, 6개 분기를 차례로 검증 구간으로 쓴다.
  각 폴드는 검증 시작 전 데이터로만 학습하고, 직전 3일은 학습에서 뺀다.
* 선택: 6개 분기의 평균 IC(예측과 실제 등락률의 피어슨 상관)로 하이퍼파라미터
  (Ridge λ, 부스팅 반복 수, 신경망 에폭 수)를 고른다.
* 평가: 2023 하반기. 2023-06-27 까지의 데이터로 학습하고 고른 설정으로 1회 평가한다.

모델 묶음
    linear  : Ridge 회귀 (감성 / 임베딩 PCA / 지표 / 결합)
    gbm     : LightGBM
    nn      : 논문 구조 신경망과 그 변형·제거 실험

    python run_suite.py --groups linear,gbm,nn --out experiments/suite
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import ModelConfig
from data.dataset import apply_label_file, load_indicators
from data.technical_indicators import MinMaxScaler
from model import NewsDrivenStockPredictor
from modules import (AsymmetricCrossAttention, GatedResidualFusion,
                     NewsSignalEncoder, ReturnPredictor, TCNEncoder)

logger = logging.getLogger("suite")

FOLDS = [("2022-01-01", "2022-03-31"), ("2022-04-01", "2022-06-30"),
         ("2022-07-01", "2022-09-30"), ("2022-10-01", "2022-12-31"),
         ("2023-01-01", "2023-03-31"), ("2023-04-01", "2023-06-30")]
TEST = ("2023-07-01", "2023-12-31")
EMBARGO_DAYS = 3
LOOKBACK = 30
PRICE_COLS = (0, 2, 3, 6, 7)   # close, ma5, ma20, bb_upper, bb_lower


def cutoff(start: str) -> str:
    return (date.fromisoformat(start) - timedelta(days=EMBARGO_DAYS)).isoformat()


# --------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------

def window_normalize(w: np.ndarray) -> np.ndarray:
    """(N, 30, 9) 지표 창을 종목·가격 수준과 무관한 값으로 바꾼다.

    가격형 지표(종가·이동평균·볼린저)는 창 마지막 종가 대비 비율, 거래량은
    창 평균 대비 로그 비율, MACD·ATR 은 종가로 나누고 RSI 는 0 중심으로 옮긴다.
    """
    c = w[:, -1, 0][:, None]                       # (N, 1) 창 마지막 종가
    out = np.empty_like(w)
    for j in PRICE_COLS:
        out[:, :, j] = w[:, :, j] / c - 1.0
    vol = w[:, :, 1]
    out[:, :, 1] = np.log((vol + 1.0) / (vol.mean(1, keepdims=True) + 1.0))
    out[:, :, 4] = w[:, :, 4] / 100.0 - 0.5
    out[:, :, 5] = w[:, :, 5] / c
    out[:, :, 8] = w[:, :, 8] / c
    return out


class Data:
    def __init__(self, cache: str, npz: str, labels: str = "") -> None:
        meta, emb = [], []
        for path in sorted(glob.glob(cache)):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    emb.append(np.asarray(r.pop("embedding"), dtype=np.float32))
                    meta.append(r)
        if labels:
            # 라벨 정의만 바꾼다. 라벨 파일에 없는 레코드는 뺀다.
            for i, r in enumerate(meta):
                r["_i"] = i
            meta = apply_label_file(meta, labels)
            emb = [emb[r.pop("_i")] for r in meta]
        self.indicators, self.date_index = load_indicators(npz)

        keep, wins, rows = [], [], []
        for i, r in enumerate(meta):
            t = r["ticker"]
            row = self.date_index.get(t, {}).get(r.get("anchor") or r["date"])
            if row is None or row + 1 < LOOKBACK:
                continue
            keep.append(i)
            rows.append(row)
            wins.append(self.indicators[t][row + 1 - LOOKBACK: row + 1])
        self.meta = [meta[i] for i in keep]
        self.E = np.vstack([emb[i] for i in keep])
        del emb
        self.W_raw = np.stack(wins)                       # (N, 30, 9)
        self.W_win = window_normalize(self.W_raw)
        self.row = np.array(rows)
        self.S = np.array([float(r["sentiment"]) for r in self.meta], dtype=np.float32)
        self.Y = np.array([float(r["label"]) for r in self.meta], dtype=np.float64)
        self.D = np.array([r["date"] for r in self.meta])
        self.T = np.array([r["ticker"] for r in self.meta])
        self.intraday = np.array([r.get("horizon") == "same_day" for r in self.meta])
        # 장 시작 전(09:30 이전) 뉴스: 시가→종가 라벨이 발행 이후 구간만 포함하는 유일한 집단
        self.premarket = np.array([r.get("published_et", "")[11:16] < "09:30" for r in self.meta])
        self._pca: dict = {}
        logger.info("레코드 %d건 (장중 %d / 마감 후 %d), 종목 %d개",
                    len(self.Y), self.intraday.sum(), (~self.intraday).sum(),
                    len(set(self.T)))

    # -- 분할 ---------------------------------------------------------------
    def splits(self):
        """(이름, 학습 마스크, 평가 마스크) 목록. 마지막이 최종 평가다."""
        out = []
        for vs, ve in FOLDS:
            out.append((f"{vs[:7]}", self.D < cutoff(vs), (self.D >= vs) & (self.D <= ve)))
        out.append(("test", self.D < cutoff(TEST[0]),
                    (self.D >= TEST[0]) & (self.D <= TEST[1])))
        return out

    # -- 특징 ---------------------------------------------------------------
    def embedding(self, tr: np.ndarray, k: int) -> np.ndarray:
        """학습 구간으로 맞춘 PCA k차원(표준화). k=1536 이면 원본 그대로."""
        if k >= self.E.shape[1]:
            return self.E
        key = (tr.tobytes().__hash__(), k)
        if key not in self._pca:
            etr = self.E[tr].astype(np.float64)
            mu = etr.mean(0)
            w, v = np.linalg.eigh(np.cov((etr - mu).T))
            basis = v[:, np.argsort(w)[::-1][:k]]
            z = (self.E.astype(np.float64) - mu) @ basis
            z /= z[tr].std(0) + 1e-8
            self._pca[key] = z.astype(np.float32)
        return self._pca[key]

    def windows(self, tr: np.ndarray, norm: str) -> np.ndarray:
        """지표 창. global 은 기존(논문 구현) 방식, window 는 종목 무관 정규화."""
        if norm == "global":
            # 학습 구간에 등장한 시점까지 모든 종목의 지표를 모아 MinMax (기존 train.py 와 동일)
            last: dict[str, int] = {}
            for t, d in zip(self.T[tr], self.D[tr]):
                r = self.date_index[t].get(d)
                if r is not None:
                    last[t] = max(last.get(t, 0), r)
            scaler = MinMaxScaler().fit(
                np.concatenate([self.indicators[t][: r + 1] for t, r in last.items()]))
            n = self.W_raw.shape[0]
            return scaler.transform(self.W_raw.reshape(-1, 9)).reshape(n, LOOKBACK, 9).astype(np.float32)
        w = self.W_win
        m, s = w[tr].mean((0, 1)), w[tr].std((0, 1)) + 1e-8
        return ((w - m) / s).astype(np.float32)


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(v)).astype(np.float64)


def metrics(pred: np.ndarray, y: np.ndarray, intraday: np.ndarray,
            premarket: np.ndarray | None = None) -> dict:
    # 방향: 예측 중앙값 이상이면 상승으로 본다(예측 편향 제거). 동률은 상승 쪽에 넣는다.
    up = pred >= np.median(pred)
    hit = up == (y > 0)
    m = intraday
    out = {"n": int(len(y)), "ic": _corr(pred, y), "rank_ic": _corr(_rank(pred), _rank(y)),
           "dir": float(hit.mean()), "dir_raw": float((np.sign(pred) == np.sign(y)).mean()),
           "n_intraday": int(m.sum()), "ic_intraday": _corr(pred[m], y[m]),
           "dir_intraday": float(hit[m].mean())}
    if premarket is not None:
        out.update({"n_premarket": int(premarket.sum()),
                    "ic_premarket": _corr(pred[premarket], y[premarket]),
                    "dir_premarket": float(hit[premarket].mean())})
    return out


# --------------------------------------------------------------------------
# 선형 / 부스팅
# --------------------------------------------------------------------------

def ridge_fit_predict(xtr, ytr, xte, lam):
    m, s = xtr.mean(0), xtr.std(0) + 1e-8
    xtr, xte = (xtr - m) / s, (xte - m) / s
    ym = ytr.mean()
    w = np.linalg.solve(xtr.T @ xtr + lam * np.eye(xtr.shape[1]), xtr.T @ (ytr - ym))
    return xte @ w + ym


def tabular_features(data: Data, tr, spec) -> np.ndarray:
    cols = []
    if spec.get("sent", True):
        cols.append(data.S[:, None].astype(np.float64))
    if spec.get("k", 0):
        cols.append(data.embedding(tr, spec["k"]).astype(np.float64))
    if spec.get("ind"):
        w = data.windows(tr, "window")
        cols.append(w.reshape(len(w), -1) if spec["ind"] == "flat" else w[:, -1, :])
        if spec["ind"] == "last":
            close = data.W_raw[:, :, 0]
            cols.append(np.stack([close[:, -1] / close[:, -2] - 1, close[:, -1] / close[:, -6] - 1,
                                  close[:, -1] / close[:, -21] - 1], 1))
    if spec.get("horizon"):
        cols.append(data.intraday[:, None].astype(np.float64))
    return np.hstack(cols)


def run_linear(data: Data, spec: dict) -> dict:
    lams = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
    splits = data.splits()
    feats = {name: tabular_features(data, tr, spec) for name, tr, _ in splits}
    fold_ic = {lam: [] for lam in lams}
    for name, tr, va in splits[:-1]:
        x = feats[name]
        for lam in lams:
            p = ridge_fit_predict(x[tr], data.Y[tr], x[va], lam)
            fold_ic[lam].append(_corr(p, data.Y[va]))
    best = max(lams, key=lambda l: np.mean(fold_ic[l]))
    name, tr, te = splits[-1]
    p = ridge_fit_predict(feats[name][tr], data.Y[tr], feats[name][te], best)
    return {"choice": {"lambda": best}, "fold_ic": fold_ic[best],
            "test": metrics(p, data.Y[te], data.intraday[te], data.premarket[te])}


def run_gbm(data: Data, spec: dict) -> dict:
    import lightgbm as lgb
    params = {"objective": "regression", "learning_rate": 0.02, "num_leaves": 15,
              "min_data_in_leaf": 300, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 0,
              "num_threads": torch.get_num_threads()}
    rounds = list(range(50, 801, 50))
    splits = data.splits()
    fold_ic = {r: [] for r in rounds}
    for name, tr, va in splits[:-1]:
        x = tabular_features(data, tr, spec)
        booster = lgb.train(params, lgb.Dataset(x[tr], data.Y[tr]), num_boost_round=rounds[-1])
        for r in rounds:
            fold_ic[r].append(_corr(booster.predict(x[va], num_iteration=r), data.Y[va]))
    best = max(rounds, key=lambda r: np.mean(fold_ic[r]))
    name, tr, te = splits[-1]
    x = tabular_features(data, tr, spec)
    booster = lgb.train(params, lgb.Dataset(x[tr], data.Y[tr]), num_boost_round=best)
    p = booster.predict(x[te])
    return {"choice": {"rounds": best}, "fold_ic": fold_ic[best],
            "test": metrics(p, data.Y[te], data.intraday[te], data.premarket[te])}


# --------------------------------------------------------------------------
# 신경망
# --------------------------------------------------------------------------

class Variant(nn.Module):
    """논문 모듈을 그대로 쓰되 일부 경로를 빼거나 바꾼 제거 실험용 모델."""

    def __init__(self, config: ModelConfig, mode: str) -> None:
        super().__init__()
        self.mode = mode
        d = config.d_model
        self.news = NewsSignalEncoder(config.news)
        self.tcn = TCNEncoder(config.tcn)
        self.attention = AsymmetricCrossAttention(config.attention)
        self.fusion = GatedResidualFusion(config.fusion)
        self.predictor = ReturnPredictor(config.predictor)
        if mode == "concat":
            self.concat = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.LayerNorm(d))

    def forward(self, e, s, w):
        if self.mode == "news_only":
            return self.predictor(self.news(e, s))
        aux = self.tcn(w)
        if self.mode == "ind_only":
            return self.predictor(aux[:, -1])
        primary = self.news(e, s)
        if self.mode == "no_attention":
            context = aux[:, -1]
        else:
            context, _ = self.attention(primary, aux)
        if self.mode == "concat":
            return self.predictor(self.concat(torch.cat([primary, context], -1)))
        fused, _ = self.fusion(primary, context)
        return self.predictor(fused)


class Paper(nn.Module):
    """논문 전체 모델(model.py)을 예측값만 돌려주도록 감싼다."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = NewsDrivenStockPredictor(config)

    def forward(self, e, s, w):
        return self.model(e, s, w).prediction


def build_model(spec: dict, k: int) -> nn.Module:
    config = ModelConfig(d_model=spec["d"])
    config.news.embedding_dim = k
    config.tcn.hidden_channels = min(128, spec["d"])
    config.attention.scaled = spec.get("scaled", True)
    if spec["d"] < 512:
        config.predictor.hidden_dims = (spec["d"], spec["d"] // 2)
    mode = spec.get("mode", "full")
    return Paper(config) if mode == "full" else Variant(config, mode)


def train_nn(spec, e, s, w, y, tr_idx, ev_idx, seed, epochs):
    """tr_idx 로 학습하며 에폭마다 ev_idx 예측을 돌려준다."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(spec, e.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec.get("wd", 1e-5))
    loss_fn = nn.HuberLoss(delta=1.0) if spec.get("loss") == "huber" else nn.MSELoss()
    bs = spec.get("bs", 32)
    et, st, wt, yt = (torch.from_numpy(a) for a in (e, s, w, y.astype(np.float32)))
    preds = []
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for b in range(0, len(perm), bs):
            idx = torch.from_numpy(perm[b: b + bs])
            loss = loss_fn(model(et[idx], st[idx], wt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            idx = torch.from_numpy(ev_idx)
            preds.append(torch.cat([model(et[idx[i:i + 2048]], st[idx[i:i + 2048]], wt[idx[i:i + 2048]])
                                    for i in range(0, len(idx), 2048)]).numpy().astype(np.float64))
    return preds


def nn_labels(data: Data, tr: np.ndarray, how: str) -> np.ndarray:
    """학습용 라벨 크기 조정. 순위·부호는 그대로라 IC 평가에는 영향이 없다."""
    if how == "ticker":
        scale = {}
        for t in set(data.T[tr]):
            v = data.Y[tr & (data.T == t)]
            scale[t] = v.std() if len(v) >= 30 and v.std() > 0 else data.Y[tr].std()
        return data.Y / np.array([scale.get(t, data.Y[tr].std()) for t in data.T])
    return data.Y / data.Y[tr].std()


def run_nn(data: Data, spec: dict) -> dict:
    epochs = spec.get("epochs", 20)
    fold_seeds = spec.get("fold_seeds", [0])
    test_seeds = spec.get("test_seeds", [0, 1, 2])
    splits = data.splits()
    curves = []                                    # [폴드][시드] -> 에폭별 IC
    for name, tr, va in splits[:-1]:
        e = data.embedding(tr, spec["k"])
        w = data.windows(tr, spec["norm"])
        y = nn_labels(data, tr, spec.get("label", "global"))
        per_seed = []
        for seed in fold_seeds:
            t0 = time.time()
            preds = train_nn(spec, e, data.S, w, y, np.where(tr)[0], np.where(va)[0], seed, epochs)
            per_seed.append([_corr(p, data.Y[va]) for p in preds])
            logger.info("    %s seed %d  최고 IC %+.4f (에폭 %d)  %.0fs", name, seed,
                        max(per_seed[-1]), int(np.argmax(per_seed[-1])) + 1, time.time() - t0)
        curves.append(np.mean(per_seed, 0))
    curves = np.array(curves)                       # (폴드, 에폭)
    mean_curve = curves.mean(0)
    best = int(np.argmax(mean_curve))              # 0-based

    name, tr, te = splits[-1]
    e = data.embedding(tr, spec["k"])
    w = data.windows(tr, spec["norm"])
    y = nn_labels(data, tr, spec.get("label", "global"))
    test_preds = []
    for seed in test_seeds:
        preds = train_nn(spec, e, data.S, w, y, np.where(tr)[0], np.where(te)[0], seed, best + 1)
        test_preds.append(preds[best])
    seed_metrics = [metrics(p, data.Y[te], data.intraday[te], data.premarket[te]) for p in test_preds]
    ens = metrics(np.mean(test_preds, 0), data.Y[te], data.intraday[te], data.premarket[te])
    avg = {k: float(np.mean([m[k] for m in seed_metrics])) for k in seed_metrics[0]}
    return {"choice": {"epochs": best + 1}, "fold_ic": curves[:, best].tolist(),
            "valid_curve": mean_curve.tolist(), "test": avg, "test_ensemble": ens,
            "test_seeds": seed_metrics}


# --------------------------------------------------------------------------
# 실험 목록
# --------------------------------------------------------------------------

COMPACT = {"k": 8, "d": 64, "norm": "window", "lr": 1e-3, "wd": 1e-4, "epochs": 20, "bs": 256,
           "fold_seeds": [0, 1], "test_seeds": [0, 1, 2]}

EXPERIMENTS = {
    "linear": [
        ("L1 감성 점수만", {"sent": True}),
        ("L2 감성 + 임베딩 PCA8", {"sent": True, "k": 8}),
        ("L3 지표 30일×9종만", {"sent": False, "ind": "flat"}),
        ("L4 감성 + PCA8 + 지표", {"sent": True, "k": 8, "ind": "flat"}),
        ("L5 L4 + 장중 여부", {"sent": True, "k": 8, "ind": "flat", "horizon": True}),
    ],
    "gbm": [
        ("G1 LightGBM 감성 + PCA8 + 최근 지표 + 장중 여부",
         {"sent": True, "k": 8, "ind": "last", "horizon": True}),
        ("G2 LightGBM 감성 + PCA32 + 최근 지표 + 장중 여부",
         {"sent": True, "k": 32, "ind": "last", "horizon": True}),
    ],
    "nn": [
        ("N2 논문 구조 축소판 (PCA8, d=64, 창 정규화)", dict(COMPACT)),
        ("N3 └ 뉴스 경로만 (TCN·어텐션 제거)", {**COMPACT, "mode": "news_only"}),
        ("N4 └ 지표 경로만 (뉴스 제거)", {**COMPACT, "mode": "ind_only"}),
        ("N5 └ 어텐션 제거 (TCN 마지막 시점 사용)", {**COMPACT, "mode": "no_attention"}),
        ("N6 └ 게이트 제거 (단순 결합)", {**COMPACT, "mode": "concat"}),
        ("N7 └ 스케일링 없는 내적 어텐션 (논문 표기)", {**COMPACT, "scaled": False}),
        ("N8 └ 종목별 라벨 정규화", {**COMPACT, "label": "ticker"}),
        ("N9 └ Huber 손실", {**COMPACT, "loss": "huber"}),
        ("N10 └ PCA32", {**COMPACT, "k": 32}),
        ("N11 └ 임베딩 1536 그대로", {**COMPACT, "k": 1536}),
        ("N1 논문 원래 크기 + 창 정규화", {"k": 1536, "d": 512, "norm": "window", "lr": 1e-4,
                                   "epochs": 15, "fold_seeds": [0], "test_seeds": [0]}),
        ("N0 논문 원래 설정 그대로 (1536, d=512, 전체 MinMax)",
         {"k": 1536, "d": 512, "norm": "global", "lr": 1e-4, "epochs": 15,
          "fold_seeds": [0], "test_seeds": [0]}),
    ],
}
RUNNERS = {"linear": run_linear, "gbm": run_gbm, "nn": run_nn}


# 보고서 머리말. main() 에서 라벨 파일에 맞게 바꾼다.
DATA_NOTE = "발행 시각 확인 뉴스(22종목, 발행 날짜 불일치 제외)."
LABEL_NOTE = "장 시작 전·장중 뉴스는 당일 시가→종가, 장 마감 후 뉴스는 당일 종가→다음날 종가."
SUBSET_NOTE = ("장 시작 전 IC: 09:30 이전 발행 뉴스만. 라벨(시가→종가)이 발행 이후 구간만 포함하는 "
               "유일한 집단이다. 장중 뉴스는 시가부터 발행 시각까지의 움직임이 라벨에 섞여 있다.")


def write_report(out: Path) -> None:
    """작업자별 결과 파일(results_*.json)을 모아 하나의 표로 만든다."""
    results: dict = {}
    for path in sorted(out.glob("results_*.json")):
        results.update(json.loads(path.read_text()))
    rows = sorted(results.items(), key=lambda kv: -np.mean(kv[1]["fold_ic"]))
    lines = ["# 모델 비교 (walk-forward)", "",
             f"- 데이터: {DATA_NOTE}",
             f"- 라벨: {LABEL_NOTE}",
             "- 검증: 2022 1분기 ~ 2023 2분기 6개 분기 평균 IC 로 설정 선택. 각 폴드는 검증 시작 3일 전까지만 학습.",
             "- 평가: 2023 하반기(약 9,600건, 장중 약 6,800건). 평가 IC 의 표준오차는 약 ±0.010(장중 ±0.012)이며,",
             "  같은 종목·날짜 기사가 라벨을 공유하므로 실제 불확실성은 이보다 크다.",
             "- 방향 적중은 예측값 중앙값을 기준으로 위/아래를 나눠 계산했다(예측 편향 제거).",
             "- 신경망 평가값은 시드 3개 평균(원래 크기 모델은 시드 1개). 축소판은 배치 256, 원래 크기는 논문대로 배치 32.", "",
             f"- {SUBSET_NOTE}", "",
             "| 모델 | 검증 평균 IC | 분기별 IC | 선택 | 평가 IC | 평가 순위 IC | 평가 방향 | 마감 전 IC | 장 시작 전 IC | 장 시작 전 방향 |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows:
        t = r["test"]
        lines.append(
            f"| {name} | {np.mean(r['fold_ic']):+.4f} | {' '.join(f'{x:+.3f}' for x in r['fold_ic'])} | "
            f"{', '.join(f'{k}={v:g}' for k, v in r['choice'].items())} | {t['ic']:+.4f} | "
            f"{t['rank_ic']:+.4f} | {t['dir']:.3f} | {t['ic_intraday']:+.4f} | "
            f"{t.get('ic_premarket', float('nan')):+.4f} | {t.get('dir_premarket', float('nan')):.3f} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def commit(out: Path, message: str) -> None:
    subprocess.run(["git", "add", "-Af", str(out)], check=False)
    msg = (f"{message}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK")
    subprocess.run(["git", "-c", "user.email=sunshine31885@gmail.com", "-c", "user.name=Claude",
                    "commit", "-q", "-m", msg], check=False)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for delay in (2, 4, 8, 16):
        if subprocess.run(["git", "push", "-q", "origin", branch]).returncode == 0:
            return
        time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz")
    ap.add_argument("--indicators", default="data/fnspid/indicators.npz")
    ap.add_argument("--labels", default="data/fnspid/labels_open_close.jsonl.gz",
                    help="라벨 교체 파일. 빈 값이면 캐시의 라벨을 그대로 쓴다.")
    ap.add_argument("--groups", default="linear,gbm,nn")
    ap.add_argument("--only", default="", help="이름 앞부분(예: N2,N3)으로 실험 제한")
    ap.add_argument("--out", default="experiments/suite")
    ap.add_argument("--commit", action="store_true", help="실험 하나가 끝날 때마다 깃에 올린다")
    ap.add_argument("--worker", default="main", help="병렬 실행 시 작업자 이름(결과 파일 구분)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / f"suite_{args.worker}.log", encoding="utf-8")])
    torch.set_num_threads(args.threads)
    data = Data(args.cache, args.indicators, args.labels)
    global DATA_NOTE, LABEL_NOTE, SUBSET_NOTE
    DATA_NOTE = f"발행 시각 확인 뉴스 {len(data.Y):,}건({len(set(data.T))}종목). 라벨 파일: `{args.labels}`."
    if "minute" in args.labels:
        LABEL_NOTE = ("장 시작 전 뉴스는 당일 시가→종가, 장중 뉴스는 발행 다음 분봉 시가→종가(Dukascopy 분봉), "
                      "장 마감 후 뉴스는 당일 종가→다음날 종가. 분봉이 없는 장중 뉴스는 제외.")
        SUBSET_NOTE = ("마감 전 IC: 장 시작 전 + 장중 뉴스. 분봉 라벨이라 모두 발행 이후 구간만 포함한다. "
                       "장 시작 전 IC: 09:30 이전 발행 뉴스만.")

    results_path = out / f"results_{args.worker}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    for group in args.groups.split(","):
        for name, spec in EXPERIMENTS[group]:
            if only and not any(name.startswith(o + " ") for o in only):
                continue
            if name in results:
                logger.info("건너뜀(완료): %s", name)
                continue
            logger.info("▶ %s  %s", name, json.dumps(spec, ensure_ascii=False))
            t0 = time.time()
            r = RUNNERS[group](data, spec)
            r["spec"], r["group"], r["sec"] = spec, group, round(time.time() - t0)
            results[name] = r
            t = r["test"]
            logger.info("  검증 평균 IC %+.4f | 평가 IC %+.4f 순위 IC %+.4f 방향 %.3f | 장중 IC %+.4f (%.0fs)",
                        np.mean(r["fold_ic"]), t["ic"], t["rank_ic"], t["dir"], t["ic_intraday"],
                        time.time() - t0)
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(out)
            if args.commit:
                commit(out, f"[suite] {name}")


if __name__ == "__main__":
    main()
