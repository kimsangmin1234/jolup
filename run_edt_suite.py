"""EDT 보도자료에서 논문 구조와 제안 구조를 비교한다.

FNSPID(22종목)에서는 뉴스 신호가 약해 신경망이 기준선을 넘지 못했다. EDT(4천여 종목, 보도자료
5.5만 건)에서는 뉴스 내용이 같은 집단 안에서도 순위 IC 약 +0.09 의 예측력을 보였다(ABLATION.md).
신호가 있는 데이터에서 논문의 전체 구조(뉴스 주 신호 + 30일 지표 TCN + 비대칭 교차 어텐션 + 게이트)가
작동하는지, 그리고 선행 연구를 반영한 입력 확장이 도움이 되는지 본다.

    P0  논문 구조 그대로: 임베딩 1536 + 감성(= LLM 방향×중요도/3) → 512, TCN, 교차 어텐션, 게이트, MLP
    P1  제안: 주 신호 입력 = [임베딩 PCA64 ‖ LLM 이벤트 속성 ‖ 발행 시점·주가 수준], d=64, 같은 날 순위 손실
    P2  └ TCN·교차 어텐션 제거 (주 신호 MLP 만)
    P3  └ 임베딩 제거 (LLM 속성 + 문맥)
    P4  └ LLM 속성 제거 (임베딩 + 문맥)
    R   Ridge (A3: 문맥 + LLM + 상호작용 + 임베딩 PCA32) 기준선

목표: 1일 초과수익률(종목 − SPY)을 20일 변동성으로 나눈 값으로 학습하고, 평가는 1일 초과수익률로 한다.
분할: 학습 ~2020-12 → 검증 2021-01~02(에폭·λ 선택) → 학습 ~2021-02 → 평가 2021-03~05.

    python run_edt_suite.py --spy <SPY 분봉 폴더> --out experiments/edt_suite --commit
"""
import argparse
import csv
import glob
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import run_event_suite as ev  # noqa: E402
import run_suite as base  # noqa: E402
from data.technical_indicators import compute_indicators  # noqa: E402
from edt_study import Spy, TRAIN_END, VALID_END, load, ls, ric, ridge  # noqa: E402
from edt_ablation import blocks, group_ric  # noqa: E402

logger = logging.getLogger("edt_suite")


class EDT:
    def __init__(self, spy_dir):
        rows = load(Spy(spy_dir))
        rows.sort(key=lambda r: r["pub_time"])
        # 일봉 → 지표
        bars = defaultdict(list)
        with gzip.open("data/edt/daily_bars.csv.gz", "rt", newline="") as f:
            r = csv.reader(f)
            next(r)
            for t, d, o, h, l, c, v in r:
                bars[t].append((d, float(h), float(l), float(c), float(v)))
        ind, dates, vol = {}, {}, {}
        for t, b in bars.items():
            b.sort()
            if len(b) < 60:
                continue
            arr = np.array([x[1:] for x in b])
            ind[t] = compute_indicators(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
            dates[t] = [x[0] for x in b]
            ret = np.diff(arr[:, 2]) / arr[:-1, 2]
            vol[t] = np.array([np.nan] * 21 + [ret[i - 20:i].std() for i in range(21, len(arr))])
        keep, wins, vols = [], [], []
        for i, r in enumerate(rows):
            t = r["ticker"]
            if t not in ind or r["a_1d"] is None or not r["core"]:
                continue
            j = np.searchsorted(dates[t], r["date"]) - 1        # 발행일 전 거래일까지
            if j < 40 or not np.isfinite(vol[t][j]) or vol[t][j] <= 0:
                continue
            keep.append(i)
            wins.append(ind[t][j - 29:j + 1])
            vols.append(vol[t][j])
        self.rows = [rows[i] for i in keep]
        self.W = base.window_normalize(np.stack(wins))
        self.vol = np.array(vols)
        a1 = np.array([r["a_1d"] for r in self.rows])
        lo, hi = np.quantile(a1, [0.01, 0.99])
        self.y = np.clip(a1, lo, hi)                                  # 평가용
        z = a1 / self.vol
        lo, hi = np.quantile(z, [0.01, 0.99])
        self.yz = np.clip(z, lo, hi)                                  # 학습용
        self.D = np.array([r["date"] for r in self.rows])
        self.S = np.array([r["direction"] * r["materiality"] / 3 for r in self.rows], dtype=np.float32)
        px = np.array([r["start_price"] for r in self.rows])
        kind = np.array([r["kind"] for r in self.rows])
        self.grp = np.array([f"{k}|{0 if p < 5 else (1 if p < 20 else 2)}" for k, p in zip(kind, px)])
        self.ctx, self.content, self.inter = blocks(self.rows)
        ids = np.load("data/edt/edt_emb_ids.npy")
        vec = np.vstack([np.load(p).astype(np.float32) for p in sorted(glob.glob("data/edt/edt_emb.part*.npy"))])
        pos = {int(i): k for k, i in enumerate(ids)}
        self.E = np.array([vec[pos[r["edt_id"]]] for r in self.rows])
        cut = lambda d: (datetime.fromisoformat(d) + timedelta(days=3)).strftime("%Y-%m-%d")
        self.tr = self.D <= TRAIN_END
        self.va = (self.D > cut(TRAIN_END)) & (self.D <= VALID_END)
        self.tr2 = self.D <= VALID_END
        self.te = self.D > cut(VALID_END)
        logger.info("핵심 이벤트 %d건 (학습 %d / 검증 %d / 평가 %d)", len(self.rows), self.tr.sum(), self.va.sum(), self.te.sum())

    def pca(self, mask, k):
        mu = self.E[mask].mean(0)
        _, _, vt = np.linalg.svd(self.E[mask] - mu, full_matrices=False)
        return ((self.E - mu) @ vt[:k].T).astype(np.float32)

    def windows(self, mask):
        m, s = self.W[mask].mean((0, 1)), self.W[mask].std((0, 1)) + 1e-8
        return ((self.W - m) / s).astype(np.float32)


def metrics(d, mask, p):
    bp, t, _ = ls(p, d.y[mask], d.D[mask])
    return {"rank_ic": ric(p, d.y[mask]), "group_ric": group_ric(p, d.y[mask], d.grp[mask]), "ls_bp": bp, "ls_t": t}


def nn_x(d, mask, spec):
    if spec.get("raw_emb"):
        return d.E.astype(np.float32)
    cols = []
    if spec.get("emb", True):
        cols.append(d.pca(mask, 64))
    if spec.get("llm", True):
        cols += [d.content, d.inter]
    cols.append(d.ctx)
    x = np.hstack(cols)
    m, s = x[mask].mean(0), x[mask].std(0) + 1e-8
    return ((x - m) / s).astype(np.float32)


def run_nn(d, spec):
    _, day_codes = np.unique(d.D, return_inverse=True)
    epochs = spec.get("epochs", 12)
    x, w = nn_x(d, d.tr, spec), d.windows(d.tr)
    y = (d.yz / d.yz[d.tr].std()).astype(np.float32)
    curves = []
    for seed in spec.get("seeds", [0]):
        t0 = time.time()
        preds = ev.train_eventnet(spec, x, d.S, w, y, day_codes, np.where(d.tr)[0], np.where(d.va)[0], seed, epochs)
        curves.append([ric(p, d.y[d.va]) for p in preds])
        logger.info("    검증 seed %d 최고 %+.4f (에폭 %d) %.0fs", seed, max(curves[-1]), int(np.argmax(curves[-1])) + 1, time.time() - t0)
    curve = np.mean(curves, 0)
    best = int(np.argmax(curve))
    x2, w2 = nn_x(d, d.tr2, spec), d.windows(d.tr2)
    y2 = (d.yz / d.yz[d.tr2].std()).astype(np.float32)
    tp = [ev.train_eventnet(spec, x2, d.S, w2, y2, day_codes, np.where(d.tr2)[0], np.where(d.te)[0], s, best + 1)[best]
          for s in spec.get("seeds", [0])]
    pred = np.mean(tp, 0)
    return {"choice": f"에폭 {best + 1}", "valid_rank_ic": float(curve[best]), "test": metrics(d, d.te, pred),
            "pred_test": pred.tolist(), "pred_valid": np.mean([ev.train_eventnet(spec, x, d.S, w, y, day_codes,
            np.where(d.tr)[0], np.where(d.va)[0], s, best + 1)[best] for s in spec.get("seeds", [0])], 0).tolist()
            if spec.get("save_valid") else None}


def run_ridge(d, spec):
    def X(mask):
        return np.hstack([d.ctx, d.content, d.inter, d.pca(mask, 32)])
    xv, xt = X(d.tr), X(d.tr2)
    lam = max((10, 100, 1e3, 1e4, 1e5), key=lambda l: ric(ridge(xv[d.tr], d.yz[d.tr], xv[d.va], l), d.y[d.va]))
    pv = ridge(xv[d.tr], d.yz[d.tr], xv[d.va], lam)
    pt = ridge(xt[d.tr2], d.yz[d.tr2], xt[d.te], lam)
    return {"choice": f"λ={lam:g}", "valid_rank_ic": ric(pv, d.y[d.va]), "test": metrics(d, d.te, pt),
            "pred_test": pt.tolist(), "pred_valid": pv.tolist()}


def run_ensemble(d, spec):
    """같은 결과 폴더의 두 모델 예측을 순위로 바꿔 평균한다(검증 예측으로 가중치 선택)."""
    res = {}
    for p in sorted(Path(spec["dir"]).glob("results_*.json")):
        res.update(json.loads(p.read_text()))
    a, b = res[spec["a"]], res[spec["b"]]
    rk = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
    best, bw = None, -9
    for w_ in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ric(w_ * rk(np.array(a["pred_valid"])) + (1 - w_) * rk(np.array(b["pred_valid"])), d.y[d.va])
        if v > bw:
            best, bw = w_, v
    pt = best * rk(np.array(a["pred_test"])) + (1 - best) * rk(np.array(b["pred_test"]))
    return {"choice": f"가중치 {best:g}", "valid_rank_ic": bw, "test": metrics(d, d.te, pt), "pred_test": pt.tolist()}


SMALL = {"d": 64, "lr": 1e-3, "wd": 1e-4, "bs": 256, "rank": 1.0, "epochs": 12, "seeds": [0, 1, 2]}
EXPERIMENTS = [
    ("R  Ridge 기준선 (문맥 + LLM + 상호작용 + 임베딩 PCA32)", run_ridge, {}),
    ("P1 제안: 확장 주 신호 + TCN·교차 어텐션·게이트 + 순위 손실", run_nn, dict(SMALL)),
    ("P2 └ TCN·교차 어텐션 제거", run_nn, {**SMALL, "tcn": False}),
    ("P3 └ 임베딩 제거", run_nn, {**SMALL, "emb": False}),
    ("P4 └ LLM 속성 제거", run_nn, {**SMALL, "llm": False}),
    ("P5 └ 순위 손실 제거", run_nn, {**SMALL, "rank": 0.0}),
    ("P6 제안(P1) 재실행: 검증 예측 저장", run_nn, {**SMALL, "save_valid": True}),
    ("R2 Ridge 재실행: 예측 저장", run_ridge, {}),
    ("E1 앙상블: 제안(P6) + Ridge(R2)", run_ensemble, {"dir": "experiments/edt_suite",
     "a": "P6 제안(P1) 재실행: 검증 예측 저장", "b": "R2 Ridge 재실행: 예측 저장"}),
    ("P0 논문 구조 그대로 (임베딩 1536 + 감성, d=512)", run_nn,
     {"raw_emb": True, "d": 512, "lr": 1e-4, "wd": 1e-5, "bs": 64, "rank": 0.0, "epochs": 10, "seeds": [0]}),
]


def write_report(out):
    res = {}
    for p in sorted(out.glob("results_*.json")):
        res.update(json.loads(p.read_text()))
    L = ["# EDT 보도자료: 논문 구조 vs 제안 구조", "",
         "핵심 이벤트, 1일 초과수익률(종목 − SPY). '집단 내 순위IC'는 발행 시점 × 주가 수준 9개 집단 안의 순위 IC 평균.",
         "롱숏: 날짜별 예측 상위 1/3 매수 − 하위 1/3 매도, 거래비용 미반영. 소형 신경망은 시드 3개 앙상블.", "",
         "| 모델 | 선택 | 검증 순위IC | 평가 순위IC | 평가 집단 내 순위IC | 롱숏 bp/일 | t |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["valid_rank_ic"]):
        t = r["test"]
        L.append(f"| {name} | {r['choice']} | {r['valid_rank_ic']:+.4f} | {t['rank_ic']:+.4f} | {t['group_ric']:+.4f} | "
                 f"{t['ls_bp']:+.1f} | {t['ls_t']:+.2f} |")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", default="experiments/edt_suite")
    ap.add_argument("--only", default="")
    ap.add_argument("--worker", default="main")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"suite_{a.worker}.log", encoding="utf-8")])
    torch.set_num_threads(a.threads)
    d = EDT(a.spy)
    path = out / f"results_{a.worker}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    for name, fn, spec in EXPERIMENTS:
        if (only and not any(name.startswith(o + " ") for o in only)) or name in res:
            continue
        logger.info("▶ %s", name)
        t0 = time.time()
        r = fn(d, spec)
        r["sec"] = round(time.time() - t0)
        res[name] = r
        t = r["test"]
        logger.info("  검증 %+.4f | 평가 %+.4f 집단내 %+.4f 롱숏 %+.1fbp t=%+.2f (%.0fs)", r["valid_rank_ic"], t["rank_ic"],
                    t["group_ric"], t["ls_bp"], t["ls_t"], time.time() - t0)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(out)
        if a.commit:
            base.commit(out, f"[edt_suite] {name}")


if __name__ == "__main__":
    main()
