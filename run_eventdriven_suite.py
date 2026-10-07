"""이벤트 기반 예측 구조 비교 (뉴스 이벤트 선행 연구 반영).

설계 근거
    - 구조화된 이벤트(Ding et al. 2014 EMNLP): 단어·감성 대신 '누가 무엇을 했나'를
      이벤트로 뽑아 쓴다. 여기서는 LLM 이 유형·방향·중요도·새 정보 여부를 뽑는다
      (extract_events.py).
    - 기업 이벤트 탐지 후 발행 시점 거래(Zhou et al. 2021, Trade the Event):
      평가를 '대상 종목이 주인공이고 새 정보를 담은 핵심 이벤트'에서도 따로 본다.
    - 장·중·단기 이벤트 영향(Ding et al. 2015 IJCAI): 같은 종목의 과거 이벤트를
      당일(발행 전)·1~7일·8~30일 창으로 나눠, 대상 이벤트를 질의로 한 어텐션으로 모은다
      (Hu et al. 2018 의 뉴스 어텐션 집계).
    - 이벤트 + 가격 TCN 결합(Deng et al. 2019, KDTCN): 이벤트 표현과 TCN 가격 표현을
      논문의 비대칭 교차 어텐션·게이트 결합으로 합친다.
    - 같은 날 순위 손실(Feng et al. 2019).

목표값·분할·지표는 run_event_suite.py 와 같다(y_abn_z, walk-forward 6분기 + 2023 하반기).

    python run_eventdriven_suite.py --out experiments/eventdriven_suite --commit
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import run_event_suite as ev
import run_suite as base
from config import ModelConfig
from extract_events import EVENT_TYPES
from modules import AsymmetricCrossAttention, GatedResidualFusion, ReturnPredictor, TCNEncoder

logger = logging.getLogger("eventdriven")
WINDOWS = (("short", 0, 1), ("mid", 1, 7), ("long", 7, 30))   # 발행 시각 기준 과거 (일)
K_HIST = 8
ATTRS = ["direction", "materiality", "about_company", "new_info", "price_recap", "core", "dir_x_mat", "dir_x_core"]


class EDData(ev.EventData):
    def __init__(self, cache, npz, events, llm):
        super().__init__(cache, npz, events)
        with gzip.open(llm, "rt", encoding="utf-8") as f:
            info = {r["news_id"]: r for r in map(json.loads, f)}
        n = len(self.meta)
        tid = {t: i for i, t in enumerate(EVENT_TYPES)}
        self.etype = np.zeros(n, dtype=np.int64)
        A = {k: np.zeros(n) for k in ATTRS}
        missing = 0
        for i, r in enumerate(self.meta):
            e = info.get(r["news_id"])
            if e is None:
                missing += 1
                self.etype[i] = tid["other"]
                continue
            self.etype[i] = tid[e["event_type"]]
            core = float(e["about_company"] and e["new_info"])
            vals = {"direction": e["direction"], "materiality": e["materiality"],
                    "about_company": e["about_company"], "new_info": e["new_info"],
                    "price_recap": e["price_recap"], "core": core,
                    "dir_x_mat": e["direction"] * e["materiality"],
                    "dir_x_core": e["direction"] * e["materiality"] * core}
            for k, v in vals.items():
                A[k][i] = v
        self.A = A
        self.core = A["core"] > 0
        logger.info("LLM 이벤트 결합: 없음 %d건 / 핵심 이벤트 %d건", missing, self.core.sum())
        self._build_history()

    def _build_history(self):
        """같은 종목의 과거 이벤트 인덱스(창별 최근 K개)와 창별 집계 특징."""
        n = len(self.meta)
        ts = np.array([datetime.fromisoformat(r["published_et"]) for r in self.meta])
        self.H = np.full((n, len(WINDOWS), K_HIST), -1, dtype=np.int64)
        agg = {f"h_{w}_{k}": np.zeros(n) for w, _, _ in WINDOWS for k in ("n", "core_n", "sig")}
        by_t = {}
        for i, t in enumerate(self.T):
            by_t.setdefault(t, []).append(i)
        for t, idx in by_t.items():
            idx = sorted(idx, key=lambda i: ts[i])
            times = [ts[i] for i in idx]
            for pos, i in enumerate(idx):
                for wi, (w, lo, hi) in enumerate(WINDOWS):
                    # 발행 시각 기준 (lo, hi] 일 전 사이에 '먼저' 나온 이벤트만
                    start, end = ts[i] - timedelta(days=hi), ts[i] - timedelta(days=lo)
                    sel = [j for j in idx[max(0, pos - 400):pos]
                           if start <= ts[j] < (end if lo else ts[i])]
                    recent = sel[-K_HIST:]
                    self.H[i, wi, :len(recent)] = recent
                    agg[f"h_{w}_n"][i] = len(sel)
                    agg[f"h_{w}_core_n"][i] = sum(self.core[j] for j in sel)
                    agg[f"h_{w}_sig"][i] = (np.mean([self.A["dir_x_mat"][j] for j in sel if self.core[j]])
                                            if any(self.core[j] for j in sel) else 0.0)
        self.HAGG = agg

    def features(self, tr, spec):
        cols = [super().features(tr, {k: v for k, v in spec.items() if k in ("sent_only", "react", "news", "k", "ind")})] \
            if any(spec.get(k) for k in ("sent_only", "react", "news", "k", "ind")) else []
        if spec.get("signal"):
            cols.append(self.A["dir_x_core"][:, None])
        if spec.get("llm"):
            cols += [self.A[k][:, None] for k in ATTRS]
            onehot = np.eye(len(EVENT_TYPES))[self.etype]
            cols += [onehot, onehot * self.A["direction"][:, None], onehot * self.A["dir_x_core"][:, None]]
        if spec.get("gate"):
            # 이벤트 게이트: 핵심 이벤트(주인공+새 정보)의 방향을 발행 시점별로 따로 둔다.
            # 이벤트 연구에서 나쁜 뉴스는 장 밖(장 시작 전·마감 후) 발행 뒤 지속 하락했다.
            for kind in ("pre", "intra", "after"):
                m = (self.kind == kind) & self.core
                cols.append((m & (self.A["direction"] < 0)).astype(float)[:, None])
                cols.append((m & (self.A["direction"] > 0)).astype(float)[:, None])
        if spec.get("hist"):
            cols += [np.log1p(v)[:, None] if k.endswith("_n") else v[:, None] for k, v in self.HAGG.items()]
        return np.hstack(cols)


_base_evaluate = ev.evaluate


def evaluate(data, idx, pred):
    out = _base_evaluate(data, idx, pred)
    m = data.core[idx]
    out["n_core"] = int(m.sum())
    out["rank_ic_core"] = ev._rank_ic(pred[m], data.Y[idx][m]) if m.sum() > 20 else float("nan")
    return out


ev.evaluate = evaluate  # run_linear / run_gbm 도 핵심 이벤트 지표를 함께 낸다


# --------------------------------------------------------------------------
# 제안 모델
# --------------------------------------------------------------------------

class EventDrivenNet(nn.Module):
    """이벤트 인코더 → 다중 기간 이력 어텐션 → (TCN 가격 표현과) 비대칭 교차 어텐션·게이트 결합."""

    def __init__(self, n_attr, n_react, d, use_hist=True, use_tcn=True, use_type=True):
        super().__init__()
        self.use_hist, self.use_tcn, self.use_type = use_hist, use_tcn, use_type
        self.type_emb = nn.Embedding(len(EVENT_TYPES), 16)
        self.enc = nn.Sequential(nn.Linear(16 + n_attr, d), nn.ReLU(), nn.LayerNorm(d), nn.Dropout(0.1))
        self.react = nn.Sequential(nn.Linear(n_react, d), nn.ReLU()) if n_react else None
        self.query = nn.Linear(d, d)
        self.combine = nn.Sequential(nn.Linear(d * (1 + len(WINDOWS)) + len(WINDOWS), d), nn.ReLU(),
                                     nn.LayerNorm(d), nn.Dropout(0.1))
        cfg = ModelConfig(d_model=d)
        cfg.tcn.hidden_channels = min(128, d)
        cfg.predictor.hidden_dims = (d, d // 2)
        self.tcn = TCNEncoder(cfg.tcn)
        self.attention = AsymmetricCrossAttention(cfg.attention)
        self.fusion = GatedResidualFusion(cfg.fusion)
        self.predictor = ReturnPredictor(cfg.predictor)
        self.scale = d ** -0.5

    def encode(self, etype, attr):
        e = self.type_emb(etype) if self.use_type else torch.zeros(*etype.shape, 16)
        return self.enc(torch.cat([e, attr], -1))

    def forward(self, etype, attr, react, hist_type, hist_attr, hist_mask, windows):
        q = self.encode(etype, attr)                                   # (B, d)
        if self.react is not None:
            q = q + self.react(react)
        parts, counts = [q], []
        for w in range(len(WINDOWS)):
            m = hist_mask[:, w]                                         # (B, K)
            counts.append(m.float().sum(-1, keepdim=True).log1p())
            if not self.use_hist:
                parts.append(torch.zeros_like(q))
                continue
            k = self.encode(hist_type[:, w], hist_attr[:, w])           # (B, K, d)
            score = (k @ self.query(q).unsqueeze(-1)).squeeze(-1) * self.scale
            score = score.masked_fill(~m, -1e9)
            wgt = torch.softmax(score, -1) * m.any(-1, keepdim=True).float()
            parts.append((wgt.unsqueeze(-1) * k).sum(1))
        news = self.combine(torch.cat(parts + counts, -1))              # 이벤트 주 신호 (B, d)
        if not self.use_tcn:
            return self.predictor(news)
        aux = self.tcn(windows)                                         # (B, 30, d)
        ctx, _ = self.attention(news, aux)
        fused, _ = self.fusion(news, ctx)
        return self.predictor(fused)


def nn_tensors(data: EDData, tr, spec):
    """이벤트별 속성 행렬(표준화)과 진입 전 반응, 지표 창."""
    attr_cols = []
    if spec.get("llm", True):
        attr_cols += [data.A[k] for k in ATTRS]
    attr_cols += [data.F["sentiment"], data.F["novelty"]]
    attr = np.stack(attr_cols, 1)
    if spec.get("k"):
        attr = np.hstack([attr, data.embedding(tr, spec["k"])])
    m, s = attr[tr].mean(0), attr[tr].std(0) + 1e-8
    attr = ((attr - m) / s).astype(np.float32)
    react = np.stack([data.F[k] for k in ev.REACT], 1) if spec.get("react", True) else np.zeros((len(attr), 0))
    if react.shape[1]:
        rm, rs = react[tr].mean(0), react[tr].std(0) + 1e-8
        react = (react - rm) / rs
    w = data.windows(tr, "window")
    y = ev.winsor(data.Y, tr)
    return attr, react.astype(np.float32), w, (y / y[tr].std()).astype(np.float32)


def train_ednet(spec, data, tens, tr_idx, ev_idx, seed, epochs, day_codes):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    attr, react, w, y = tens
    etype = data.etype if spec.get("llm", True) else np.full_like(data.etype, EVENT_TYPES.index("other"))
    model = EventDrivenNet(attr.shape[1], react.shape[1], spec["d"], spec.get("hist", True),
                           spec.get("tcn", True), spec.get("llm", True))
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec.get("wd", 1e-4))
    At, Rt, Wt, Yt, Et = (torch.from_numpy(a) for a in (attr, react, w, y, etype))
    H = torch.from_numpy(data.H)
    Dt = torch.from_numpy(day_codes)

    def batch_inputs(bi):
        h = H[bi]                                     # (B, W, K)
        mask = h >= 0
        hc = h.clamp(min=0)
        return (Et[bi], At[bi], Rt[bi], Et[hc], At[hc], mask, Wt[bi])

    preds = []
    for _ in range(epochs):
        model.train()
        for b in ev.date_batches(day_codes, tr_idx, spec.get("bs", 256), rng):
            bi = torch.from_numpy(b)
            p = model(*batch_inputs(bi))
            loss = nn.functional.mse_loss(p, Yt[bi])
            if spec.get("rank", 0):
                loss = loss + spec["rank"] * ev.pairwise_rank_loss(p, Yt[bi], Dt[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ei = torch.from_numpy(ev_idx)
            preds.append(torch.cat([model(*batch_inputs(ei[i:i + 2048])) for i in range(0, len(ei), 2048)])
                         .numpy().astype(np.float64))
    return preds


def run_ednet(data: EDData, spec: dict) -> dict:
    epochs = spec.get("epochs", 15)
    splits = data.splits()
    _, day_codes = np.unique(data.D, return_inverse=True)
    curves = []
    for name, tr, va in splits[:-1]:
        tens = nn_tensors(data, tr, spec)
        per_seed = []
        for seed in spec.get("fold_seeds", [0, 1]):
            t0 = time.time()
            preds = train_ednet(spec, data, tens, np.where(tr)[0], np.where(va)[0], seed, epochs, day_codes)
            per_seed.append([ev._rank_ic(p, data.Y[va]) for p in preds])
            logger.info("    %s seed %d 최고 순위IC %+.4f (에폭 %d) %.0fs", name, seed, max(per_seed[-1]),
                        int(np.argmax(per_seed[-1])) + 1, time.time() - t0)
        curves.append(np.mean(per_seed, 0))
    curves = np.array(curves)
    best = int(np.argmax(curves.mean(0)))
    _, tr, te = splits[-1]
    tens = nn_tensors(data, tr, spec)
    test_preds = [train_ednet(spec, data, tens, np.where(tr)[0], np.where(te)[0], s, best + 1, day_codes)[best]
                  for s in spec.get("test_seeds", [0, 1, 2])]
    return {"choice": {"epochs": best + 1}, "fold": curves[:, best].tolist(),
            "valid_curve": curves.mean(0).tolist(),
            "test": evaluate(data, np.where(te)[0], np.mean(test_preds, 0))}


NN = {"d": 64, "lr": 1e-3, "wd": 1e-4, "epochs": 15, "bs": 256, "rank": 1.0, "k": 8,
      "fold_seeds": [0, 1], "test_seeds": [0, 1, 2]}

EXPERIMENTS = {
    "linear": [
        ("X0 LLM 핵심 이벤트 방향×중요도 신호 하나", {"signal": True}),
        ("X1 LLM 이벤트 특징 (유형·방향·중요도·유형×방향)", {"llm": True}),
        ("X2 X1 + 장·중·단기 이벤트 이력", {"llm": True, "hist": True}),
        ("X3 X2 + 지표 30일", {"llm": True, "hist": True, "ind": "flat"}),
        ("X4 X3 + 감성·새로움 + 임베딩 PCA8", {"llm": True, "hist": True, "ind": "flat", "news": True, "k": 8}),
        ("X5 지표 30일만 (기준선)", {"ind": "flat"}),
        ("X6 제안: 지표 30일 + 핵심 이벤트 게이트(발행 시점별 좋은·나쁜 뉴스)", {"ind": "flat", "gate": True}),
        ("X7 핵심 이벤트 게이트만", {"gate": True}),
    ],
    "gbm": [
        ("XG LightGBM 이벤트 + 이력 + 반응 + 최근 지표", {"llm": True, "hist": True, "react": True, "ind": "last"}),
    ],
    "nn": [
        ("D1 제안: 이벤트 인코더 + 장·중·단기 이력 어텐션 + TCN·교차 어텐션·게이트", dict(NN)),
        ("D2 └ 이벤트 이력 제거", {**NN, "hist": False}),
        ("D3 └ LLM 이벤트 유형·속성 제거 (감성·임베딩만)", {**NN, "llm": False}),
        ("D4 └ TCN·교차 어텐션 제거", {**NN, "tcn": False}),
    ],
}
RUNNERS = {"linear": ev.run_linear, "gbm": ev.run_gbm, "nn": run_ednet}


def write_report(out: Path) -> None:
    results = {}
    for p in sorted(out.glob("results_*.json")):
        results.update(json.loads(p.read_text()))
    rows = sorted(results.items(), key=lambda kv: -np.mean(kv[1]["fold"]))
    L = ["# 이벤트 기반 구조 비교 (뉴스 이벤트 선행 연구 반영)", "",
         "- 목표값: 시장(SPY) 조정 초과수익률 / 20일 변동성. 진입은 발행 이후.",
         "- 선택: 검증 6개 분기 평균 순위 IC. 평가: 2023 하반기 1회. 신경망은 시드 3개 앙상블.",
         "- 핵심 이벤트: LLM 이 '대상 종목이 주인공이고 새 정보를 보도'로 판정한 기사.",
         "- 롱숏: 날짜별 예측 상위 1/3 − 하위 1/3 초과수익률(bp/일), 거래비용 미반영.", "",
         "| 모델 | 검증 순위IC | 분기별 | 선택 | 평가 순위IC | 핵심 이벤트 | 장 시작 전 | 장중 | 장 마감 후 | 롱숏 bp/일 | t |",
         "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows:
        t = r["test"]
        L.append(f"| {name} | {np.mean(r['fold']):+.4f} | {' '.join(f'{x:+.3f}' for x in r['fold'])} | "
                 f"{', '.join(f'{k}={v:g}' for k, v in r['choice'].items())} | {t['rank_ic']:+.4f} | "
                 f"{t.get('rank_ic_core', float('nan')):+.3f} | {t['rank_ic_pre']:+.3f} | {t['rank_ic_intra']:+.3f} | "
                 f"{t['rank_ic_after']:+.3f} | {t['ls_mean_bp']:+.1f} | {t['ls_t']:+.2f} |")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz")
    ap.add_argument("--indicators", default="data/fnspid/indicators.npz")
    ap.add_argument("--events", default="data/fnspid/event_features.jsonl.gz")
    ap.add_argument("--llm", default="data/fnspid/events_llm.jsonl.gz")
    ap.add_argument("--groups", default="linear,gbm,nn")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="experiments/eventdriven_suite")
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
    data = EDData(args.cache, args.indicators, args.events, args.llm)
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
            logger.info("  검증 순위IC %+.4f | 평가 순위IC %+.4f 핵심 %+.3f | 롱숏 %+.1fbp t=%+.2f (%.0fs)",
                        np.mean(r["fold"]), t["rank_ic"], t.get("rank_ic_core", float("nan")),
                        t["ls_mean_bp"], t["ls_t"], time.time() - t0)
            path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(out)
            if args.commit:
                base.commit(out, f"[eventdriven] {name}")


if __name__ == "__main__":
    main()
