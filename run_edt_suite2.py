"""EDT 에서 개선 구조 후보를 여러 평가 구간(walk-forward)으로 비교한다.

평가 구간이 하나(3개월)뿐이면 0.01~0.02 차이를 잡음과 구분하기 어렵다. 세 구간으로 나눠
각 구간마다 '그 전 2개월로 에폭 선택 → 학습+검증으로 다시 맞춤 → 평가'를 반복한다.

    구간 1: 학습 ~2020-08, 검증 2020-09~10, 평가 2020-11~12
    구간 2: 학습 ~2020-10, 검증 2020-11~12, 평가 2021-01~02
    구간 3: 학습 ~2020-12, 검증 2021-01~02, 평가 2021-03~05
    (구간 사이 3일 엠바고)

모델 (모두 논문의 다섯 모듈 — 주 신호 FC → TCN → 비대칭 교차 어텐션 → 게이트 결합 → MLP — 을 바탕으로 한다)
    R    Ridge 기준선 (문맥 + LLM + 상호작용 + 임베딩 PCA32)
    P0   논문 구조 그대로 (임베딩 1536 + 감성 = LLM 방향×중요도/3, d=512)
    P1   개선 구조: 주 신호 입력 = [임베딩 PCA64 ‖ LLM 이벤트 속성 ‖ 발행 시점·주가 수준], d=64, 순위 손실
    V1   P1 + 다중 과제: 1·2·3일 초과수익률과 상승 여부를 함께 학습 (Chen et al. 2019 의 다중 과제 학습)
    V2   P1 + 문맥별 전문가 혼합: 발행 시점·주가 수준으로 가중치를 정하는 예측 머리 3개
    V3   P1 + 회사 뉴스 이력 어텐션: 같은 회사의 직전 30일 보도자료(최대 8건)를 대상 기사로 질의해 모음
         (Ding et al. 2015 장·단기 이벤트, Hu et al. 2018 뉴스 어텐션)
    V4   P1 + 임베딩 PCA256

    python run_edt_suite2.py --spy <SPY 분봉 폴더> --out experiments/edt_suite2 --commit
"""
import argparse
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
from modules import AsymmetricCrossAttention, GatedResidualFusion, NewsSignalEncoder, ReturnPredictor, TCNEncoder
from run_edt_suite import EDT, metrics
from edt_study import ric, ridge

logger = logging.getLogger("edt_suite2")
FOLDS = [("2020-08-31", "2020-10-31", "2020-12-31"),
         ("2020-10-31", "2020-12-31", "2021-02-28"),
         ("2020-12-31", "2021-02-28", "2021-06-01")]
K_HIST = 8


def after(d, days=3):
    return (datetime.fromisoformat(d) + timedelta(days=days)).strftime("%Y-%m-%d")


def fold_masks(d):
    out = []
    for tr_end, va_end, te_end in FOLDS:
        tr = d.D <= tr_end
        va = (d.D > after(tr_end)) & (d.D <= va_end)
        tr2 = d.D <= va_end
        te = (d.D > after(va_end)) & (d.D <= te_end)
        out.append((tr, va, tr2, te))
    return out


def extra_targets(d):
    """2·3일 초과수익률(변동성 단위, 윈저화)과 회사 뉴스 이력 인덱스."""
    ys = {}
    for h in (2, 3):
        a = np.array([r[f"a_{h}d"] if r[f"a_{h}d"] is not None else np.nan for r in d.rows]) / d.vol
        lo, hi = np.nanquantile(a, [0.01, 0.99])
        a = np.clip(a, lo, hi)
        ys[h] = np.where(np.isnan(a), 0.0, a), ~np.isnan(a)
    ts = np.array([datetime.fromisoformat(r["pub_time"]) for r in d.rows])
    H = np.full((len(d.rows), K_HIST), -1, dtype=np.int64)
    by = {}
    for i, r in enumerate(d.rows):
        by.setdefault(r["ticker"], []).append(i)
    for idx in by.values():
        idx.sort(key=lambda i: ts[i])
        for p, i in enumerate(idx):
            prev = [j for j in idx[max(0, p - 50):p] if ts[i] - timedelta(days=30) <= ts[j] < ts[i]]
            prev = prev[-K_HIST:]
            H[i, :len(prev)] = prev
    return ys, H


class EDTNet(nn.Module):
    def __init__(self, in_dim, n_ctx, d, use_tcn=True, hist=False, moe=0, multitask=False):
        super().__init__()
        cfg = ModelConfig(d_model=d)
        cfg.news.embedding_dim = in_dim
        cfg.tcn.hidden_channels = min(128, d)
        cfg.predictor.hidden_dims = (d, d // 2) if d < 512 else (256, 64)
        self.news = NewsSignalEncoder(cfg.news)              # 모듈 1
        self.use_tcn, self.hist, self.moe, self.multitask = use_tcn, hist, moe, multitask
        if use_tcn:
            self.tcn = TCNEncoder(cfg.tcn)                   # 모듈 2
            self.attention = AsymmetricCrossAttention(cfg.attention)  # 모듈 3
            self.fusion = GatedResidualFusion(cfg.fusion)    # 모듈 4
        if hist:
            self.hq = nn.Linear(d, d)
            self.hmix = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.LayerNorm(d))
        if moe:
            self.heads = nn.ModuleList([ReturnPredictor(cfg.predictor) for _ in range(moe)])
            self.gate = nn.Linear(n_ctx, moe)
        else:
            self.predictor = ReturnPredictor(cfg.predictor)  # 모듈 5
        if multitask:
            self.aux = nn.Linear(d, 3)                       # 2일, 3일, 상승 여부(로짓)

    def forward(self, x, s, w, ctx=None, hx=None, hs=None, hmask=None):
        p = self.news(x, s)
        if self.hist:
            k = self.news(hx.flatten(0, 1), hs.flatten(0, 1)).view(*hx.shape[:2], -1)
            score = (k @ self.hq(p).unsqueeze(-1)).squeeze(-1) / k.size(-1) ** 0.5
            score = score.masked_fill(~hmask, -1e9)
            wgt = torch.softmax(score, -1) * hmask.any(-1, keepdim=True).float()
            pooled = (wgt.unsqueeze(-1) * k).sum(1)
            p = self.hmix(torch.cat([p, pooled, hmask.float().sum(-1, keepdim=True).log1p()], -1))
        if self.use_tcn:
            c, _ = self.attention(p, self.tcn(w))
            f, _ = self.fusion(p, c)
        else:
            f = p
        if self.moe:
            g = torch.softmax(self.gate(ctx), -1)
            out = (g * torch.stack([h(f) for h in self.heads], -1)).sum(-1)
        else:
            out = self.predictor(f)
        aux = self.aux(f) if self.multitask else None
        return out, aux


def nn_inputs(d, mask, spec):
    if spec.get("raw_emb"):
        x = d.E.astype(np.float32)
    else:
        x = np.hstack([d.pca(mask, spec.get("k", 64)), d.content, d.inter, d.ctx])
        m, s = x[mask].mean(0), x[mask].std(0) + 1e-8
        x = ((x - m) / s).astype(np.float32)
    c = d.ctx
    c = ((c - c[mask].mean(0)) / (c[mask].std(0) + 1e-8)).astype(np.float32)
    return x, c, d.windows(mask)


def train(d, spec, H, ys, mask_tr, mask_ev, seed, epochs, day_codes):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, c, w = nn_inputs(d, mask_tr, spec)
    y = (d.yz / d.yz[mask_tr].std()).astype(np.float32)
    model = EDTNet(x.shape[1], c.shape[1], spec["d"], spec.get("tcn", True), spec.get("hist", False),
                   spec.get("moe", 0), spec.get("multitask", False))
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec.get("wd", 1e-4))
    X, S, W, C, Y = (torch.from_numpy(a) for a in (x, d.S, w, c, y))
    Ht = torch.from_numpy(H)
    Y2 = torch.from_numpy((ys[2][0] / ys[2][0][mask_tr].std()).astype(np.float32))
    Y3 = torch.from_numpy((ys[3][0] / ys[3][0][mask_tr].std()).astype(np.float32))
    M2, M3 = torch.from_numpy(ys[2][1]), torch.from_numpy(ys[3][1])
    Dc = torch.from_numpy(day_codes)
    # 이력은 '학습 구간 이전에 공개된' 기사만 쓰도록 평가 시점 기준으로 이미 과거만 담겨 있다.

    def fwd(bi):
        kw = {"ctx": C[bi]}
        if spec.get("hist"):
            h = Ht[bi]
            m = h >= 0
            hc = h.clamp(min=0)
            kw.update(hx=X[hc], hs=S[hc], hmask=m)
        return model(X[bi], S[bi], W[bi], **kw)

    tr_idx, ev_idx = np.where(mask_tr)[0], np.where(mask_ev)[0]
    preds = []
    for _ in range(epochs):
        model.train()
        for b in ev.date_batches(day_codes, tr_idx, spec.get("bs", 256), rng):
            bi = torch.from_numpy(b)
            p, aux = fwd(bi)
            loss = nn.functional.mse_loss(p, Y[bi])
            if spec.get("rank", 0):
                loss = loss + spec["rank"] * ev.pairwise_rank_loss(p, Y[bi], Dc[bi])
            if aux is not None:
                m2, m3 = M2[bi], M3[bi]
                loss = loss + 0.5 * ((aux[:, 0] - Y2[bi]) ** 2 * m2).sum() / m2.sum().clamp(min=1)
                loss = loss + 0.5 * ((aux[:, 1] - Y3[bi]) ** 2 * m3).sum() / m3.sum().clamp(min=1)
                loss = loss + 0.2 * nn.functional.binary_cross_entropy_with_logits(aux[:, 2], (Y[bi] > 0).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ei = torch.from_numpy(ev_idx)
            preds.append(torch.cat([fwd(ei[i:i + 2048])[0] for i in range(0, len(ei), 2048)]).numpy().astype(np.float64))
    return preds


def run_nn(d, spec, H, ys):
    _, day_codes = np.unique(d.D, return_inverse=True)
    out = []
    for k, (tr, va, tr2, te) in enumerate(fold_masks(d)):
        curves = []
        for seed in spec.get("seeds", [0, 1, 2]):
            preds = train(d, spec, H, ys, tr, va, seed, spec.get("epochs", 12), day_codes)
            curves.append([ric(p, d.y[va]) for p in preds])
        curve = np.mean(curves, 0)
        best = int(np.argmax(curve))
        tp = [train(d, spec, H, ys, tr2, te, s, best + 1, day_codes)[best] for s in spec.get("seeds", [0, 1, 2])]
        m = metrics(d, te, np.mean(tp, 0))
        m.update(valid_rank_ic=float(curve[best]), epochs=best + 1)
        logger.info("    구간 %d: 검증 %+.4f | 평가 %+.4f 집단내 %+.4f 롱숏 %+.1fbp t=%+.2f (에폭 %d)", k + 1,
                    m["valid_rank_ic"], m["rank_ic"], m["group_ric"], m["ls_bp"], m["ls_t"], best + 1)
        out.append(m)
    return out


def run_ridge(d, spec, H, ys):
    out = []
    for tr, va, tr2, te in fold_masks(d):
        xv = np.hstack([d.ctx, d.content, d.inter, d.pca(tr, 32)])
        xt = np.hstack([d.ctx, d.content, d.inter, d.pca(tr2, 32)])
        lam = max((10, 100, 1e3, 1e4, 1e5), key=lambda l: ric(ridge(xv[tr], d.yz[tr], xv[va], l), d.y[va]))
        m = metrics(d, te, ridge(xt[tr2], d.yz[tr2], xt[te], lam))
        m.update(valid_rank_ic=ric(ridge(xv[tr], d.yz[tr], xv[va], lam), d.y[va]), lam=lam)
        out.append(m)
    return out


SMALL = {"d": 64, "lr": 1e-3, "wd": 1e-4, "bs": 256, "rank": 1.0, "epochs": 12, "seeds": [0, 1, 2]}
EXPERIMENTS = [
    ("R  Ridge 기준선", run_ridge, {}),
    ("P1 개선 구조 (논문 구조 + 입력 확장)", run_nn, dict(SMALL)),
    ("V1 P1 + 다중 과제 (1·2·3일 + 상승 여부)", run_nn, {**SMALL, "multitask": True}),
    ("V2 P1 + 문맥별 전문가 혼합 (3개)", run_nn, {**SMALL, "moe": 3}),
    ("V3 P1 + 회사 뉴스 이력 어텐션 (30일·8건)", run_nn, {**SMALL, "hist": True}),
    ("V4 P1 + 임베딩 PCA256", run_nn, {**SMALL, "k": 256}),
    ("P0 논문 구조 그대로 (d=512)", run_nn,
     {"raw_emb": True, "d": 512, "lr": 1e-4, "wd": 1e-5, "bs": 64, "rank": 0.0, "epochs": 8, "seeds": [0]}),
]


def write_report(out):
    res = {}
    for p in sorted(out.glob("results_*.json")):
        res.update(json.loads(p.read_text()))
    L = ["# EDT 개선 구조 비교 (평가 구간 3개 walk-forward)", "",
         "평가 구간: 2020-11~12 / 2021-01~02 / 2021-03~05. 각 구간마다 직전 2개월로 에폭(λ) 선택 후 다시 학습해 평가.",
         "핵심 이벤트, 1일 초과수익률. '집단 내'는 발행 시점 × 주가 수준 9개 집단 안의 순위 IC 평균. 신경망은 시드 3개 앙상블.", "",
         "| 모델 | 평균 평가 순위IC | 구간별 | 평균 집단 내 | 구간별 | 평균 롱숏 bp | 구간별 t |",
         "|---|---:|---|---:|---|---:|---|"]
    rows = sorted(res.items(), key=lambda kv: -np.mean([m["group_ric"] for m in kv[1]]))
    for name, ms in rows:
        def col(key, fmt):
            return " ".join(format(m[key], fmt) for m in ms)
        L.append("| %s | %+.4f | %s | %+.4f | %s | %+.1f | %s |" % (
            name, np.mean([m["rank_ic"] for m in ms]), col("rank_ic", "+.3f"),
            np.mean([m["group_ric"] for m in ms]), col("group_ric", "+.3f"),
            np.mean([m["ls_bp"] for m in ms]), col("ls_t", "+.2f")))
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", default="experiments/edt_suite2")
    ap.add_argument("--only", default="")
    ap.add_argument("--worker", default="main")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"suite_{a.worker}.log", encoding="utf-8")])
    torch.set_num_threads(a.threads)
    d = EDT(a.spy)
    ys, H = extra_targets(d)
    logger.info("이력 있는 이벤트 %d건 (평균 %.1f건)", (H[:, 0] >= 0).sum(), (H >= 0).sum(1).mean())
    path = out / f"results_{a.worker}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    for name, fn, spec in EXPERIMENTS:
        if (only and not any(name.startswith(o + " ") for o in only)) or name in res:
            continue
        logger.info("▶ %s", name)
        t0 = time.time()
        res[name] = fn(d, spec, H, ys)
        ms = res[name]
        logger.info("  평균: 평가 %+.4f 집단내 %+.4f 롱숏 %+.1fbp (%.0fs)", np.mean([m["rank_ic"] for m in ms]),
                    np.mean([m["group_ric"] for m in ms]), np.mean([m["ls_bp"] for m in ms]), time.time() - t0)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(out)
        if a.commit:
            base.commit(out, f"[edt_suite2] {name}")


if __name__ == "__main__":
    main()
