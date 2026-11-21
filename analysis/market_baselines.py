"""널리 쓰이는 시계열 모델과의 비교 (같은 데이터·같은 평가 구간).

같은 구조 비교 실험(sentiment_only_ablation.py)과 데이터·구간·평가 방식을 똑같이 두고, 뉴스 없이
기술적 지표 30일 × 9종만 쓰는 대표 모델들을 학습해 논문 구조와 비교한다.

    lstm  LSTM 2층(은닉 64) → 선형. 주가 예측 연구에서 가장 흔한 기준 모델
    lgb   LightGBM (지표 30일 × 9종 = 270개 값). 실무에서 많이 쓰는 트리 부스팅
    팩터  단기 반전(1일·5일), 모멘텀(20일). 퀀트 업계 표준 팩터, 학습 없음

신경망은 논문 구조와 같은 규칙(AdamW, 날짜 단위 배치, MSE, 최대 8에폭 중 검증으로 선택, 시드 3개)을 따른다.

    python analysis/market_baselines.py --model lstm
    python analysis/market_baselines.py --model lgb
    python analysis/market_baselines.py --report
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_only_ablation as ab  # noqa: E402  (FOLDS 를 같게 맞춘다)
import run_edt_suite2 as s2  # noqa: E402
import run_event_suite as ev  # noqa: E402
from edt_study import ric  # noqa: E402
from run_edt_suite import EDT  # noqa: E402

SEEDS = ab.SEEDS
ABL = Path("experiments/sentiment_ablation")
logger = logging.getLogger("market_baselines")


class LSTMNet(nn.Module):
    def __init__(self, n_in=9, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, w):
        h, _ = self.lstm(w)
        return self.head(h[:, -1]).squeeze(-1)


def train_lstm(d, tr, ev_mask, seed, epochs, day_codes, lr=1e-3, wd=1e-5, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    W = torch.from_numpy(d.windows(tr))
    Y = torch.from_numpy((d.yz / d.yz[tr].std()).astype(np.float32))
    model = LSTMNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr_idx, ev_idx = np.where(tr)[0], torch.from_numpy(np.where(ev_mask)[0])
    preds = []
    for _ in range(epochs):
        model.train()
        for b in ev.date_batches(day_codes, tr_idx, bs, rng):
            bi = torch.from_numpy(b)
            loss = nn.functional.mse_loss(model(W[bi]), Y[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            preds.append(torch.cat([model(W[ev_idx[i:i + 4096]]) for i in range(0, len(ev_idx), 4096)]).numpy().astype(np.float64))
    return preds


def run_lstm(d, out):
    _, day_codes = np.unique(d.D, return_inverse=True)
    save = {}
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        curves = []
        for seed in SEEDS:
            t0 = time.time()
            curves.append([ric(p, d.y[va]) for p in train_lstm(d, tr, va, seed, 8, day_codes)])
            logger.info("lstm 구간 %d 시드 %d 검증 곡선 %s (%.0fs)", k + 1, seed, " ".join("%+.3f" % v for v in curves[-1]), time.time() - t0)
        best = int(np.argmax(np.mean(curves, 0)))
        save[f"best{k}"] = np.array(best + 1)
        for seed in SEEDS:
            p = train_lstm(d, tr2, te, seed, best + 1, day_codes)[best]
            save[f"test{k}_seed{seed}"] = p
            logger.info("lstm 구간 %d 시드 %d 평가 순위IC %+.4f (에폭 %d)", k + 1, seed, ric(p, d.y[te]), best + 1)
        np.savez(out / "pred_lstm.npz", **save)


def run_lgb(d, out):
    import lightgbm as lgb
    X = d.W.reshape(len(d.y), -1)          # 창 안 상대값 30일 × 9종 (트리는 표준화가 필요 없다)
    save = {}
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        for seed in SEEDS:
            params = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.03, "num_leaves": 15,
                      "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "lambda_l2": 10.0, "verbose": -1, "seed": seed, "num_threads": torch.get_num_threads()}
            bst = lgb.train(params, lgb.Dataset(X[tr], d.yz[tr]), num_boost_round=600)
            best = max(range(50, 601, 50), key=lambda r: ric(bst.predict(X[va], num_iteration=r), d.y[va]))
            p = lgb.train(params, lgb.Dataset(X[tr2], d.yz[tr2]), num_boost_round=best).predict(X[te])
            save[f"test{k}_seed{seed}"], save[f"best{k}_seed{seed}"] = p, np.array(best)
            logger.info("lgb 구간 %d 시드 %d 평가 순위IC %+.4f (반복 %d)", k + 1, seed, ric(p, d.y[te]), best)
    np.savez(out / "pred_lgb.npz", **save)


def report(d, out):
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    folds = s2.fold_masks(d)
    idx = np.concatenate([np.where(te)[0] for *_, te in folds])
    y, D = d.y[idx], d.D[idx]

    def nn_pred(path):
        P = np.load(path)
        return np.concatenate([rk(np.mean([P[f"test{k}_seed{s}"] for s in SEEDS], 0)) for k in range(len(folds))])

    close = d.W[:, :, 0]                   # 창 마지막 종가 대비 비율 − 1 → 과거 k일 수익률로 되돌린다
    ret = lambda lag: 1.0 / (1.0 + close[:, -1 - lag]) - 1.0
    rows = [("논문 구조 (뉴스 임베딩 + 감성 + 지표)", "신경망", nn_pred(ABL / "pred_full.npz"))]
    for name, kind, path in (("같은 구조, 기술적 지표만", "신경망", ABL / "pred_tech.npz"),
                             ("LSTM (지표만)", "신경망", out / "pred_lstm.npz"),
                             ("LightGBM (지표만)", "트리 부스팅", out / "pred_lgb.npz")):
        if path.exists():
            rows.append((name, kind, nn_pred(path)))
    for name, v in (("단기 반전 1일 (전날 수익률의 반대)", -ret(1)), ("단기 반전 5일", -ret(5)), ("모멘텀 20일", ret(20))):
        rows.append((name, "팩터 (학습 없음)", v[idx]))

    ud, inv = np.unique(D, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    rng = np.random.default_rng(0)
    samples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]
    boot = lambda p: np.array([ric(p[s], y[s]) for s in samples])
    ref = boot(rows[0][2])

    L = ["# 널리 쓰이는 시계열 모델과의 비교", "",
         "같은 구조 비교 실험과 같은 데이터·평가 구간(2021-01~02, 2021-03~05, 뉴스 %d건, 거래일 %d일). "
         "논문 구조만 뉴스를 쓰고, 나머지는 모두 기술적 지표 30일 × 9종만 쓴다. 신경망은 시드 3개 평균, "
         "신뢰구간은 날짜 블록 부트스트랩 2,000회." % (len(y), len(ud)), "",
         "| 모델 | 종류 | 순위 IC | 95% 신뢰구간 | 구간 1 | 구간 2 | 논문 구조와의 차이 [95% 신뢰구간] |",
         "|---|---|---:|---|---:|---:|---|"]
    n1 = folds[0][3].sum()
    for i, (name, kind, p) in enumerate(rows):
        b = boot(p)
        lo, hi = np.percentile(b, [2.5, 97.5])
        diff = "—" if i == 0 else "%+.4f [%+.4f, %+.4f]" % (ric(rows[0][2], y) - ric(p, y), *np.percentile(ref - b, [2.5, 97.5]))
        L.append("| %s | %s | %+.4f | [%+.4f, %+.4f] | %+.3f | %+.3f | %s |"
                 % (name, kind, ric(p, y), lo, hi, ric(p[:n1], y[:n1]), ric(p[n1:], y[n1:]), diff))
    L += ["", "'논문 구조와의 차이'는 논문 구조 순위 IC − 해당 모델 순위 IC. 신뢰구간이 0 을 넘으면 논문 구조가 확실히 높다는 뜻."]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("lstm", "lgb"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="experiments/market_baselines")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"run_{a.model or 'report'}.log", encoding="utf-8")])
    torch.set_num_threads(a.threads)
    d = EDT(ab.SPY)
    if a.report:
        report(d, out)
    elif a.model == "lstm":
        run_lstm(d, out)
    else:
        run_lgb(d, out)


if __name__ == "__main__":
    main()
