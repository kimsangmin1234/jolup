"""같은 구조에서 입력만 바꾼 비교: 감성 점수만 vs 임베딩 + 감성 (논문 구조 P0).

패널의 '감성 점수만'은 학습 없이 감성 점수를 그대로 예측값으로 쓴 비교 기준이다. 여기서는 논문 구조
(주 신호 FC → TCN → 교차 어텐션 → 게이트 잔차 결합 → MLP)와 학습 설정을 그대로 두고 주 신호 입력만 바꿔,
임베딩이 같은 구조 안에서 얼마나 기여하는지 본다.

    full  임베딩 1536 + 감성 1  → 주 신호 512   (P0 와 같은 설정)
    sent  감성 1               → 주 신호 512   (임베딩 크기 0)
    tech  뉴스 입력 없음        (임베딩 크기 0, 감성 점수 0 → 기술적 지표 30일만 남는다)

보고서에는 기술적 지표 30일 × 9종을 펼친 270개 값으로 학습한 Ridge 기준선도 함께 넣는다.

시장 지수(SPY) 분봉은 저장소의 정규장 자료(2020-08-31~)만 쓰므로, 학습은 2020-09 부터 시작하고
평가 구간은 패널의 2·3번째 구간(2021-01~02, 2021-03~05)을 그대로 쓴다.

    python analysis/sentiment_only_ablation.py --model full --out experiments/sentiment_ablation
    python analysis/sentiment_only_ablation.py --model sent --out experiments/sentiment_ablation
    python analysis/sentiment_only_ablation.py --model tech --out experiments/sentiment_ablation
    python analysis/sentiment_only_ablation.py --report --out experiments/sentiment_ablation
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_edt_suite2 as s2  # noqa: E402
from edt_study import ric, ridge  # noqa: E402
from run_edt_suite import EDT  # noqa: E402

SPY = "data/fnspid/minute"
FOLDS = [("2020-10-31", "2020-12-31", "2021-02-28"),
         ("2020-12-31", "2021-02-28", "2021-06-01")]
SEEDS = [0, 1, 2]
P0 = {"raw_emb": True, "d": 512, "lr": 1e-4, "wd": 1e-5, "bs": 64, "rank": 0.0, "epochs": 8}
logger = logging.getLogger("sent_ablation")

_nn_inputs = s2.nn_inputs


def nn_inputs(d, mask, spec):
    x, c, w = _nn_inputs(d, mask, spec)
    if spec.get("sent_only"):
        x = np.zeros((len(d.y), 0), dtype=np.float32)   # 임베딩 없이 감성 점수만 주 신호에 들어간다
    return x, c, w


s2.nn_inputs = nn_inputs
s2.FOLDS = FOLDS


def run(model, out):
    spec = dict(P0, sent_only=(model in ("sent", "tech")))
    d = EDT(SPY)
    if model == "tech":
        d.S = np.zeros_like(d.S)   # 감성 점수도 지워 뉴스 쪽 입력이 상수가 되게 한다
    ys, H = s2.extra_targets(d)
    _, day_codes = np.unique(d.D, return_inverse=True)
    save = {}
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        curves = []
        for seed in SEEDS:
            t0 = time.time()
            preds = s2.train(d, spec, H, ys, tr, va, seed, spec["epochs"], day_codes)
            curves.append([ric(p, d.y[va]) for p in preds])
            logger.info("%s 구간 %d 시드 %d 검증 곡선 %s (%.0fs)", model, k + 1, seed,
                        " ".join("%+.3f" % v for v in curves[-1]), time.time() - t0)
        curve = np.mean(curves, 0)
        best = int(np.argmax(curve))
        save[f"valid_curve{k}"] = np.array(curves)
        save[f"best{k}"] = np.array(best + 1)
        for seed in SEEDS:
            p = s2.train(d, spec, H, ys, tr2, te, seed, best + 1, day_codes)[best]
            save[f"test{k}_seed{seed}"] = p
            logger.info("%s 구간 %d 시드 %d 평가 순위IC %+.4f (에폭 %d)", model, k + 1, seed, ric(p, d.y[te]), best + 1)
        np.savez(out / f"pred_{model}.npz", **save)


def report(out):
    d = EDT(SPY)
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    P = {m: np.load(out / f"pred_{m}.npz") for m in ("full", "sent", "tech") if (out / f"pred_{m}.npz").exists()}
    folds = s2.fold_masks(d)
    idx = np.concatenate([np.where(te)[0] for _, _, _, te in folds])
    y, D = d.y[idx], d.D[idx]
    direction = np.array([r["direction"] for r in d.rows], float)

    def ridge_tech(k, tr, va, tr2, te):
        flat = lambda m: d.windows(m).reshape(len(d.y), -1)
        f = flat(tr)
        lam = max((10.0 ** e for e in range(0, 7)), key=lambda l: ric(ridge(f[tr], d.yz[tr], f[va], l), d.y[va]))
        f = flat(tr2)
        logger.info("Ridge 지표만 구간 %d: lambda=%g", k + 1, lam)
        return ridge(f[tr2], d.yz[tr2], f[te], lam)

    def pooled(get):
        return np.concatenate([rk(get(k, te)) for k, (_, _, _, te) in enumerate(folds)])

    preds = {
        "논문 구조 (임베딩 + 감성) · 시드 3개 평균": pooled(lambda k, te: np.mean([P["full"][f"test{k}_seed{s}"] for s in SEEDS], 0)),
        "같은 구조, 감성 점수만 · 시드 3개 평균": pooled(lambda k, te: np.mean([P["sent"][f"test{k}_seed{s}"] for s in SEEDS], 0)),
    }
    if "tech" in P:
        preds["같은 구조, 기술적 지표만 (뉴스 입력 없음) · 시드 3개 평균"] = pooled(
            lambda k, te: np.mean([P["tech"][f"test{k}_seed{s}"] for s in SEEDS], 0))
    preds["Ridge, 기술적 지표만 (30일 × 9종)"] = pooled(lambda k, te: ridge_tech(k, *folds[k]))
    preds.update({
        "규칙: 감성 점수 그대로 (학습 없음)": np.concatenate([d.S[te].astype(float) for *_, te in folds]),
        "규칙: LLM 호재·악재 판단 그대로 (학습 없음)": np.concatenate([direction[te] for *_, te in folds]),
    })
    ud, inv = np.unique(D, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    rng = np.random.default_rng(0)
    samples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]

    def ci(p):
        bs = np.array([ric(p[s], y[s]) for s in samples])
        return np.percentile(bs, [2.5, 97.5]), (bs <= 0).mean()

    L = ["# 같은 구조에서 입력만 바꾼 비교: 기술적 지표만 / 감성 점수만 / 임베딩 + 감성", "",
         "논문 구조(주 신호 FC → TCN → 교차 어텐션 → 게이트 잔차 결합 → MLP)와 학습 설정(학습률 1e-4, 배치 64, MSE, "
         "에폭은 검증으로 선택)을 그대로 두고 주 신호 입력만 바꿨다. 시드 3개.", "",
         "SPY 분봉은 저장소의 정규장 자료(2020-08-31~)만 있어 학습은 2020-09 부터 시작한다. 평가 구간은 패널의 2·3번째 구간과 같다.",
         "", "| 회차 | 학습 | 검증 | 평가 | 평가 건수 |", "|---|---|---|---|---:|"]
    for k, (tr, va, tr2, te) in enumerate(folds):
        L.append("| %d | 2020-09 ~ %s | ~%s | ~%s | %d |" % (k + 1, FOLDS[k][0], FOLDS[k][1], FOLDS[k][2], te.sum()))
    L += ["", "## 결과 (평가 %d건, 거래일 %d일)" % (len(y), len(ud)), "",
          "| 모델 | 순위 IC | 95% 신뢰구간 | 0 이하 확률 | 구간 1 | 구간 2 |", "|---|---:|---|---:|---:|---:|"]
    for name, p in preds.items():
        (lo, hi), p0 = ci(p)
        parts, o = [], 0
        for *_, te in folds:
            n = te.sum()
            parts.append(ric(p[o:o + n], y[o:o + n]))
            o += n
        L.append("| %s | %+.4f | [%+.4f, %+.4f] | %.3f | %+.3f | %+.3f |" % (name, ric(p, y), lo, hi, p0, *parts))
    names = list(preds)
    pairs = [(0, 1, "임베딩의 기여 (논문 구조 − 감성만)")]
    if "tech" in P:
        pairs += [(0, 2, "뉴스 전체의 기여 (논문 구조 − 기술적 지표만)"), (1, 2, "감성 점수의 기여 (감성만 − 기술적 지표만)")]
    L += [""]
    for i, j, label in pairs:
        diff = np.array([ric(preds[names[i]][s], y[s]) - ric(preds[names[j]][s], y[s]) for s in samples])
        dv = ric(preds[names[i]], y) - ric(preds[names[j]], y)
        L += ["- **%s: %+.4f**, 95%% 신뢰구간 [%+.4f, %+.4f], 0 이하 확률 %.3f"
              % (label, dv, *np.percentile(diff, [2.5, 97.5]), (diff <= 0).mean())]
    L += ["", "## 시드별 순위 IC (두 구간 합산)", "", "| 모델 | 시드 0 | 시드 1 | 시드 2 |", "|---|---:|---:|---:|"]
    for m, label in (("full", "논문 구조"), ("sent", "감성만"), ("tech", "기술적 지표만")):
        if m not in P:
            continue
        vals = [ric(np.concatenate([rk(P[m][f"test{k}_seed{s}"]) for k in range(len(folds))]), y) for s in SEEDS]
        L.append("| %s | %s |" % (label, " | ".join("%+.4f" % v for v in vals)))
    L += ["", "선택된 에폭: " + ", ".join("%s 구간%d=%d" % (m, k + 1, int(P[m][f"best{k}"])) for m in P for k in range(len(folds)))]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("full", "sent", "tech"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="experiments/sentiment_ablation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"run_{a.model or 'report'}.log", encoding="utf-8")])
    torch.set_num_threads(a.threads)
    if a.report:
        report(out)
    else:
        run(a.model, out)


if __name__ == "__main__":
    main()
