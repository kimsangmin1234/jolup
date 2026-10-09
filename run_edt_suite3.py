"""EDT 3라운드: 예측 저장 → 앙상블, 일별 순위 목표값.

run_edt_suite2.py 와 같은 3개 평가 구간·같은 모델 정의를 쓰되, 각 구간의 검증·평가 예측을 저장해
앙상블을 만든다. 구조 선택은 검증 성적으로만 하고, 평가 성적은 보고에만 쓴다.

    V5  P1 + 일별 순위 목표값: 학습 목표를 '그날 이벤트들 사이의 순위'를 정규분포 점수로 바꾼 값으로 한다
        (평가 지표가 순위 IC 이고, 극단값 영향을 줄인다. Gu·Kelly·Xiu 2020 등 금융 ML 의 순위 변환 관행)
    E-eq   앙상블: 지정 모델들의 예측 순위를 같은 가중치로 평균
    E-val  앙상블: 검증 순위 IC 에 비례한 가중치(음수는 0)

    python run_edt_suite3.py --spy <SPY 분봉 폴더> --only P1 --worker B --commit
    python run_edt_suite3.py --spy <SPY 분봉 폴더> --ensemble --commit
"""
import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import run_edt_suite2 as s2
import run_suite as base
from edt_study import ric, ridge
from run_edt_suite import EDT, metrics

logger = logging.getLogger("edt_suite3")


def day_rank_target(d):
    """날짜별 순위를 (순위+0.5)/n → 표준정규 분위수로 바꾼다. 그날 이벤트가 1건이면 0."""
    out = np.zeros(len(d.yz))
    for day in np.unique(d.D):
        idx = np.where(d.D == day)[0]
        if len(idx) < 2:
            continue
        r = np.argsort(np.argsort(d.yz[idx]))
        u = (r + 0.5) / len(idx)
        out[idx] = torch.erfinv(torch.tensor(2 * u - 1)).numpy() * np.sqrt(2)
    return out


def run_nn(d, spec, H, ys, save):
    dd = d
    if spec.get("target") == "rank":
        dd = copy.copy(d)
        dd.yz = day_rank_target(d)
    _, day_codes = np.unique(d.D, return_inverse=True)
    out, store = [], {}
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        runs = [s2.train(dd, spec, H, ys, tr, va, s, spec.get("epochs", 12), day_codes) for s in spec.get("seeds", [0, 1, 2])]
        curve = np.mean([[ric(p, d.y[va]) for p in r] for r in runs], 0)
        best = int(np.argmax(curve))
        pv = np.mean([r[best] for r in runs], 0)
        pt = np.mean([s2.train(dd, spec, H, ys, tr2, te, s, best + 1, day_codes)[best]
                      for s in spec.get("seeds", [0, 1, 2])], 0)
        m = metrics(d, te, pt)
        m.update(valid_rank_ic=ric(pv, d.y[va]), epochs=best + 1)
        logger.info("    구간 %d: 검증 %+.4f | 평가 %+.4f 집단내 %+.4f 롱숏 %+.1fbp t=%+.2f (에폭 %d)", k + 1,
                    m["valid_rank_ic"], m["rank_ic"], m["group_ric"], m["ls_bp"], m["ls_t"], best + 1)
        out.append(m)
        store[f"valid{k}"], store[f"test{k}"] = pv, pt
    np.savez_compressed(save, **store)
    return out


def run_ridge(d, spec, H, ys, save):
    out, store = [], {}
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        xv = np.hstack([d.ctx, d.content, d.inter, d.pca(tr, 32)])
        xt = np.hstack([d.ctx, d.content, d.inter, d.pca(tr2, 32)])
        lam = max((10, 100, 1e3, 1e4, 1e5), key=lambda l: ric(ridge(xv[tr], d.yz[tr], xv[va], l), d.y[va]))
        pv, pt = ridge(xv[tr], d.yz[tr], xv[va], lam), ridge(xt[tr2], d.yz[tr2], xt[te], lam)
        m = metrics(d, te, pt)
        m.update(valid_rank_ic=ric(pv, d.y[va]), lam=lam)
        out.append(m)
        store[f"valid{k}"], store[f"test{k}"] = pv, pt
    np.savez_compressed(save, **store)
    return out


def run_gbm(d, spec, H, ys, save):
    """LightGBM (문맥 + LLM + 상호작용 + 임베딩 PCA). 반복 수는 검증으로 고른다.

    spec: k(PCA 차원, 기본 32), target("rank" 면 일별 순위 목표값), ind(창 마지막 날 지표 9종 추가),
          seeds(여러 시드 평균), leaves(잎 수)
    """
    import lightgbm as lgb
    yt = day_rank_target(d) if spec.get("target") == "rank" else d.yz
    out, store = [], {}
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        def X(mask):
            cols = [d.ctx, d.content, d.inter, d.pca(mask, spec.get("k", 32))]
            if spec.get("ind"):
                cols.append(d.windows(mask)[:, -1, :])
            return np.hstack(cols)
        xv, xt = X(tr), X(tr2)
        pvs, pts, bests = [], [], []
        for seed in spec.get("seeds", [0]):
            params = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.03, "num_leaves": spec.get("leaves", 15),
                      "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "lambda_l2": 10.0, "verbose": -1, "seed": seed, "num_threads": torch.get_num_threads()}
            bst = lgb.train(params, lgb.Dataset(xv[tr], yt[tr]), num_boost_round=600)
            best = max(range(50, 601, 50), key=lambda r: ric(bst.predict(xv[va], num_iteration=r), d.y[va]))
            pvs.append(bst.predict(xv[va], num_iteration=best))
            pts.append(lgb.train(params, lgb.Dataset(xt[tr2], yt[tr2]), num_boost_round=best).predict(xt[te]))
            bests.append(best)
        pv, pt = np.mean(pvs, 0), np.mean(pts, 0)
        m = metrics(d, te, pt)
        m.update(valid_rank_ic=ric(pv, d.y[va]), rounds=bests)
        out.append(m)
        store[f"valid{k}"], store[f"test{k}"] = pv, pt
    np.savez_compressed(save, **store)
    return out


SMALL = s2.SMALL
EXPERIMENTS = [
    ("R  Ridge 기준선", run_ridge, {}),
    ("P1 개선 구조 (논문 구조 + 입력 확장)", run_nn, dict(SMALL)),
    ("V3 P1 + 회사 뉴스 이력 어텐션", run_nn, {**SMALL, "hist": True}),
    ("V4 P1 + 임베딩 PCA256", run_nn, {**SMALL, "k": 256}),
    ("V5 P1 + 일별 순위 목표값", run_nn, {**SMALL, "target": "rank"}),
    ("G  LightGBM (문맥 + LLM + 임베딩 PCA32)", run_gbm, {}),
    ("V6 V4(PCA256) + 일별 순위 목표값", run_nn, {**SMALL, "k": 256, "target": "rank"}),
    ("V7 V3(뉴스 이력) + 일별 순위 목표값", run_nn, {**SMALL, "hist": True, "target": "rank"}),
    ("P0 논문 구조 그대로 (d=512)", run_nn,
     {"raw_emb": True, "d": 512, "lr": 1e-4, "wd": 1e-5, "bs": 64, "rank": 0.0, "epochs": 8, "seeds": [0]}),
    ("G2 LightGBM + 일별 순위 목표값", run_gbm, {"target": "rank"}),
    ("G3 LightGBM + 임베딩 PCA128 + 최근 지표", run_gbm, {"k": 128, "ind": True}),
    ("G4 LightGBM 시드 5개 평균 (잎 31)", run_gbm, {"seeds": [0, 1, 2, 3, 4], "leaves": 31}),
]


def code(name):
    return name.split()[0]


def ensemble(d, out, members, tag):
    """예측을 구간 안에서 순위(0~1)로 바꿔 평균한다."""
    preds = {m: np.load(out / "preds" / f"{m}.npz") for m in members}
    rk = lambda v: np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    res_eq, res_val = [], []
    for k, (tr, va, tr2, te) in enumerate(s2.fold_masks(d)):
        vic = {m: ric(preds[m][f"valid{k}"], d.y[va]) for m in members}
        w = np.array([max(vic[m], 0) for m in members])
        w = w / w.sum() if w.sum() > 0 else np.ones(len(members)) / len(members)
        for weights, res in ((np.ones(len(members)) / len(members), res_eq), (w, res_val)):
            pv = sum(wi * rk(preds[m][f"valid{k}"]) for wi, m in zip(weights, members))
            pt = sum(wi * rk(preds[m][f"test{k}"]) for wi, m in zip(weights, members))
            mm = metrics(d, te, pt)
            mm.update(valid_rank_ic=ric(pv, d.y[va]))
            res.append(mm)
    return {f"{tag} 같은 가중치 ({'+'.join(members)})": res_eq, f"{tag} 검증 비례 가중치 ({'+'.join(members)})": res_val}


def write_report(out):
    res = {}
    for p in sorted(out.glob("results_*.json")):
        res.update(json.loads(p.read_text()))
    L = ["# EDT 3라운드: 앙상블과 순위 목표값 (평가 구간 3개)", "",
         "평가 구간: 2020-11~12 / 2021-01~02 / 2021-03~05. 정렬 기준은 **검증** 평균 순위 IC(평가 성적으로 고르지 않는다).",
         "핵심 이벤트, 1일 초과수익률. '집단 내'는 발행 시점 × 주가 수준 9개 집단 안의 순위 IC 평균.", "",
         "| 모델 | 검증 평균 | 평가 평균 순위IC | 구간별 | 평가 평균 집단 내 | 구간별 | 롱숏 bp | 구간별 t |",
         "|---|---:|---:|---|---:|---|---:|---|"]
    for name, ms in sorted(res.items(), key=lambda kv: -np.mean([m["valid_rank_ic"] for m in kv[1]])):
        col = lambda key, fmt: " ".join(format(m[key], fmt) for m in ms)
        L.append("| %s | %+.4f | %+.4f | %s | %+.4f | %s | %+.1f | %s |" % (
            name, np.mean([m["valid_rank_ic"] for m in ms]), np.mean([m["rank_ic"] for m in ms]), col("rank_ic", "+.3f"),
            np.mean([m["group_ric"] for m in ms]), col("group_ric", "+.3f"), np.mean([m["ls_bp"] for m in ms]),
            col("ls_t", "+.2f")))
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy", required=True)
    ap.add_argument("--out", default="experiments/edt_suite3")
    ap.add_argument("--only", default="")
    ap.add_argument("--worker", default="main")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"suite_{a.worker}.log", encoding="utf-8")])
    torch.set_num_threads(a.threads)
    d = EDT(a.spy)
    path = out / f"results_{a.worker}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if a.ensemble:
        have = sorted(p.stem for p in (out / "preds").glob("*.npz"))
        logger.info("앙상블 대상: %s", have)
        r3 = ["P1", "R", "V3", "V4", "V5"]
        r4 = ["G", "P0", "P1", "R", "V3", "V4", "V5", "V6", "V7"]
        for members, tag in ((["P1", "V4"], "E1"), (["P1", "V3", "V4", "R"], "E2"), (r3, "E3"),
                             (r4, "E4"), ([m for m in r4 if m != "P0"], "E5"), (have, "E6")):
            members = [m for m in members if m in have]
            if len(members) >= 2:
                res.update(ensemble(d, out, members, tag))
    else:
        ys, H = s2.extra_targets(d)
        only = [x.strip() for x in a.only.split(",") if x.strip()]
        for name, fn, spec in EXPERIMENTS:
            if (only and code(name) not in only) or name in res:
                continue
            logger.info("▶ %s", name)
            t0 = time.time()
            res[name] = fn(d, spec, H, ys, out / "preds" / f"{code(name)}.npz")
            logger.info("  (%.0fs)", time.time() - t0)
            path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            write_report(out)
            if a.commit:
                base.commit(out, f"[edt_suite3] {name}")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out)
    if a.commit:
        base.commit(out, "[edt_suite3] 앙상블" if a.ensemble else "[edt_suite3] 갱신")


if __name__ == "__main__":
    main()
