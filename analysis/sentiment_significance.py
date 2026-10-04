"""감성 점수의 예측력이 우연인지, 사후 보도 효과인지 검증한다.

감성 점수만 쓰는 모델은 학습할 것이 사실상 없다(부호가 '긍정이면 상승'으로
고정된 단조 함수라 IC 는 감성과 등락률의 상관과 같다). 따라서 전체 기간을
과적합 걱정 없이 평가에 쓸 수 있다.

1) 우연 여부: 같은 날 기사끼리 묶어 날짜 단위 블록 부트스트랩으로 신뢰구간을 낸다.
   같은 날 종목들이 함께 움직이고, 같은 종목·날짜 기사가 라벨을 공유하므로
   기사 단위 표준오차는 불확실성을 과소평가한다.
2) 일관성: 분기별·종목별로 부호가 얼마나 일관되는지 본다.
3) 사후 보도: 발행 시각 이후의 수익률만 쓰는 깨끗한 구간(장 마감 후 뉴스 → 다음날)과
   발행 전 움직임이 섞인 구간(장중 뉴스 → 당일 종가 대비)을 비교하고,
   요약문에 주가 움직임 표현이 있는 기사를 따로 본다.

    python analysis/sentiment_significance.py --out experiments/diagnosis/SENTIMENT_SIGNIFICANCE.md
"""
from __future__ import annotations

import argparse, glob, gzip, json, re
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_indicators

MOVE = re.compile(r"\b(shares?|stock|stocks)\b.{0,40}\b(rose|rise|rises|risen|jump\w*|surg\w*|soar\w*|rall\w*|gain\w*|climb\w*|fell|fall\w*|drop\w*|slid|slump\w*|plung\w*|tumbl\w*|sank|sink\w*|declin\w*|down|up)\b|\d+(\.\d+)?\s?%",
                  re.IGNORECASE)


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        for line in gzip.open(p, "rt", encoding="utf-8"):
            r = json.loads(line); r.pop("embedding", None); rows.append(r)
    return rows


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def block_boot(s, y, d, n=2000, seed=0):
    """날짜 단위로 복원 추출해 IC 분포를 만든다."""
    rng = np.random.default_rng(seed)
    days, inv = np.unique(d, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(days))]
    ics = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))])
        ics.append(corr(s[idx], y[idx]))
    ics = np.array(ics)
    return np.percentile(ics, [2.5, 97.5]), float((ics <= 0).mean())


def row_stats(name, s, y, d):
    ic = corr(s, y)
    (lo, hi), p = block_boot(s, y, d)
    naive_t = ic * np.sqrt(len(s))
    hit = float(((s > 0) == (y > 0))[s != 0].mean())
    return (f"| {name} | {len(s):,} | {len(np.unique(d)):,} | {ic:+.4f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{p:.4f} | {naive_t:.1f} | {hit:.3f} |")


HEAD = ["| 구분 | 기사 | 거래일 | IC | 95% 신뢰구간(날짜 블록) | p(IC≤0) | 단순 t | 방향 일치(감성≠0) |",
        "|---|---:|---:|---:|---|---:|---:|---:|"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="experiments/fnspid-timed-norm/news_cache_timed.part*.jsonl.gz")
    ap.add_argument("--indicators", default="data/fnspid/indicators.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-dir", default="", help="FNSPID full_history (시가 필요, 선택)")
    a = ap.parse_args()

    rows = load(a.cache)
    ind, idx = load_indicators(a.indicators)
    S = np.array([float(r["sentiment"]) for r in rows]); Y = np.array([float(r["label"]) for r in rows])
    D = np.array([r["date"] for r in rows]); T = np.array([r["ticker"] for r in rows])
    hm = np.array([int(r["published_et"][11:13]) * 60 + int(r["published_et"][14:16]) for r in rows])
    move = np.array([bool(MOVE.search(r.get("summary", ""))) for r in rows])

    # 뉴스 날짜 t 기준 전일·당일·익일 종가 수익률
    def ret(t, d, k):
        i = idx[t].get(d); c = ind[t][:, 0]
        if i is None or i + k < 1 or i + k >= len(c): return np.nan
        return c[i + k] / c[i + k - 1] - 1
    Rm1 = np.array([ret(t, d, -1) for t, d in zip(T, D)])
    R0 = np.array([ret(t, d, 0) for t, d in zip(T, D)])
    R1 = np.array([ret(t, d, 1) for t, d in zip(T, D)])

    pre, intra, after = hm < 570, (hm >= 570) & (hm < 960), hm >= 960
    L = ["# 감성 점수 예측력: 우연인가, 사후 보도인가", "",
         f"데이터: 발행 시각 확인 뉴스 {len(rows):,}건(22종목, 2020-09 ~ 2023-12). 감성 점수는 학습 없이 그대로 쓴다.",
         "신뢰구간·p값은 거래일 단위 블록 부트스트랩(2,000회). '단순 t'는 기사를 독립으로 본 값(과대평가됨).", "",
         "## 1. 전체와 기간별", ""] + HEAD
    test = D >= "2023-07-01"
    L.append(row_stats("전체 (현행 라벨)", S, Y, D))
    L.append(row_stats("2023 하반기 평가 구간", S[test], Y[test], D[test]))
    for y0 in ("2020", "2021", "2022", "2023"):
        m = np.char.startswith(D, y0); L.append(row_stats(f"{y0}년", S[m], Y[m], D[m]))

    q = np.array([f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}" for d in D])
    qs = sorted(set(q)); qic = [corr(S[q == k], Y[q == k]) for k in qs]
    L += ["", f"분기별 IC ({sum(x > 0 for x in qic)}/{len(qs)} 분기 양수): " +
          ", ".join(f"{k} {v:+.3f}" for k, v in zip(qs, qic))]
    tic = {t: corr(S[T == t], Y[T == t]) for t in sorted(set(T))}
    L += ["", f"종목별 IC ({sum(v > 0 for v in tic.values())}/{len(tic)} 종목 양수): " +
          ", ".join(f"{t} {v:+.3f}" for t, v in sorted(tic.items(), key=lambda kv: -kv[1]))]

    L += ["", "## 2. 발행 시각과 수익률 구간 (사후 보도 점검)", "",
          "라벨 구간이 발행 시각 **이후**만 포함하면 진짜 예측, 발행 전 움직임을 포함하면 사후 보도가 섞인다.", ""] + HEAD
    ok = ~np.isnan(R1) & ~np.isnan(Rm1)
    for name, m, r in [
        ("장 시작 전(<9:30) → 당일 [전일 종가→당일 종가]", pre, R0),
        ("장중(9:30~16:00) → 당일 [발행 전 움직임 포함]", intra, R0),
        ("장 마감 후(≥16:00) → 당일 [이미 끝난 움직임]", after, R0),
        ("장 마감 후(≥16:00) → 다음날 [발행 이후만, 깨끗]", after, R1),
        ("장중 → 다음날 [발행 이후만, 깨끗]", intra, R1),
        ("장 시작 전 → 다음날", pre, R1),
        ("전체 → 전일 수익률 [과거, 예측 불가]", np.ones_like(pre), Rm1),
    ]:
        mm = m & ok; L.append(row_stats(name, S[mm], r[mm], D[mm]))

    L += ["", "## 3. 요약문에 주가 움직임 표현이 있는 기사", "",
          f"'shares rose/fell', 'stock jumped', 'N%' 같은 표현이 요약에 있는 기사: {move.mean():.1%}", ""] + HEAD
    for name, m in [("움직임 표현 있음 (현행 라벨)", move), ("움직임 표현 없음 (현행 라벨)", ~move),
                    ("움직임 없음 · 장중 → 당일", ~move & intra), ("움직임 없음 · 깨끗한 구간(마감 후→다음날, 장중→다음날)", ~move & (after | intra))]:
        r = R1 if "깨끗" in name else (R0 if "장중" in name else Y)
        mm = m & ~np.isnan(r); L.append(row_stats(name, S[mm], r[mm], D[mm]))

    if a.price_dir:
        import csv
        oc, gap = {}, {}
        for t in sorted(set(T)):
            path = next((q for q in Path(a.price_dir).glob("*.csv") if q.stem.upper() == t.replace("GOOGL", "GOOG")), None)
            if path is None:
                continue
            # 2020-07-06 이전은 소스 이어붙임 경계 때문에 쓰지 않는다(DATASET.md 3.2)
            bars = sorted((r["date"], float(r["open"]), float(r["close"]))
                          for r in csv.DictReader(open(path))
                          if r["date"] >= "2020-07-06" and r["open"] and r["close"])
            for (d0, _, c0), (d1, o1, c1) in zip(bars, bars[1:]):
                if min(o1, c0, c1) <= 0:
                    continue
                oc[(t, d1)] = c1 / o1 - 1          # 시가 → 종가
                gap[(t, d1)] = o1 / c0 - 1         # 전일 종가 → 시가
        OC = np.array([oc.get((t, d), np.nan) for t, d in zip(T, D)])
        GP = np.array([gap.get((t, d), np.nan) for t, d in zip(T, D)])
        L += ["", "## 4. 시가를 이용한 분해", "",
              "전일 종가→당일 종가 = [전일 종가→시가 갭] + [시가→종가]. 장 시작 전 뉴스에게 시가→종가는",
              "발행 이후 구간이다. 장중 뉴스에게 갭은 발행 전 구간이다.", ""] + HEAD
        for name, m, r in [("장 시작 전 → 갭 [대부분 발행 전후 혼재]", pre, GP),
                           ("장 시작 전 → 시가→종가 [발행 이후, 깨끗]", pre, OC),
                           ("장중 → 갭 [발행 전, 예측 불가]", intra, GP),
                           ("장중 → 시가→종가 [발행 전후 혼재]", intra, OC)]:
            mm = m & ~np.isnan(r); L.append(row_stats(name, S[mm], r[mm], D[mm]))

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
