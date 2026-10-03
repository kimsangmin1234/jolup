"""학습 결과가 무작위 수준인 원인을 데이터 쪽에서 진단한다.

신경망을 거치지 않고, 아래 질문에 직접 답한다.

1. 감성 점수 하나만으로 등락률과 상관이 있는가?
   (있는데 모델이 못 잡으면 모델 문제, 없으면 데이터 문제)
2. 장중(당일 라벨) / 마감 후(다음날 라벨) / 시각 미확인 중 어디에 신호가 있는가?
3. 같은 종목·같은 날 뉴스를 묶으면 신호가 선명해지는가?
4. 기사가 실제로 그 종목을 다루는지(티커 태깅 정확도)에 따라 달라지는가?
5. 임베딩 전체를 선형 회귀(Ridge)에 넣으면 신경망보다 나은가?

사용법::

    python analysis/diagnose_signal.py \\
        --cache "experiments/fnspid-server/news_cache_timed.part*.jsonl.gz" \\
        --out experiments/diagnosis/REPORT.md
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np

TRAIN_END, VALID_END = "2022-12-31", "2023-06-30"

# 기사 본문(요약)이 실제로 해당 종목을 다루는지 판단할 이름
COMPANY = {
    "AAPL": r"apple|애플", "AMD": r"\bamd\b|advanced micro", "AMZN": r"amazon|아마존",
    "BA": r"boeing|보잉", "BLNK": r"blink", "CVX": r"chevron|쉐브론", "DIS": r"disney|디즈니",
    "F": r"\bford\b|포드", "FCEL": r"fuelcell|fuel cell", "GE": r"general electric|\bge\b",
    "GM": r"general motors|\bgm\b", "GME": r"gamestop|게임스탑", "INTC": r"intel|인텔",
    "KO": r"coca-cola|coca cola|\bcoke\b|코카콜라", "MRK": r"merck|머크",
    "MSFT": r"microsoft|마이크로소프트", "MU": r"micron|마이크론", "NKLA": r"nikola|니콜라",
    "NVDA": r"nvidia|엔비디아", "TSLA": r"tesla|테슬라", "WMT": r"walmart|월마트",
    "AMC": r"\bamc\b", "JPM": r"jpmorgan|jp morgan",
}
# 알고리즘으로 찍어내는 옵션·배당 안내 기사의 흔적
FILLER = re.compile(r"put contract|call contract|yieldboost|options? (?:chain|trading)|"
                    r"strike price|ex-dividend|dividend (?:payment|date)", re.I)


def load(pattern: str):
    rows, emb = [], []
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                e = r.pop("embedding", None)
                if not e or r.get("sentiment") is None:
                    continue
                pub = r.get("published_et")
                r["group"] = ("장중→당일" if r.get("horizon") == "same_day"
                              else "마감후→다음날" if pub else "시각미확인→다음날")
                r["split"] = ("train" if r["date"] <= TRAIN_END
                              else "valid" if r["date"] <= VALID_END else "test")
                s = r.get("summary", "")
                r["relevant"] = bool(re.search(COMPANY.get(r["ticker"], r"$^"), s, re.I)) \
                    or bool(re.search(rf"\b{re.escape(r['ticker'])}\b", s))
                r["filler"] = bool(FILLER.search(s))
                rows.append(r)
                emb.append(np.asarray(e, dtype=np.float32))
    return rows, np.vstack(emb)


def stats(sent: np.ndarray, label: np.ndarray) -> dict:
    """감성과 등락률의 관계를 요약한다."""
    n = len(label)
    if n < 30:
        return {"n": n}
    pear = float(np.corrcoef(sent, label)[0, 1]) if sent.std() > 0 else float("nan")
    rk = lambda a: np.argsort(np.argsort(a))
    spear = float(np.corrcoef(rk(sent), rk(label))[0, 1]) if sent.std() > 0 else float("nan")
    nz = sent != 0
    hit = float((np.sign(sent[nz]) == np.sign(label[nz])).mean()) if nz.any() else float("nan")
    strong = np.abs(sent) >= 0.5
    hit_s = float((np.sign(sent[strong]) == np.sign(label[strong])).mean()) if strong.sum() > 30 else float("nan")
    # 상관계수의 표준오차 근사: 1/sqrt(n). 2배를 넘으면 우연이 아닐 가능성이 높다.
    return {"n": n, "pearson": pear, "spearman": spear, "hit": hit,
            "hit_strong": hit_s, "n_strong": int(strong.sum()), "z": pear * np.sqrt(n)}


def fmt(name: str, s: dict) -> str:
    if s["n"] < 30:
        return f"| {name} | {s['n']:,} | – | – | – | – | – |"
    flag = " ✅" if abs(s["z"]) >= 2 else ""
    return (f"| {name} | {s['n']:,} | {s['pearson']:+.4f}{flag} | {s['spearman']:+.4f} | "
            f"{s['hit']:.3f} | {s['hit_strong']:.3f} | {s['z']:+.1f} |")


def ridge(X, y, split, lambdas=(1, 10, 100, 1e3, 1e4, 1e5)):
    """임베딩 → 등락률 선형 회귀. 검증 구간 MSE로 규제 강도를 고른다."""
    tr, va, te = (split == "train"), (split == "valid"), (split == "test")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = (X - mu) / sd
    ym = y[tr].mean()
    A, b = Z[tr].T @ Z[tr], Z[tr].T @ (y[tr] - ym)
    best = None
    for lam in lambdas:
        w = np.linalg.solve(A + lam * np.eye(A.shape[0]), b)
        mse = float(((Z[va] @ w + ym - y[va]) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, w)
    _, lam, w = best
    p, t = Z[te] @ w + ym, y[te]
    return {"lambda": lam, "corr": float(np.corrcoef(p, t)[0, 1]),
            "dir_acc": float((np.sign(p) == np.sign(t)).mean()),
            "mse_vs_zero": float(((p - t) ** 2).mean() / (t ** 2).mean()), "n_test": int(te.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, E = load(args.cache)
    S = np.array([r["sentiment"] for r in rows], dtype=np.float64)
    Y = np.array([r["label"] for r in rows], dtype=np.float64)
    G = np.array([r["group"] for r in rows])
    SP = np.array([r["split"] for r in rows])
    REL = np.array([r["relevant"] for r in rows])
    FIL = np.array([r["filler"] for r in rows])
    T = np.array([r["ticker"] for r in rows])
    out = ["# 신호 진단 보고서", "",
           f"대상: `{args.cache}` — {len(rows):,}건", "",
           "상관계수 옆 ✅ 는 |z| ≥ 2, 즉 우연으로 보기 어려운 수준이다 (z ≈ r·√n).",
           "방향 일치율은 감성 부호와 등락률 부호가 같은 비율이며 50%가 무작위 기준이다.", ""]
    head = ["| 구분 | 건수 | Pearson | Spearman | 방향 일치 | 강한 감성(|s|≥0.5) 일치 | z |",
            "|---|---:|---:|---:|---:|---:|---:|"]

    out += ["## 1. 감성 점수 하나만으로 본 신호", ""] + head
    out.append(fmt("전체", stats(S, Y)))
    for g in ("장중→당일", "마감후→다음날", "시각미확인→다음날"):
        m = G == g
        out.append(fmt(g, stats(S[m], Y[m])))
    out.append("")

    out += ["## 2. 기간별 (학습·검증·평가 구간이 같은 경향인가)", ""] + head
    for sp in ("train", "valid", "test"):
        for g in ("장중→당일", "마감후→다음날"):
            m = (SP == sp) & (G == g)
            out.append(fmt(f"{sp} · {g}", stats(S[m], Y[m])))
    out.append("")

    out += ["## 3. 같은 종목·같은 날 뉴스를 묶은 경우", "",
            "같은 날 여러 기사가 같은 라벨을 공유하므로, 기사 단위로 세면 표본이 부풀려진다.",
            "종목·날짜·예측 대상별로 감성을 평균 내 다시 본다.", ""] + head
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["ticker"], r["date"], r["horizon"])].append(r)
    for g in ("장중→당일", "마감후→다음날", "시각미확인→다음날"):
        ks = [k for k, v in agg.items() if v[0]["group"] == g]
        s = np.array([np.mean([x["sentiment"] for x in agg[k]]) for k in ks])
        y = np.array([agg[k][0]["label"] for k in ks])
        out.append(fmt(f"{g} (일 단위)", stats(s, y)))
    out.append(f"\n기사 {len(rows):,}건 → 종목·일 단위 {len(agg):,}건 "
               f"(평균 {len(rows)/len(agg):.1f}건이 같은 라벨을 공유)\n")

    out += ["## 4. 기사가 실제로 그 종목을 다루는가", "",
            f"요약문에 회사명·티커가 나오는 기사: {REL.mean()*100:.1f}%  ·  "
            f"옵션/배당 안내형 기사: {FIL.mean()*100:.1f}%", ""] + head
    for name, m in (("회사명 언급 O", REL), ("회사명 언급 X", ~REL),
                    ("언급 O · 장중→당일", REL & (G == "장중→당일")),
                    ("옵션/배당 안내형 제외", ~FIL)):
        out.append(fmt(name, stats(S[m], Y[m])))
    out.append("")

    out += ["## 5. 종목별 (장중→당일)", "", "| 종목 | 건수 | Pearson | 방향 일치 | z |", "|---|---:|---:|---:|---:|"]
    for t in sorted(set(T)):
        m = (T == t) & (G == "장중→당일")
        s = stats(S[m], Y[m])
        if s["n"] >= 30:
            out.append(f"| {t} | {s['n']:,} | {s['pearson']:+.3f}{' ✅' if abs(s['z'])>=2 else ''} | {s['hit']:.3f} | {s['z']:+.1f} |")
    out.append("")

    out += ["## 6. 임베딩 전체를 선형 회귀(Ridge)에 넣은 경우", "",
            "신경망 대신 가장 단순한 선형 모델로, 같은 입력에 신호가 있는지 본다.",
            "신경망보다 나으면 모델·학습 쪽 문제, 비슷하게 무작위면 데이터 쪽 문제다.", "",
            "| 입력 · 대상 | 평가 건수 | λ | 상관 | 방향 적중 | 0예측 대비 MSE |", "|---|---:|---:|---:|---:|---:|"]
    X = np.hstack([E.astype(np.float64), S[:, None]])
    for name, m in (("임베딩+감성 · 전체", np.ones(len(rows), bool)),
                    ("임베딩+감성 · 장중→당일", G == "장중→당일"),
                    ("감성만 · 장중→당일", G == "장중→당일")):
        Xm = X[m] if "임베딩" in name else S[m][:, None]
        r = ridge(Xm, Y[m], SP[m])
        out.append(f"| {name} | {r['n_test']:,} | {r['lambda']:g} | {r['corr']:+.4f} | "
                   f"{r['dir_acc']:.3f} | {r['mse_vs_zero']:.3f} |")
    out += ["", "참고: 신경망(`fnspid-timed-raw`) 평가 결과는 상관 +0.014, 방향 0.519, 0예측 대비 1.042 였다.", ""]

    text = "\n".join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
