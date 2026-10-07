"""이벤트 유형별 발행 이후 초과수익률 (이벤트 연구).

Trade the Event(Zhou et al. 2021)는 특정 기업 이벤트가 탐지되면 발행 시점에
거래해 초과수익을 얻었다. 여기서는 LLM 이 뽑은 이벤트 유형·방향별로 발행 이후
초과수익률(y_abn_z: 시장 조정, 20일 변동성 단위)의 평균이 학습 구간과 평가 구간에서
같은 부호로 나타나는지 본다.

    python analysis/event_study.py --out experiments/diagnosis/EVENT_STUDY.md
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

TEST_START = "2023-07-01"


def load(features: str, events: str):
    with gzip.open(events, "rt", encoding="utf-8") as f:
        ev = {r["news_id"]: r for r in map(json.loads, f)}
    rows = []
    with gzip.open(features, "rt", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["news_id"] in ev:
                rows.append({**r, **ev[r["news_id"]]})
    return rows


def tstat(x):
    x = np.asarray(x)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


def date_clustered(rows, key):
    """같은 날 같은 종목 이벤트를 평균해 하나로 묶은 뒤의 값 목록."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["ticker"], r["date"], r["kind"])].append(key(r))
    return [np.mean(v) for v in g.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/fnspid/event_features.jsonl.gz")
    ap.add_argument("--events", default="data/fnspid/events_llm.jsonl.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.features, a.events)
    tr = [r for r in rows if r["date"] < TEST_START]
    te = [r for r in rows if r["date"] >= TEST_START]
    L = ["# 이벤트 유형별 발행 이후 초과수익률", "",
         f"대상: 이벤트 특징과 LLM 이벤트가 모두 있는 {len(rows):,}건 (학습 구간 {len(tr):,} / 평가 구간 {len(te):,}).",
         "값은 y_abn_z(시장 조정 초과수익률 / 20일 변동성)의 평균. '방향 반영'은 LLM 이 매긴 방향(−1/0/+1)을 곱한 값이다.",
         "t 는 같은 종목·날짜·유형 이벤트를 하나로 묶은 뒤 계산했다.", ""]

    # 1. 분포
    L += ["## 1. 기사 성격 분포", "", "| 항목 | 비율 |", "|---|---:|"]
    for k, name in (("about_company", "대상 종목이 주인공"), ("new_info", "새로운 기업 사건 보도"),
                    ("price_recap", "주가 움직임 재보도")):
        L.append(f"| {name} | {np.mean([r[k] for r in rows]):.1%} |")
    core = [r for r in rows if r["about_company"] and r["new_info"]]
    L.append(f"| 주인공 + 새 정보 (핵심 이벤트) | {len(core) / len(rows):.1%} |")

    # 2. 방향 신호의 예측력 (집단별)
    L += ["", "## 2. LLM 방향 신호의 발행 이후 예측력", "",
          "| 집단 | 학습 n | 학습 방향 반영 평균 | t | 평가 n | 평가 방향 반영 평균 | t |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    groups = [
        ("전체", lambda r: True),
        ("방향 ≠ 0", lambda r: r["direction"] != 0),
        ("주인공 + 새 정보 + 방향 ≠ 0", lambda r: r["about_company"] and r["new_info"] and r["direction"] != 0),
        ("  └ 주가 재보도 아님", lambda r: r["about_company"] and r["new_info"] and r["direction"] != 0 and not r["price_recap"]),
        ("  └ 중요도 2 이상", lambda r: r["about_company"] and r["new_info"] and r["direction"] != 0 and r["materiality"] >= 2),
        ("  └ 장 시작 전 발행", lambda r: r["about_company"] and r["new_info"] and r["direction"] != 0 and r["kind"] == "pre"),
        ("  └ 장중 발행", lambda r: r["about_company"] and r["new_info"] and r["direction"] != 0 and r["kind"] == "intra"),
        ("  └ 장 마감 후 발행", lambda r: r["about_company"] and r["new_info"] and r["direction"] != 0 and r["kind"] == "after"),
    ]
    for name, cond in groups:
        cells = []
        for part in (tr, te):
            sel = [r for r in part if cond(r)]
            v = date_clustered(sel, lambda r: r["direction"] * r["y_abn_z"]) if sel else []
            cells += [f"{len(sel):,}", f"{np.mean(v):+.3f}" if v else "-", f"{tstat(v):+.2f}" if len(v) > 2 else "-"]
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # 3. 유형 × 방향
    L += ["", "## 3. 이벤트 유형별 (주인공 + 새 정보)", "",
          "| 유형 | 방향 | 학습 n | 학습 평균 | t | 평가 n | 평가 평균 | t | 같은 부호 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|:-:|"]
    combos = collections.Counter((r["event_type"], r["direction"]) for r in core)
    for (et, d), n in combos.most_common():
        if n < 40:
            continue
        cells, means = [], []
        for part in (tr, te):
            sel = [r for r in part if r["about_company"] and r["new_info"] and r["event_type"] == et and r["direction"] == d]
            v = date_clustered(sel, lambda r: r["y_abn_z"]) if sel else []
            means.append(np.mean(v) if v else np.nan)
            cells += [f"{len(sel):,}", f"{np.mean(v):+.3f}" if v else "-", f"{tstat(v):+.2f}" if len(v) > 2 else "-"]
        same = "○" if np.isfinite(means).all() and np.sign(means[0]) == np.sign(means[1]) else "×"
        L.append(f"| {et} | {d:+d} | " + " | ".join(cells) + f" | {same} |")

    # 4. 유형 분포 전체
    L += ["", "## 4. 이벤트 유형 분포 (전체)", "", "| 유형 | 건수 | 비율 |", "|---|---:|---:|"]
    for et, n in collections.Counter(r["event_type"] for r in rows).most_common():
        L.append(f"| {et} | {n:,} | {n / len(rows):.1%} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
