"""GPT-4o-mini 로 뉴스에서 구조화된 기업 이벤트를 뽑는다 (Batch API).

선행 연구
    - Ding et al. (2014, EMNLP; 2015, IJCAI): 뉴스에서 구조화된 이벤트를 뽑아 쓰면
      단어 묶음·감성보다 예측이 낫다.
    - Zhou, Ma, Liu (2021, Findings of ACL) "Trade the Event": 기업 이벤트(인수, 임상,
      가이던스, 신규 계약, 자사주, 분할, 배당 등 11종)를 탐지해 발행 시점에 거래한다.
    - Chen et al. (2019) "Incorporating Fine-grained Events in Stock Movement Prediction".

여기서는 LLM 에 기사와 대상 종목을 주고 다음을 JSON 으로 받는다.
    event_type     이벤트 유형(아래 EVENT_TYPES)
    about_company  대상 종목이 기사의 주인공인가 (1/0)
    new_info       하루 안에 공개된 새로운 기업 사건을 보도하는가 (1/0)
    price_recap    이미 일어난 주가 움직임을 주로 다루는가 (1/0)
    direction      이미 반영된 것 이상으로 주가에 미칠 방향 (-1/0/1)
    materiality    중요도·의외성 (0~3)

발행 시각이 확인된 기사(data/fnspid/labels_open_close.jsonl.gz)만 대상이다.
본문은 앞 1,500자만 넣는다(이벤트 유형은 제목·첫 문단에 담긴다).

    export OPENAI_API_KEY=...
    python extract_events.py --commit
결과: data/fnspid/events_llm.jsonl.gz
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

COMPANY = {
    "AAPL": "Apple", "AMC": "AMC Entertainment", "AMD": "Advanced Micro Devices", "AMZN": "Amazon",
    "BA": "Boeing", "BLNK": "Blink Charging", "CVX": "Chevron", "DIS": "Walt Disney", "F": "Ford Motor",
    "FCEL": "FuelCell Energy", "GE": "General Electric", "GM": "General Motors", "GME": "GameStop",
    "INTC": "Intel", "KO": "Coca-Cola", "MRK": "Merck", "MSFT": "Microsoft", "MU": "Micron Technology",
    "NKLA": "Nikola", "NVDA": "Nvidia", "TSLA": "Tesla", "WMT": "Walmart",
}

EVENT_TYPES = [
    "earnings_beat", "earnings_miss", "earnings_inline", "guidance_raise", "guidance_cut",
    "analyst_upgrade", "analyst_downgrade", "price_target_change", "acquisition_merger",
    "new_contract_partnership", "product_launch", "regulatory_approval_clinical",
    "regulatory_legal_risk", "lawsuit_settlement", "management_change", "dividend_increase",
    "dividend_cut", "share_buyback", "stock_split", "equity_offering", "insider_institutional_trade",
    "layoffs_restructuring", "operational_issue", "market_commentary", "stock_picks_opinion",
    "price_move_report", "other",
]

PROMPT = """You are an equity analyst. Classify the news article below with respect to {name} (ticker {ticker}).
Reply with JSON only:
{{"event_type": "...", "about_company": 0, "new_info": 0, "price_recap": 0, "direction": 0, "materiality": 0}}

- event_type: exactly one of {types}
- about_company: 1 if {ticker} is the main subject of the article; 0 if it is only mentioned in passing or in a list.
- new_info: 1 if the article reports a concrete company event or announcement made public within about one day
  (earnings release, deal, approval, guidance, rating change, lawsuit, executive change, ...);
  0 for opinion pieces, stock picks, previews, recaps of older news, or general market commentary.
- price_recap: 1 if the article mainly describes how the stock price has already moved.
- direction: the likely effect of this news on {ticker}'s stock price beyond what has already happened:
  1 positive, -1 negative, 0 none or unclear.
- materiality: 0 none, 1 minor, 2 notable, 3 major and unexpected.

[ARTICLE]
{text}
"""

MAX_CHARS = 1500
MAX_ENQUEUED_TOKENS = 1_500_000


def estimate_tokens(req: dict) -> int:
    return int(len(req["body"]["messages"][0]["content"]) / 3.5) + 150


def request(nid: str, ticker: str, text: str) -> dict:
    content = PROMPT.format(name=COMPANY.get(ticker, ticker), ticker=ticker,
                            types=", ".join(EVENT_TYPES), text=text[:MAX_CHARS])
    return {"custom_id": nid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 120,
                     "response_format": {"type": "json_object"},
                     "messages": [{"role": "user", "content": content}]}}


def parse(body: dict) -> dict | None:
    try:
        p = json.loads(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None

    def num(key, lo, hi):
        try:
            return max(lo, min(hi, int(round(float(p.get(key, 0))))))
        except (TypeError, ValueError):
            return 0
    et = str(p.get("event_type", "other")).strip()
    return {"event_type": et if et in EVENT_TYPES else "other",
            "about_company": num("about_company", 0, 1), "new_info": num("new_info", 0, 1),
            "price_recap": num("price_recap", 0, 1), "direction": num("direction", -1, 1),
            "materiality": num("materiality", 0, 3)}


def git_commit(path: str, message: str) -> None:
    subprocess.run(["git", "add", "-f", path], check=False)
    msg = (f"{message}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK")
    subprocess.run(["git", "-c", "user.email=sunshine31885@gmail.com", "-c", "user.name=Claude",
                    "commit", "-q", "-m", msg], check=False)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    for delay in (2, 4, 8, 16):
        subprocess.run(["git", "pull", "-q", "--no-rebase", "origin", branch], check=False)
        if subprocess.run(["git", "push", "-q", "origin", branch]).returncode == 0:
            return
        time.sleep(delay)


def run_realtime(api, pending, texts, done, save, args) -> None:
    """일반 API 병렬 호출. 요청 한도(429)에 걸리면 지수 백오프로 기다린다."""
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()

    def one(nid):
        body = request(nid, *texts[nid])["body"]
        delay = 2.0
        for _ in range(8):
            try:
                resp = api.chat.completions.create(**body)
                return nid, parse({"choices": [{"message": {"content": resp.choices[0].message.content}}]})
            except Exception as exc:                      # 한도·연결 오류는 재시도
                if not any(k in type(exc).__name__ for k in ("RateLimit", "Connection", "Timeout", "InternalServer")):
                    logger.warning("  %s 실패: %s", nid, type(exc).__name__)
                    return nid, None
                time.sleep(delay * (1 + random.uniform(-0.3, 0.3)))
                delay = min(delay * 2, 60)
        return nid, None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, n) for n in pending]
        for k, fut in enumerate(as_completed(futures), 1):
            nid, parsed = fut.result()
            if parsed:
                with lock:
                    done[nid] = {"news_id": nid, **parsed}
            if k % 1000 == 0 or k == len(futures):
                save()
                logger.info("  %d/%d (누적 %d건, %.0f건/분)", k, len(futures), len(done), k / (time.time() - t0) * 60)
            if args.commit and k % 10000 == 0:
                git_commit(args.out, f"이벤트 추출 진행: {len(done)}건")
    save()
    if args.commit:
        git_commit(args.out, f"이벤트 추출 완료: {len(done)}건")
    logger.info("전량 완료: %d건", len(done))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/fnspid/labels_open_close.jsonl.gz")
    ap.add_argument("--articles", default="data/fnspid/articles/records_article.part*.jsonl.gz")
    ap.add_argument("--out", default="data/fnspid/events_llm.jsonl.gz")
    ap.add_argument("--work", default=".work/events")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--mode", choices=("batch", "realtime"), default="batch",
                    help="realtime: 일반 API 를 병렬 호출(빠르지만 비용 2배). 한도가 넉넉한 계정용")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from openai import OpenAI
    api = OpenAI()

    with gzip.open(args.labels, "rt", encoding="utf-8") as f:
        targets = {json.loads(line)["news_id"] for line in f}
    texts = {}
    for p in sorted(glob.glob(args.articles)):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["news_id"] in targets and r.get("text"):
                    texts[r["news_id"]] = (r["ticker"], r["text"])
    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        with gzip.open(out_path, "rt", encoding="utf-8") as f:
            done = {r["news_id"]: r for r in map(json.loads, f)}
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    pending = [n for n in sorted(texts) if n not in done]
    if args.limit:
        pending = pending[:args.limit]
    logger.info("대상 %d건 / 완료 %d건 / 남은 %d건", len(texts), len(done), len(pending))

    def save():
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")

    if args.mode == "realtime":
        run_realtime(api, pending, texts, done, save, args)
        return

    cycle = 0
    while pending:
        cycle += 1
        chunk, tokens = [], 0
        for n in pending:
            req = request(n, *texts[n])
            t = estimate_tokens(req)
            if chunk and tokens + t > MAX_ENQUEUED_TOKENS:
                break
            chunk.append(req)
            tokens += t
        path = work / f"events_{cycle:03d}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in chunk) + "\n", encoding="utf-8")
        up = api.files.create(file=path.open("rb"), purpose="batch")
        batch = api.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                   completion_window="24h", metadata={"stage": "events"})
        logger.info("[%d회차] %d건 제출 (추정 %d토큰) %s", cycle, len(chunk), tokens, batch.id)
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(60)
            batch = api.batches.retrieve(batch.id)
        if batch.status != "completed":
            codes = [e.code for e in (batch.errors.data if batch.errors else [])]
            logger.warning("  %s %s — 5분 후 재시도", batch.status, codes)
            time.sleep(300)
            continue
        got = 0
        for line in api.files.content(batch.output_file_id).text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            parsed = parse((item.get("response") or {}).get("body") or {})
            if parsed:
                done[item["custom_id"]] = {"news_id": item["custom_id"], **parsed}
                got += 1
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
        pending = [n for n in pending if n not in done]
        logger.info("  완료 %d건 수거, 누적 %d건, 남은 %d건", got, len(done), len(pending))
        path.unlink(missing_ok=True)
        if args.commit and (cycle % 3 == 0 or not pending):
            git_commit(str(out_path), f"이벤트 추출 진행: {len(done)}건")
    logger.info("전량 완료: %d건", len(done))


if __name__ == "__main__":
    main()
