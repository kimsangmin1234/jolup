"""흩어진 진행분을 모아 Batch API 작업 디렉터리를 복원한다.

요약·감성과 임베딩 결과가 여러 실험 폴더와 OpenAI 계정(완료된 배치 출력)에
나뉘어 있다. 이를 news_id 기준으로 합쳐 ``batch_enrich.py run`` 이 이어서
처리할 수 있는 상태로 만든다. 이미 돈을 낸 결과를 다시 요청하지 않기 위함이다.

    python recover_batches.py --work .work/batch --out .work/news_cache.jsonl

결과:
    <work>/chat_results.jsonl   요약·감성이 끝난 레코드 (본문 제외)
    <out>                       위 레코드 중 임베딩까지 붙은 것
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
from pathlib import Path

from modules.news_encoder import _clip_sentiment

logger = logging.getLogger(__name__)


def read_gz(pattern: str):
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", default="data/fnspid/articles/records_article.part*.jsonl.gz")
    ap.add_argument("--no-api", action="store_true", help="OpenAI 배치 출력은 회수하지 않는다")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 원본 레코드: 라벨·앵커 등 필드의 기준이다(재라벨링 전 형태).
    base = {r["news_id"]: {k: v for k, v in r.items() if k != "text"}
            for r in read_gz(args.articles)}
    logger.info("원본 레코드 %d건", len(base))

    # 1) 요약·감성: 실험 폴더의 chat_results 를 모두 합친다.
    chat: dict[str, dict] = {}
    for r in read_gz("experiments/*/chat_results.jsonl.gz"):
        if r["news_id"] in base and r.get("summary") and r["news_id"] not in chat:
            rec = dict(base[r["news_id"]])
            rec["summary"], rec["sentiment"] = r["summary"], float(r["sentiment"])
            chat[r["news_id"]] = rec
    if not args.no_api:
        from openai import OpenAI
        api = OpenAI()
        for batch in api.batches.list(limit=100):
            if batch.endpoint != "/v1/chat/completions" or batch.status != "completed":
                continue
            added = 0
            for line in api.files.content(batch.output_file_id).text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                nid = item["custom_id"]
                body = (item.get("response") or {}).get("body") or {}
                if nid not in base or nid in chat:
                    continue
                try:
                    payload = json.loads(body["choices"][0]["message"]["content"])
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    continue
                summary = str(payload.get("summary", "")).strip()
                if summary:
                    rec = dict(base[nid])
                    rec["summary"] = summary
                    rec["sentiment"] = _clip_sentiment(payload.get("sentiment", 0.0))
                    chat[nid] = rec
                    added += 1
            if added:
                logger.info("  %s: +%d", batch.id, added)
    logger.info("요약·감성 %d건", len(chat))

    # 2) 임베딩: 깃의 캐시들 + OpenAI 에 남은 완료 배치 출력.
    emb: dict[str, list] = {}
    for r in read_gz("experiments/*/news_cache*.jsonl.gz"):
        if r.get("embedding") and r["news_id"] in chat:
            emb.setdefault(r["news_id"], r["embedding"])
    logger.info("깃 캐시에서 임베딩 %d건", len(emb))
    if not args.no_api:
        for batch in api.batches.list(limit=100):
            if batch.endpoint != "/v1/embeddings" or batch.status != "completed":
                continue
            added = 0
            for line in api.files.content(batch.output_file_id).text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                body = (item.get("response") or {}).get("body") or {}
                nid = item["custom_id"]
                if nid in chat and nid not in emb and body.get("data"):
                    emb[nid] = body["data"][0]["embedding"]
                    added += 1
            if added:
                logger.info("  %s: +%d", batch.id, added)
    logger.info("임베딩 합계 %d건", len(emb))

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    with (work / "chat_results.jsonl").open("w", encoding="utf-8") as f:
        for rec in chat.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with Path(args.out).open("w", encoding="utf-8") as f:
        for nid, rec in chat.items():
            if nid in emb:
                f.write(json.dumps({**rec, "embedding": emb[nid]}, ensure_ascii=False) + "\n")
    logger.info("남은 요약·감성 %d건 / 남은 임베딩 %d건",
                len(base) - len(chat), len(chat) - len(emb))


if __name__ == "__main__":
    main()
