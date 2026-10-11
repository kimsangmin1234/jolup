"""FNSPID 기사 본문 앞 800자로 임베딩을 만든다 (EDT 와 같은 방식, 전이 검증용).

EDT 임베딩은 '제목 + 본문 앞 800자'를 text-embedding-3-small 로 만들었다. FNSPID 레코드에는 제목이 없어
본문 앞 800자만 쓴다. 기존 FNSPID 임베딩은 GPT 요약문으로 만든 것이라 EDT 와 직접 비교할 수 없다.

    export OPENAI_API_KEY=...
    python embed_fnspid_articles.py
결과: data/fnspid/article_emb.partNN.npy (float16), data/fnspid/article_emb_ids.json
"""
import glob
import gzip
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    from openai import OpenAI
    api = OpenAI()
    with gzip.open("data/fnspid/event_features.jsonl.gz", "rt") as f:
        want = {json.loads(line)["news_id"] for line in f}
    texts = {}
    for p in sorted(glob.glob("data/fnspid/articles/records_article.part*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["news_id"] in want and r.get("text"):
                    texts[r["news_id"]] = r["text"][:800]
    ids = sorted(texts)
    vecs = np.zeros((len(ids), 1536), dtype=np.float16)
    batches = [list(range(i, min(i + 100, len(ids)))) for i in range(0, len(ids), 100)]

    def emb(b):
        delay = 2.0
        for _ in range(8):
            try:
                resp = api.embeddings.create(model="text-embedding-3-small", input=[texts[ids[j]] for j in b])
                return b, [d.embedding for d in resp.data]
            except Exception:
                time.sleep(delay * (1 + random.uniform(-0.3, 0.3)))
                delay = min(delay * 2, 60)
        return b, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        for k, fut in enumerate(as_completed([pool.submit(emb, b) for b in batches]), 1):
            b, v = fut.result()
            if v is not None:
                vecs[b] = np.asarray(v, dtype=np.float16)
            if k % 50 == 0:
                logging.info("%d/%d", k, len(batches))
    Path("data/fnspid/article_emb_ids.json").write_text(json.dumps(ids))
    for p, s in enumerate(range(0, len(ids), 20000)):
        np.save(f"data/fnspid/article_emb.part{p:02d}.npy", vecs[s:s + 20000])
    logging.info("완료 %d건 (실패 %d)", len(ids), int((np.abs(vecs).sum(1) == 0).sum()))


if __name__ == "__main__":
    main()
