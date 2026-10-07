"""EDT 보도자료에서 LLM 이벤트와 임베딩을 만든다 (일반 API 병렬 호출).

extract_events.py 와 같은 프롬프트·이벤트 유형을 쓰되, 회사 이름 대신 '보도자료 발행사'로
지칭한다. 임베딩은 제목 + 본문 앞 800자를 text-embedding-3-small 로 만든다.

    export OPENAI_API_KEY=...
    python -I extract_edt_events.py --text <작업폴더>/edt_text.jsonl.gz --commit
결과
    data/edt/edt_llm.jsonl.gz                 edt_id 별 이벤트 속성
    data/edt/edt_emb.partNN.npy (float16)     임베딩, data/edt/edt_emb_ids.npy 순서
"""
import argparse
import gzip
import json
import logging
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_events import EVENT_TYPES, PROMPT, parse  # noqa: E402

logger = logging.getLogger(__name__)


def retry(fn):
    delay = 2.0
    for _ in range(8):
        try:
            return fn()
        except Exception as exc:
            if not any(k in type(exc).__name__ for k in ("RateLimit", "Connection", "Timeout", "InternalServer")):
                logger.warning("  실패: %s", type(exc).__name__)
                return None
            time.sleep(delay * (1 + random.uniform(-0.3, 0.3)))
            delay = min(delay * 2, 60)
    return None


def commit(paths, msg):
    subprocess.run(["git", "add", "-f", *paths], check=False)
    subprocess.run(["git", "-c", "user.email=sunshine31885@gmail.com", "-c", "user.name=Claude", "commit", "-q", "-m",
                    msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                    "Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK"], check=False)
    b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
    for d in (2, 4, 8, 16):
        subprocess.run(["git", "pull", "-q", "--no-rebase", "origin", b], check=False)
        if subprocess.run(["git", "push", "-q", "origin", b]).returncode == 0:
            return
        time.sleep(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="data/edt/edt_llm.jsonl.gz")
    ap.add_argument("--emb-dir", default="data/edt")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from openai import OpenAI
    api = OpenAI()
    with gzip.open(a.text, "rt", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    out = Path(a.out)
    done = {}
    if out.exists():
        with gzip.open(out, "rt", encoding="utf-8") as f:
            done = {r["edt_id"]: r for r in map(json.loads, f)}
    todo = [x for x in items if x["edt_id"] not in done]
    logger.info("전체 %d건 / 완료 %d건 / 남은 %d건", len(items), len(done), len(todo))

    def one(x):
        content = PROMPT.format(name="the issuer of this press release", ticker=x["ticker"],
                                types=", ".join(EVENT_TYPES), text=(x["title"] + "\n" + x["text"])[:1500])
        resp = retry(lambda: api.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=120, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}]))
        if resp is None:
            return x["edt_id"], None
        return x["edt_id"], parse({"choices": [{"message": {"content": resp.choices[0].message.content}}]})

    def save():
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")

    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(one, x) for x in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            eid, p = fut.result()
            if p:
                with lock:
                    done[eid] = {"edt_id": eid, **p}
            if k % 2000 == 0 or k == len(futs):
                save()
                logger.info("  이벤트 %d/%d (%.0f건/분)", k, len(futs), k / (time.time() - t0) * 60)
    save()

    # 임베딩: 100건씩 묶어 요청
    ids = np.array([x["edt_id"] for x in items])
    vecs = np.zeros((len(items), 1536), dtype=np.float16)
    batches = [list(range(i, min(i + 100, len(items)))) for i in range(0, len(items), 100)]

    def emb(b):
        texts = [(items[j]["title"] + "\n" + items[j]["text"])[:800] for j in b]
        resp = retry(lambda: api.embeddings.create(model="text-embedding-3-small", input=texts))
        return b, (None if resp is None else [d.embedding for d in resp.data])

    with ThreadPoolExecutor(max_workers=8) as pool:
        for k, fut in enumerate(as_completed([pool.submit(emb, b) for b in batches]), 1):
            b, v = fut.result()
            if v is not None:
                vecs[b] = np.asarray(v, dtype=np.float16)
            if k % 100 == 0:
                logger.info("  임베딩 %d/%d", k, len(batches))
    ed = Path(a.emb_dir)
    np.save(ed / "edt_emb_ids.npy", ids)
    parts = []
    for p, s in enumerate(range(0, len(vecs), 25000)):
        path = ed / f"edt_emb.part{p:02d}.npy"
        np.save(path, vecs[s:s + 25000])
        parts.append(str(path))
    logger.info("완료: 이벤트 %d건, 임베딩 %d건", len(done), int((np.abs(vecs).sum(1) > 0).sum()))
    if a.commit:
        commit([str(out), str(ed / "edt_emb_ids.npy"), *parts], f"EDT 이벤트·임베딩 생성: {len(done)}건")


if __name__ == "__main__":
    main()
