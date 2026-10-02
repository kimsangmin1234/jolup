"""OpenAI Batch API로 감성·임베딩을 대량 생성한다.

일반 API는 gpt-4o-mini 의 일일 요청 한도(Tier 1 기준 10,000건)에 걸려
82,903건 처리에 8일 이상이 걸린다. Batch API 는 이 한도와 별개로 동작하고
비용이 절반이며, 결과는 최대 24시간 안에 나온다.

처리 순서 (임베딩이 요약 결과에 의존하므로 두 단계로 나뉜다)::

    submit chat    본문 → 요약 + 감성 요청 제출
    collect chat   결과 수거 → summary, sentiment 확보
    submit embed   요약 → 임베딩 요청 제출
    collect embed  결과 수거 → 최종 학습 캐시 완성

각 단계는 재실행 안전하다. 이미 처리된 레코드는 건너뛴다.

사용 예::

    python batch_enrich.py submit  --stage chat  --input records_article.jsonl
    python batch_enrich.py status
    python batch_enrich.py collect --stage chat  --input records_article.jsonl
    python batch_enrich.py submit  --stage embed
    python batch_enrich.py collect --stage embed --out news_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import NewsEncoderConfig
from modules.news_encoder import (LANGUAGE_SAME, SUMMARY_SENTIMENT_PROMPT,
                                  _clip_sentiment)

logger = logging.getLogger(__name__)

# 본문 길이 상한. 전체 레코드의 98%가 잘리지 않고 통과하는 지점이다.
MAX_CHARS = 11_114

# Batch 파일 한도: 요청 50,000건, 200MB. 여유를 두고 자른다.
MAX_REQUESTS_PER_FILE = 40_000
MAX_BYTES_PER_FILE = 150 * 1024 * 1024

# 조직당 대기 토큰 상한. gpt-4o-mini 는 2,000,000 이며 이를 넘기면 배치가
# token_limit_exceeded 로 즉시 실패한다. 여유를 두고 자른 뒤 순차 제출한다.
MAX_ENQUEUED_TOKENS = 1_200_000


def estimate_tokens(request: dict) -> int:
    """요청 하나의 대기 토큰을 어림한다.

    영문 기준 문자 3.5개당 1토큰으로 보고, 출력분 여유를 더한다.
    정확할 필요는 없고 상한을 넘지 않을 만큼만 보수적이면 된다.
    """
    body = request["body"]
    if "messages" in body:
        chars = sum(len(m["content"]) for m in body["messages"])
        return int(chars / 3.0) + 500
    return int(len(str(body.get("input", ""))) / 3.5) + 20


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def client():
    from openai import OpenAI
    return OpenAI()


# --------------------------------------------------------------------------
# 요청 파일 만들기
# --------------------------------------------------------------------------

def chat_request(record: dict, config: NewsEncoderConfig) -> dict:
    article = str(record.get("text") or record.get("summary") or "")[:MAX_CHARS]
    return {
        "custom_id": record["news_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": config.llm_model,
            "messages": [{"role": "user", "content": SUMMARY_SENTIMENT_PROMPT.format(
                article=article, language=LANGUAGE_SAME)}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            # 요약 3문장 + 감성 점수면 충분하다. 상한을 두면 대기 토큰 산정이
            # 안정되고 출력이 길어지는 일도 막는다.
            "max_tokens": 500,
        },
    }


def embed_request(record: dict, config: NewsEncoderConfig) -> dict:
    return {
        "custom_id": record["news_id"],
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": config.embedding_model, "input": record["summary"]},
    }


def write_batch_files(requests: list[dict], work: Path, stage: str,
                      start_index: int = 0) -> list[Path]:
    """요청을 파일·토큰 한도에 맞춰 여러 조각으로 나눠 쓴다."""
    paths: list[Path] = []
    chunk: list[str] = []
    size = 0
    tokens = 0

    def flush() -> None:
        nonlocal chunk, size, tokens
        if not chunk:
            return
        path = work / f"{stage}_requests_{start_index + len(paths):03d}.jsonl"
        path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        paths.append(path)
        chunk, size, tokens = [], 0, 0

    for request in requests:
        line = json.dumps(request, ensure_ascii=False)
        encoded = len(line.encode()) + 1
        estimate = estimate_tokens(request)
        if chunk and (len(chunk) >= MAX_REQUESTS_PER_FILE
                      or size + encoded > MAX_BYTES_PER_FILE
                      or tokens + estimate > MAX_ENQUEUED_TOKENS):
            flush()
        chunk.append(line)
        size += encoded
        tokens += estimate
    flush()
    return paths


# --------------------------------------------------------------------------
# 단계별 동작
# --------------------------------------------------------------------------

def cmd_submit(args) -> None:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    config = NewsEncoderConfig()
    state_path = work / f"{args.stage}_batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}

    if args.stage == "chat":
        records = load_jsonl(Path(args.input))
        done = _collected_ids(work, "chat")
        pending = [r for r in records if r["news_id"] not in done]
        builder = chat_request
    else:
        records = load_jsonl(work / "chat_results.jsonl")
        done = _collected_ids(work, "embed")
        pending = [r for r in records if r["news_id"] not in done]
        builder = embed_request

    if args.limit:
        pending = pending[: args.limit]
    logger.info("%s 단계: 전체 %d / 미처리 %d", args.stage, len(records), len(pending))
    if not pending:
        logger.info("제출할 요청이 없습니다.")
        return

    paths = write_batch_files([builder(r, config) for r in pending], work, args.stage)
    logger.info("요청 파일 %d개 생성", len(paths))

    api = client()
    for path in paths:
        uploaded = api.files.create(file=path.open("rb"), purpose="batch")
        batch = api.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions" if args.stage == "chat" else "/v1/embeddings",
            completion_window="24h",
            metadata={"stage": args.stage, "file": path.name},
        )
        state["batches"].append({"id": batch.id, "file": path.name, "status": batch.status})
        logger.info("  제출 %s ← %s (%s)", batch.id, path.name, batch.status)

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_status(args) -> None:
    work = Path(args.work)
    api = client()
    for stage in ("chat", "embed"):
        state_path = work / f"{stage}_batches.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        print(f"[{stage}]")
        for entry in state["batches"]:
            batch = api.batches.retrieve(entry["id"])
            counts = batch.request_counts
            print(f"  {batch.id}  {batch.status:12} "
                  f"완료 {counts.completed}/{counts.total} 실패 {counts.failed}")
            entry["status"] = batch.status
            entry["output_file_id"] = batch.output_file_id
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_collect(args) -> None:
    work = Path(args.work)
    api = client()
    state_path = work / f"{args.stage}_batches.json"
    state = json.loads(state_path.read_text())

    rows: dict[str, dict] = {}
    for entry in state["batches"]:
        batch = api.batches.retrieve(entry["id"])
        if batch.status != "completed":
            logger.warning("  %s 아직 %s — 건너뜀", batch.id, batch.status)
            continue
        content = api.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            body = (item.get("response") or {}).get("body")
            if not body:
                continue
            rows[item["custom_id"]] = body
    logger.info("수거한 응답 %d건", len(rows))

    if args.stage == "chat":
        records = load_jsonl(Path(args.input))
        out_path = work / "chat_results.jsonl"
        existing = {r["news_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
        written = 0
        with out_path.open("a", encoding="utf-8") as f:
            for record in records:
                if record["news_id"] in existing or record["news_id"] not in rows:
                    continue
                body = rows[record["news_id"]]
                try:
                    payload = json.loads(body["choices"][0]["message"]["content"])
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                updated = {k: v for k, v in record.items() if k != "text"}
                updated["summary"] = str(payload.get("summary", "")).strip()
                updated["sentiment"] = _clip_sentiment(payload.get("sentiment", 0.0))
                if not updated["summary"]:
                    continue
                f.write(json.dumps(updated, ensure_ascii=False) + "\n")
                written += 1
        logger.info("chat_results.jsonl 에 %d건 추가 (누적 %d)", written, len(existing) + written)
    else:
        records = load_jsonl(work / "chat_results.jsonl")
        out_path = Path(args.out)
        existing = {r["news_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
        written = 0
        with out_path.open("a", encoding="utf-8") as f:
            for record in records:
                if record["news_id"] in existing or record["news_id"] not in rows:
                    continue
                body = rows[record["news_id"]]
                try:
                    vector = body["data"][0]["embedding"]
                except (KeyError, IndexError):
                    continue
                updated = dict(record)
                updated["embedding"] = vector
                f.write(json.dumps(updated, ensure_ascii=False) + "\n")
                written += 1
        logger.info("%s 에 %d건 추가 (누적 %d)", out_path, written, len(existing) + written)


def _collected_ids(work: Path, stage: str) -> set[str]:
    path = work / ("chat_results.jsonl" if stage == "chat" else "embed_results.jsonl")
    if not path.exists():
        return set()
    return {r["news_id"] for r in load_jsonl(path)}


def cmd_run(args) -> None:
    """한 번에 한 배치씩 제출·대기·수거를 반복해 전량을 끝낸다.

    조직당 대기 토큰 상한(2,000,000) 때문에 여러 배치를 동시에 올릴 수 없다.
    하나가 끝나야 다음을 올릴 수 있으므로 순차로 돈다. 중간에 끊겨도
    수거분은 남으므로 같은 명령으로 이어서 실행하면 된다.
    """
    import time

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    config = NewsEncoderConfig()
    api = client()

    for cycle in range(1, args.max_cycles + 1):
        if args.stage == "chat":
            records = load_jsonl(Path(args.input))
            done = _collected_ids(work, "chat")
            builder = chat_request
            endpoint = "/v1/chat/completions"
        else:
            records = load_jsonl(work / "chat_results.jsonl")
            done = {r["news_id"] for r in load_jsonl(Path(args.out))} if Path(args.out).exists() else set()
            builder = embed_request
            endpoint = "/v1/embeddings"

        pending = [r for r in records if r["news_id"] not in done]
        logger.info("[%d회차] %s 전체 %d / 남은 %d", cycle, args.stage, len(records), len(pending))
        if not pending:
            logger.info("전량 완료")
            return

        # 대기 토큰 한도에 맞춰 이번에 올릴 한 조각만 만든다.
        paths = write_batch_files([builder(r, config) for r in pending], work,
                                  args.stage, start_index=cycle)
        path = paths[0]
        for extra in paths[1:]:
            extra.unlink(missing_ok=True)

        uploaded = api.files.create(file=path.open("rb"), purpose="batch")
        batch = api.batches.create(input_file_id=uploaded.id, endpoint=endpoint,
                                   completion_window="24h",
                                   metadata={"stage": args.stage})
        logger.info("  제출 %s (%s)", batch.id, path.name)

        # 배치 ID를 기록해 둔다. 중간에 프로세스가 죽어도 나중에 수거할 수 있다.
        state_path = work / f"{args.stage}_batches.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
        state["batches"].append({"id": batch.id, "file": path.name, "status": batch.status})
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        # 완료까지 대기
        while True:
            time.sleep(args.poll)
            batch = api.batches.retrieve(batch.id)
            counts = batch.request_counts
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                logger.info("  %s — 완료 %d 실패 %d", batch.status,
                            counts.completed, counts.failed)
                break
            logger.info("  %s 진행 %d/%d", batch.status, counts.completed, counts.total)

        if batch.status != "completed":
            codes = [e.code for e in (batch.errors.data if batch.errors else [])]
            for err in (batch.errors.data if batch.errors else [])[:2]:
                logger.error("  %s: %s", err.code, str(err.message)[:200])
            path.unlink(missing_ok=True)
            # 대기 토큰 한도 초과는 앞선 배치의 토큰이 아직 반환되지 않아 생긴다.
            # 기다렸다 다시 올리면 된다. 그 밖의 실패만 중단한다.
            if "token_limit_exceeded" in codes:
                logger.warning("  대기 토큰 한도 초과 — %d초 후 다시 제출", args.retry_wait)
                time.sleep(args.retry_wait)
                continue
            logger.error("배치가 %s 로 끝났습니다. 중단합니다.", batch.status)
            raise SystemExit(2)

        # 수거
        content = api.files.content(batch.output_file_id).text
        rows = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            body = (item.get("response") or {}).get("body")
            if body:
                rows[item["custom_id"]] = body
        _write_results(args, work, records, rows)
        path.unlink(missing_ok=True)


def _write_results(args, work: Path, records: list[dict], rows: dict) -> None:
    """수거한 응답을 결과 파일에 덧붙인다."""
    if args.stage == "chat":
        out_path = work / "chat_results.jsonl"
        existing = _collected_ids(work, "chat")
        written = 0
        with out_path.open("a", encoding="utf-8") as f:
            for record in records:
                if record["news_id"] in existing or record["news_id"] not in rows:
                    continue
                try:
                    payload = json.loads(
                        rows[record["news_id"]]["choices"][0]["message"]["content"])
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                summary = str(payload.get("summary", "")).strip()
                if not summary:
                    continue
                updated = {k: v for k, v in record.items() if k != "text"}
                updated["summary"] = summary
                updated["sentiment"] = _clip_sentiment(payload.get("sentiment", 0.0))
                f.write(json.dumps(updated, ensure_ascii=False) + "\n")
                written += 1
        logger.info("  chat_results.jsonl +%d건", written)
    else:
        out_path = Path(args.out)
        existing = {r["news_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
        written = 0
        with out_path.open("a", encoding="utf-8") as f:
            for record in records:
                if record["news_id"] in existing or record["news_id"] not in rows:
                    continue
                try:
                    vector = rows[record["news_id"]]["data"][0]["embedding"]
                except (KeyError, IndexError):
                    continue
                updated = dict(record)
                updated["embedding"] = vector
                f.write(json.dumps(updated, ensure_ascii=False) + "\n")
                written += 1
        logger.info("  %s +%d건", out_path.name, written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch API 기반 감성·임베딩 생성")
    parser.add_argument("command", choices=("submit", "status", "collect", "run"))
    parser.add_argument("--stage", choices=("chat", "embed"), default="chat")
    parser.add_argument("--input", default="", help="본문 레코드 JSONL (chat 단계)")
    parser.add_argument("--out", default="", help="최종 캐시 경로 (embed 단계)")
    parser.add_argument("--work", required=True, help="배치 작업 디렉터리")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--poll", type=int, default=60, help="run: 상태 확인 주기(초)")
    parser.add_argument("--retry-wait", type=int, default=300,
                        help="run: 대기 토큰 한도 초과 시 재제출 전 대기(초)")
    parser.add_argument("--max-cycles", type=int, default=400,
                        help="run: 최대 반복 횟수")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"submit": cmd_submit, "status": cmd_status,
     "collect": cmd_collect, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
