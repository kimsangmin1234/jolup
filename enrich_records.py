"""준비된 레코드에 감성 점수와 의미 임베딩을 채운다.

``prepare_fnspid.py --embedding none`` 으로 만든 레코드(요약·라벨만 있는 상태)를
읽어, 논문 모듈 1의 나머지 절반인 감성 추출과 의미 임베딩을 수행한다.

전처리와 학습을 분리해 두면 다음이 가능하다.

* 대용량 데이터 준비를 API 키 없이 먼저 끝내 둔다.
* 키가 준비된 뒤 이 스크립트만 실행하면 학습으로 넘어갈 수 있다.
* 중간에 끊겨도 이미 채운 레코드는 건너뛰므로 재실행이 안전하다.

사용 예::

    export OPENAI_API_KEY=sk-...
    python enrich_records.py \\
        --input data/fnspid_records_raw.jsonl \\
        --output data/news_cache.jsonl

    python train.py --records data/news_cache.jsonl \\
                    --indicators data/indicators.npz \\
                    --train-end 2022-06-30 --valid-end 2023-03-31
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
from pathlib import Path

from config import NewsEncoderConfig

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_done_ids(path: Path) -> set[str]:
    """이미 감성·임베딩이 채워진 news_id 집합."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for record in load_jsonl(path):
        if record.get("embedding") and "sentiment" in record:
            done.add(record["news_id"])
    return done


def pick_text(record: dict, override: str = "") -> str:
    """LLM에 넣을 원문을 고른다.

    논문은 GPT-4o-mini 가 뉴스 본문을 요약한다고 기술하므로 본문(``text``)을
    우선한다. 본문이 없는 구형 레코드는 ``summary`` 로 넘어간다.
    """
    if override:
        return str(record.get(override, "")).strip()
    return str(record.get("text") or record.get("summary") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="감성 점수 + 의미 임베딩 채우기")
    parser.add_argument("--input", required=True, help="prepare_fnspid.py 출력 JSONL")
    parser.add_argument("--output", required=True, help="학습용 캐시 JSONL")
    parser.add_argument("--batch-size", type=int, default=64, help="임베딩 배치 크기")
    parser.add_argument("--limit", type=int, default=0,
                        help="처리할 최대 건수 (0이면 전체). 비용 시험용")
    parser.add_argument("--workers", type=int, default=8,
                        help="동시 처리 스레드 수. 8만 건 규모에서는 필수다.")
    parser.add_argument("--text-field", default="",
                        help="LLM에 넣을 필드. 기본은 text(본문)가 있으면 text, 없으면 summary.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    done = load_done_ids(output_path)
    pending = [r for r in records if r["news_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    logger.info("전체 %d건 / 이미 처리 %d건 / 이번 처리 %d건",
                len(records), len(done), len(pending))
    if not pending:
        logger.info("처리할 레코드가 없습니다.")
        return

    from modules.news_encoder import NewsLLMExtractor

    config = NewsEncoderConfig()
    extractor = NewsLLMExtractor(config)

    todo: queue.Queue = queue.Queue()
    for record in pending:
        todo.put(record)

    lock = threading.Lock()
    done_count = [0]
    usage = {"prompt": 0, "completion": 0, "embed": 0}

    def worker(out) -> None:
        while True:
            try:
                record = todo.get_nowait()
            except queue.Empty:
                return

            source = pick_text(record, args.text_field)
            if not source:
                continue

            # 1) 본문 → 요약 + 감성 (논문 모듈 1의 LLM 단계)
            try:
                summary, sentiment = extractor.summarize_and_score(source)
            except Exception:
                logger.exception("요약·감성 실패(%s)", record["news_id"])
                continue
            summary = summary or source[:1000]

            # 2) 요약 → 의미 임베딩
            try:
                vector = extractor.embed([summary])[0]
            except Exception:
                logger.exception("임베딩 실패(%s)", record["news_id"])
                continue

            updated = dict(record)
            updated["summary"] = summary
            updated["sentiment"] = sentiment
            updated["embedding"] = vector
            # 본문은 캐시 용량이 커지므로 남기지 않는다(원본은 articles/ 에 있다).
            updated.pop("text", None)

            line = json.dumps(updated, ensure_ascii=False)
            with lock:
                out.write(line + "\n")
                out.flush()
                done_count[0] += 1
                current = done_count[0]
            if current % 100 == 0:
                logger.info("진행 %d / %d", current, len(pending))

    with output_path.open("a", encoding="utf-8") as out:
        threads = [threading.Thread(target=worker, args=(out,), daemon=True)
                   for _ in range(max(1, args.workers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    logger.info("처리 완료 %d건", done_count[0])
    logger.info("완료 — %s", output_path)


if __name__ == "__main__":
    main()
