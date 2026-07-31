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


def main() -> None:
    parser = argparse.ArgumentParser(description="감성 점수 + 의미 임베딩 채우기")
    parser.add_argument("--input", required=True, help="prepare_fnspid.py 출력 JSONL")
    parser.add_argument("--output", required=True, help="학습용 캐시 JSONL")
    parser.add_argument("--batch-size", type=int, default=64, help="임베딩 배치 크기")
    parser.add_argument("--limit", type=int, default=0,
                        help="처리할 최대 건수 (0이면 전체). 비용 시험용")
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

    with output_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]

            # 1) 감성 점수 — 이미 있으면(FNSPID Sentiment_gpt 등) 재사용한다.
            for record in chunk:
                if "sentiment" in record:
                    continue
                try:
                    _, sentiment = extractor.summarize_and_score(record["summary"])
                except Exception:
                    logger.exception("감성 추출 실패(%s) → 0.0", record["news_id"])
                    sentiment = 0.0
                record["sentiment"] = sentiment

            # 2) 의미 임베딩
            try:
                vectors = extractor.embed([r["summary"] for r in chunk])
            except Exception:
                logger.exception("임베딩 실패, 이 배치를 건너뜁니다.")
                continue

            for record, vector in zip(chunk, vectors):
                record["embedding"] = vector
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            logger.info("진행 %d / %d",
                        min(start + args.batch_size, len(pending)), len(pending))

    logger.info("완료 — %s", output_path)


if __name__ == "__main__":
    main()
