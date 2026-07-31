"""LLM 전처리 캐시 생성 스크립트.

뉴스 원문(JSONL)을 GPT-4o-mini / text-embedding-3-small로 한 번만 처리하여
학습에서 재사용할 캐시를 만든다. 이미 처리한 news_id는 건너뛰므로 중단 후
재실행해도 안전하다.

입력 JSONL 한 줄::

    {"news_id": "...", "ticker": "005930", "date": "2026-03-04",
     "article": "뉴스 본문 ...", "label": 0.0132}

출력 JSONL 한 줄::

    {"news_id": "...", "ticker": "...", "date": "...",
     "summary": "...", "sentiment": 0.42, "embedding": [...1536...], "label": 0.0132}

사용 예::

    OPENAI_API_KEY=... python preprocess.py --input news_raw.jsonl \\
                                            --output data/news_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import NewsEncoderConfig
from modules.news_encoder import NewsLLMExtractor

logger = logging.getLogger(__name__)


def load_done_ids(output_path: Path) -> set[str]:
    """이미 처리된 news_id를 읽어 재실행 시 중복 호출을 막는다."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["news_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="뉴스 LLM 전처리 캐시 생성")
    parser.add_argument("--input", required=True, help="뉴스 원문 JSONL")
    parser.add_argument("--output", required=True, help="캐시 출력 JSONL")
    parser.add_argument("--batch-size", type=int, default=16, help="임베딩 배치 크기")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(output_path)
    logger.info("이미 처리된 뉴스 %d건은 건너뜁니다.", len(done))

    pending = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["news_id"] not in done:
                pending.append(record)

    logger.info("처리 대상 뉴스 %d건", len(pending))

    extractor = NewsLLMExtractor(NewsEncoderConfig())

    with output_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            try:
                analyses = extractor.analyze([r["article"] for r in chunk])
            except Exception:
                logger.exception("배치 %d 처리 실패, 건너뜁니다.", start // args.batch_size)
                continue

            for record, analysis in zip(chunk, analyses):
                out.write(json.dumps({
                    "news_id": record["news_id"],
                    "ticker": record["ticker"],
                    "date": record["date"],
                    "summary": analysis.summary,
                    "sentiment": analysis.sentiment,
                    "embedding": analysis.embedding,
                    "label": record["label"],
                }, ensure_ascii=False) + "\n")
            out.flush()
            logger.info("진행 %d / %d", min(start + args.batch_size, len(pending)), len(pending))


if __name__ == "__main__":
    main()
