"""OpenAI 호환 목(mock) 서버.

실제 API 키나 네트워크 없이 전체 파이프라인(전처리 → 학습)을 검증하기 위한
로컬 서버이다. `openai` SDK가 실제로 호출하는 두 엔드포인트를 구현한다.

    POST /v1/chat/completions   → 요약 + 감성 점수 (GPT-4o-mini 대체)
    POST /v1/embeddings         → 1536차원 의미 임베딩 (text-embedding-3-small 대체)

응답은 입력 텍스트로부터 결정론적으로 생성되며, **감성과 임베딩이 서로 상관되도록**
설계했다. 따라서 이 서버로 만든 데이터에서는 모델이 실제로 학습 가능한 신호가
존재하고, 손실이 내려가는지로 파이프라인의 정상 동작을 확인할 수 있다.

사용법::

    python tests/mock_openai_server.py --port 8000 &
    export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    export OPENAI_API_KEY=dummy-key
    python preprocess.py --input news_raw.jsonl --output data/news_cache.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

EMBEDDING_DIM = 1536


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def fake_sentiment(text: str) -> float:
    """텍스트에서 결정론적으로 -1 ~ +1 감성 점수를 만든다.

    긍정/부정 단어가 있으면 그 방향을 따르고, 없으면 해시 기반으로 정한다.
    """
    lowered = text.lower()
    positive = sum(lowered.count(w) for w in
                   ("surge", "beat", "record", "growth", "upgrade", "profit", "호재", "상승"))
    negative = sum(lowered.count(w) for w in
                   ("plunge", "miss", "lawsuit", "decline", "downgrade", "loss", "악재", "하락"))

    if positive or negative:
        score = (positive - negative) / (positive + negative)
    else:
        score = ((_seed(text) % 2001) - 1000) / 1000.0
    return max(-1.0, min(1.0, score))


def fake_embedding(text: str) -> list[float]:
    """감성과 상관된 결정론적 1536차원 단위 벡터.

    앞쪽 차원에 감성 성분을 실어 두어, 하류 모델이 학습할 신호가 존재하게 한다.
    """
    sentiment = fake_sentiment(text)
    seed = _seed(text)

    # 선형 합동 생성기로 난수 성분을 만든다 (numpy 의존 없이 결정론적).
    vector: list[float] = []
    state = seed
    for _ in range(EMBEDDING_DIM):
        state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
        vector.append(((state >> 33) / float(2**31)) - 1.0)

    # 앞쪽 64차원에 감성 성분을 주입한다.
    for i in range(64):
        vector[i] = 0.7 * sentiment + 0.3 * vector[i]

    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")

        if self.path.endswith("/chat/completions"):
            self._respond(self._chat(payload))
        elif self.path.endswith("/embeddings"):
            self._respond(self._embeddings(payload))
        else:
            self.send_error(404, f"unknown path: {self.path}")

    # -- 엔드포인트 --------------------------------------------------------
    def _chat(self, payload: dict) -> dict:
        prompt = payload["messages"][-1]["content"]

        # preprocess.py의 프롬프트에서 [뉴스 본문] 이후를 기사로 취급한다.
        match = re.search(r"\[뉴스 본문\]\s*(.*)", prompt, re.DOTALL)
        article = (match.group(1) if match else prompt).strip()

        # 요약: 앞 2문장. 목 서버이므로 실제 요약 알고리즘은 쓰지 않는다.
        sentences = re.split(r"(?<=[.!?])\s+", article)
        summary = " ".join(sentences[:2])[:500] or article[:200]

        content = json.dumps(
            {"summary": summary, "sentiment": round(fake_sentiment(article), 4)},
            ensure_ascii=False,
        )
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "gpt-4o-mini"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _embeddings(self, payload: dict) -> dict:
        texts = payload["input"]
        if isinstance(texts, str):
            texts = [texts]
        return {
            "object": "list",
            "model": payload.get("model", "text-embedding-3-small"),
            "data": [
                {"object": "embedding", "index": i, "embedding": fake_embedding(t)}
                for i, t in enumerate(texts)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    # -- 유틸 --------------------------------------------------------------
    def _respond(self, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args) -> None:
        pass  # 요청 로그를 끈다 (검증 출력이 묻히지 않도록)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 호환 목 서버")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), MockOpenAIHandler)
    print(f"mock OpenAI server on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
