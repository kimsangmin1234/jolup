"""모듈 1. LLM 기반 주 신호 생성 (Ⅱ.1)

한 건의 뉴스 본문을 입력으로 받아 의미적 정보와 감성적 정보를 동시에 반영한
512차원 주 신호 벡터를 생성한다.

    뉴스 본문
      └─ GPT-4o-mini ─┬─ 요약 텍스트 ─ text-embedding-3-small ─ 1536차원 임베딩 ─┐
                      └─ 감성 점수 (-1 ~ +1 스칼라) ───────────────────────────┤
                                                                              │
                                          1537차원 결합 벡터 ─ 완전연결층 ─ 512차원 주 신호

이 파일은 두 부분으로 구성된다.

* ``NewsLLMExtractor`` : OpenAI API를 호출하는 **비학습** 전처리기.
  요약/감성/임베딩 추출은 학습 루프 밖에서 한 번만 수행하고 캐시하는 것이
  비용·속도 면에서 유리하므로 nn.Module과 분리했다.
* ``NewsSignalEncoder`` : 1537차원 결합 벡터를 512차원으로 사영하는 **학습 가능**
  완전연결층. 실제 역전파가 흐르는 부분이다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from config import NewsEncoderConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1-1. LLM 전처리 (비학습): 요약 + 감성 점수 + 의미 임베딩
# --------------------------------------------------------------------------

SUMMARY_SENTIMENT_PROMPT = """당신은 금융 뉴스 분석가입니다.
주어진 뉴스 본문을 분석하여 아래 JSON 형식으로만 답하십시오.

- summary: 주가에 영향을 줄 수 있는 핵심 사건 중심으로 3문장 이내 한국어 요약.
- sentiment: 이 뉴스가 해당 종목 주가에 미칠 영향의 방향과 강도.
             -1.0(매우 부정) ~ +1.0(매우 긍정) 사이의 실수 하나.

{{"summary": "...", "sentiment": 0.0}}

[뉴스 본문]
{article}
"""


@dataclass
class NewsAnalysis:
    """LLM 전처리 결과 한 건."""

    summary: str
    sentiment: float          # -1.0 ~ +1.0
    embedding: list[float]    # 1536차원 의미 임베딩

    def to_vector(self) -> list[float]:
        """1537차원 결합 벡터 (임베딩 ‖ 감성 점수)."""
        return list(self.embedding) + [self.sentiment]


class NewsLLMExtractor:
    """GPT-4o-mini 요약/감성 추출 + text-embedding-3-small 임베딩 생성.

    학습 전에 오프라인으로 실행하여 결과를 캐시하는 용도이다.
    ``openai`` 패키지와 ``OPENAI_API_KEY`` 환경변수가 필요하다.
    """

    def __init__(self, config: NewsEncoderConfig | None = None, client=None) -> None:
        self.config = config or NewsEncoderConfig()
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI  # 지연 임포트: 학습만 할 때는 불필요

            self.client = OpenAI()

    # -- 요약 + 감성 -------------------------------------------------------
    def summarize_and_score(self, article: str) -> tuple[str, float]:
        """뉴스 본문 → (요약 텍스트, 감성 점수)."""
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user",
                       "content": SUMMARY_SENTIMENT_PROMPT.format(article=article)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        payload = json.loads(response.choices[0].message.content)
        summary = str(payload.get("summary", "")).strip()
        sentiment = _clip_sentiment(payload.get("sentiment", 0.0))
        return summary, sentiment

    # -- 의미 임베딩 -------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """요약 텍스트 → 1536차원 의미 임베딩."""
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=list(texts),
        )
        vectors = [item.embedding for item in response.data]
        for vector in vectors:
            if len(vector) != self.config.embedding_dim:
                raise ValueError(
                    f"임베딩 차원 불일치: {len(vector)} != {self.config.embedding_dim}"
                )
        return vectors

    # -- 파이프라인 --------------------------------------------------------
    def analyze(self, articles: Sequence[str]) -> list[NewsAnalysis]:
        """뉴스 본문 목록 → NewsAnalysis 목록."""
        summaries: list[str] = []
        sentiments: list[float] = []
        for article in articles:
            summary, sentiment = self.summarize_and_score(article)
            # 요약이 비면 원문 앞부분으로 대체하여 임베딩 실패를 막는다.
            summaries.append(summary or article[:1000])
            sentiments.append(sentiment)

        embeddings = self.embed(summaries)
        return [
            NewsAnalysis(summary=s, sentiment=v, embedding=e)
            for s, v, e in zip(summaries, sentiments, embeddings)
        ]


def _clip_sentiment(value) -> float:
    """감성 점수를 [-1, +1] 실수로 강제한다."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        logger.warning("감성 점수 파싱 실패(%r) → 0.0으로 대체", value)
        return 0.0
    return max(-1.0, min(1.0, score))


# --------------------------------------------------------------------------
# 1-2. 학습 가능 완전연결층: 1537차원 결합 벡터 → 512차원 주 신호
# --------------------------------------------------------------------------

class NewsSignalEncoder(nn.Module):
    """1537차원 결합 벡터를 512차원 주 신호 벡터로 사영한다.

    입력  : (B, 1536) 의미 임베딩, (B,) 또는 (B, 1) 감성 점수
    출력  : (B, 512) 주 신호 벡터
    """

    def __init__(self, config: NewsEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or NewsEncoderConfig()

        self.projection = nn.Linear(self.config.fusion_input_dim, self.config.d_model)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(
        self,
        embedding: torch.Tensor,
        sentiment: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.dim() != 2:
            raise ValueError(f"embedding은 (B, D) 형태여야 합니다: {tuple(embedding.shape)}")
        if embedding.size(-1) != self.config.embedding_dim:
            raise ValueError(
                f"임베딩 차원 불일치: {embedding.size(-1)} != {self.config.embedding_dim}"
            )

        if sentiment.dim() == 1:
            sentiment = sentiment.unsqueeze(-1)          # (B,) -> (B, 1)
        if sentiment.size(-1) != self.config.sentiment_dim:
            raise ValueError(
                f"감성 점수 차원 불일치: {sentiment.size(-1)} != {self.config.sentiment_dim}"
            )

        sentiment = sentiment.to(embedding.dtype)
        combined = torch.cat([embedding, sentiment], dim=-1)  # (B, 1537)

        signal = self.projection(combined)   # (B, 512)
        signal = self.activation(signal)
        signal = self.norm(signal)
        return self.dropout(signal)

    def forward_from_vector(self, combined: torch.Tensor) -> torch.Tensor:
        """이미 결합된 1537차원 벡터를 그대로 받는 경로."""
        embedding = combined[..., : self.config.embedding_dim]
        sentiment = combined[..., self.config.embedding_dim :]
        return self.forward(embedding, sentiment)
