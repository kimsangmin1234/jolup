"""논문 "뉴스 이벤트 기반 주식 가격 예측 모델"의 제안 모듈 5종.

1. news_encoder      — LLM 기반 주 신호 생성
2. tcn_encoder       — TCN 기반 보조 벡터 생성
3. cross_attention   — 비대칭 교차 어텐션
4. gated_fusion      — 게이트 기반 잔차 결합
5. return_predictor  — 등락률 예측
"""

from modules.cross_attention import AsymmetricCrossAttention
from modules.gated_fusion import GatedResidualFusion
from modules.news_encoder import NewsAnalysis, NewsLLMExtractor, NewsSignalEncoder
from modules.return_predictor import ReturnPredictor
from modules.tcn_encoder import CausalConv1d, TCNEncoder, TemporalBlock

__all__ = [
    "NewsSignalEncoder",
    "NewsLLMExtractor",
    "NewsAnalysis",
    "TCNEncoder",
    "TemporalBlock",
    "CausalConv1d",
    "AsymmetricCrossAttention",
    "GatedResidualFusion",
    "ReturnPredictor",
]
