# 뉴스 이벤트 기반 주식 가격 예측 모델

논문 **"뉴스 이벤트 기반 주식 가격 예측 모델"**(김상민, 성우제, 유재수 / 충북대학교 정보통신공학부)에서 제안한 뉴스 주도형 멀티모달 예측 모델의 구현체입니다.

논문의 **모듈 하나당 파일 하나** 구조로 작성했습니다.

## 모델 구조

```
              뉴스 본문 (이벤트 발생)
                      │
        ┌─────────────┴──────────────┐
        │  [1] LLM 기반 주 신호 생성   │       ┌──────────────────────────┐
        │   GPT-4o-mini               │       │ [2] TCN 기반 보조 벡터 생성│
        │    ├ 뉴스 요약 → 임베딩 1536 │       │   9종 기술적 지표 (30일)   │
        │    └ 감성 점수 (-1 ~ +1)     │       │        ↓ TCN (인과 합성곱) │
        │   결합 1537 → 완전연결층      │       │   보조 표현 (30, 512)     │
        └─────────────┬──────────────┘       └────────────┬─────────────┘
                주 신호 (512)                         보조 표현 (30, 512)
                      │                                   │
                      └──────────┬────────────────────────┘
                                 ▼
                  [3] 비대칭 교차 어텐션  (뉴스 = Query, 시장 = Key/Value)
                                 ▼
                        보조 문맥 벡터 (512)
                                 ▼
                  [4] 게이트 기반 잔차 결합  h = h_news + g ⊙ c_aux
                                 ▼
                        통합 표현 벡터 (512)
                                 ▼
                  [5] 등락률 예측 MLP  512 → 256 → 64 → 1
                                 ▼
                             등락률
```

## 파일 구성

| 논문 절 | 모듈 | 파일 | 핵심 클래스 |
|---|---|---|---|
| Ⅱ.1 | LLM 기반 주 신호 생성 | `modules/news_encoder.py` | `NewsLLMExtractor`, `NewsSignalEncoder` |
| Ⅱ.2 | TCN 기반 보조 벡터 생성 | `modules/tcn_encoder.py` | `TCNEncoder`, `CausalConv1d` |
| Ⅱ.3 | 비대칭 교차 어텐션 | `modules/cross_attention.py` | `AsymmetricCrossAttention` |
| Ⅱ.4 | 게이트 기반 잔차 결합 | `modules/gated_fusion.py` | `GatedResidualFusion` |
| Ⅱ.5 | 등락률 예측 | `modules/return_predictor.py` | `ReturnPredictor` |

그 외 지원 파일:

| 파일 | 역할 |
|---|---|
| `config.py` | 차원·하이퍼파라미터 설정 (모두 논문 명시값) |
| `model.py` | 5개 모듈을 결합한 전체 모델 `NewsDrivenStockPredictor` |
| `data/technical_indicators.py` | 9종 기술적 지표 계산 + 학습 구간 전용 최소-최대 정규화 |
| `data/dataset.py` | 뉴스·지표·라벨을 묶는 `NewsStockDataset`, 시간 순 분할 |
| `preprocess.py` | GPT-4o-mini / text-embedding-3-small 전처리 캐시 생성 |
| `train.py` | 학습·평가 루프 (MSE + 방향성 적중률) |
| `tests/test_modules.py` | 모듈별 형태·불변식 검증 (API 키 불필요) |

## 논문 명시 사양 대응

| 항목 | 논문 값 | 구현 위치 |
|---|---|---|
| 요약·감성 추출 LLM | GPT-4o-mini | `NewsEncoderConfig.llm_model` |
| 감성 점수 범위 | -1 ~ +1 실수 1개 | `news_encoder._clip_sentiment` |
| 의미 임베딩 | text-embedding-3-small, 1536차원 | `NewsEncoderConfig.embedding_model` |
| 결합 벡터 | 1536 + 1 = 1537차원 | `NewsEncoderConfig.fusion_input_dim` |
| 주 신호 벡터 | 완전연결층 통과, 512차원 | `NewsSignalEncoder.projection` |
| 기술적 지표 | 9종 (종가/거래량/MA5/MA20/RSI/MACD/BB 상·하한/ATR) | `config.TECHNICAL_INDICATORS` |
| 참조 기간 | 과거 30일 | `TCNConfig.lookback` |
| 정규화 | 학습 구간 통계만 사용한 최소-최대 | `MinMaxScaler`, `fit_scaler_on_train` |
| 데이터 누수 방지 | 합성곱 전 시계열 앞쪽 0 패딩 | `CausalConv1d` |
| 보조 표현 | 30개 시점 × 512차원 | `TCNEncoder` 출력 |
| 교차 어텐션 | 뉴스=질의, 지표=키/값, 내적 → softmax(30) → 가중 평균 | `AsymmetricCrossAttention` |
| 게이트 | [주 신호;보조 문맥] → 차원별 0~1 | `GatedResidualFusion.gate_net` |
| 잔차 결합 | 조절된 보조 문맥을 주 신호에 가산 | `GatedResidualFusion.forward` |
| 예측 MLP | 512→256→64→1, 은닉층 ReLU, 출력 활성함수 없음 | `PredictorConfig.hidden_dims` |

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 검증 (API 키 불필요)

임의 텐서로 모든 모듈의 형태와 불변식(어텐션 가중치 합=1, 인과성, 게이트 범위 등)을 확인합니다.

```bash
python tests/test_modules.py
```

### 2. LLM 전처리 캐시 생성

뉴스 원문 JSONL을 준비합니다.

```json
{"news_id": "n001", "ticker": "005930", "date": "2026-03-04", "article": "뉴스 본문 ...", "label": 0.0132}
```

```bash
OPENAI_API_KEY=... python preprocess.py --input news_raw.jsonl --output data/news_cache.jsonl
```

요약·감성·임베딩 추출은 비용이 크므로 학습 루프 밖에서 한 번만 수행하고 캐시합니다. 이미 처리한 `news_id`는 건너뛰므로 중단 후 재실행해도 안전합니다.

### 2-1. FNSPID 데이터셋으로 준비 (권장)

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) (KDD 2024)는 S&P500 4,775개 종목의 뉴스 1,570만 건과 주가 2,970만 건을 담고 있으며, GPT 감성 점수와 요약문이 이미 포함되어 있습니다.

```bash
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
unzip full_history.zip

python prepare_fnspid.py \
    --news nasdaq_exteral_data.csv \
    --price-dir full_history \
    --tickers AAPL,MSFT,NVDA,AMZN,GOOGL \
    --start 2015-01-01 --end 2023-12-31 \
    --out-records data/news_cache.jsonl \
    --out-indicators data/indicators.npz \
    --embedding openai
```

`prepare_fnspid.py`가 처리하는 내용:

- 5GB 뉴스 CSV를 청크 없이 한 줄씩 스트리밍하여 대상 종목·기간만 추출
- FNSPID의 `Sentiment_gpt`(1~5 척도)를 논문의 **-1 ~ +1** 범위로 선형 변환 (열이 없으면 GPT-4o-mini 호출)
- `Textrank_summary`를 요약문으로 사용하여 임베딩 생성
- 주가에서 9종 지표 계산 + **다음 거래일 등락률**을 라벨로 산출

이 단계를 거치면 3번을 건너뛰고 바로 4번(학습)으로 갑니다.

`--embedding local` 옵션은 외부 API를 쓸 수 없는 환경용 대안입니다. 다만 차원이 논문의 1536과 다르므로 `NewsEncoderConfig.embedding_dim`을 함께 맞춰야 하며, 논문 사양에서 벗어납니다.

### 3. 기술적 지표 준비

종목별 OHLCV로 9종 지표를 계산하여 npz로 저장합니다.

```python
import numpy as np
from data.technical_indicators import compute_indicators

matrix = compute_indicators(high, low, close, volume)  # (T, 9)
np.savez("data/indicators.npz",
         **{"005930__values": matrix, "005930__dates": np.array(dates)})
```

### 4. 학습

```bash
python train.py --records data/news_cache.jsonl \
                --indicators data/indicators.npz \
                --train-end 2025-06-30 \
                --valid-end 2025-09-30
```

분할은 **시간 순**입니다. 무작위 분할은 미래 정보 누수를 일으키므로 사용하지 않습니다. 정규화 통계(min/max)도 학습 구간에서만 산출합니다.

### 5. 개별 모듈 사용

각 모듈은 독립적으로 동작합니다.

```python
import torch
from modules import AsymmetricCrossAttention

attention = AsymmetricCrossAttention()
primary = torch.randn(8, 512)          # 뉴스 주 신호
auxiliary = torch.randn(8, 30, 512)    # 30일 보조 표현

context, weights = attention(primary, auxiliary)
print(context.shape, weights.shape)    # (8, 512) (8, 30)
print(weights.sum(dim=-1))             # 모두 1.0
```

## 구현상의 판단

논문 본문이 명시하지 않은 부분은 다음과 같이 정했으며, 모두 설정으로 변경할 수 있습니다.

- **어텐션 스케일링** — 논문은 관련도를 "내적"으로만 기술합니다. 512차원 내적은 값의 크기가 커져 softmax가 포화되기 쉬우므로 `1/sqrt(d)` 스케일링을 기본값으로 두었습니다. 논문 표기를 문자 그대로 재현하려면 `CrossAttentionConfig.scaled = False`로 설정하십시오.
- **TCN 세부 구조** — 논문은 인과 패딩과 출력 차원(30×512)만 명시합니다. 커널 3, dilation 1·2·4·8의 잔차 블록 4단으로 구성하여 수용 영역이 30일을 덮도록 했습니다.
- **손실 함수** — 등락률이 연속값이므로 MSE 회귀로 학습하고, 참고 지표로 방향성 적중률을 함께 보고합니다.
- **정규화 계층** — 학습 안정화를 위해 각 모듈 출력에 LayerNorm을 두었습니다. 시점별로 독립 적용되므로 TCN의 인과성은 유지됩니다(`test_tcn_shape_and_causality`에서 검증).

## 참고 문헌

1. Lopez-Lira, A. and Tang, Y., "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models," SSRN Electronic Journal, 2023.
2. Liu, C., Arulappan, A., Naha, R., Mahanti, A., Kamruzzaman, J., & Ra, I. H. (2024). Large language models and sentiment analysis in financial markets: a review, datasets, and case study. IEEE Access, 12, 134041-134061.
