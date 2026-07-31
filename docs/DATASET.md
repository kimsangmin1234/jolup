# 학습 데이터셋 구성

논문 "뉴스 이벤트 기반 주식 가격 예측 모델"의 학습 데이터를 FNSPID에서 구성한 전 과정을 기록한다.

**결과: 22종목 · 82,903건 · 2020-09-01 ~ 2023-12-27**

---

## 1. 원본 데이터

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) (Financial News and Stock Price Integration Dataset, KDD 2024)

| 파일 | 크기 | 내용 |
|---|---|---|
| `Stock_news/nasdaq_exteral_data.csv` | **22GB** | 뉴스 15,549,299건 |
| `Stock_price/full_history.zip` | 563MB | 주가 7,693종목 |

뉴스 CSV 열 구성:

```
Unnamed: 0, Date, Article_title, Stock_symbol, Url, Publisher, Author,
Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
```

> **주의**: 공개 배포판에는 `Sentiment_gpt` 열이 **없다**. GitHub 저장소의 중간 산출물 예시에는 있지만 HuggingFace 배포판에는 포함되지 않았다. 따라서 감성 점수는 전량 GPT-4o-mini로 새로 추출해야 한다.

---

## 2. 파이프라인

```
[1] stream_fnspid_news.py     22GB 뉴스 스트리밍 → 대상 종목만 추출
        │
        ├─ 디스크 저장 없이 curl 파이프로 처리 (세션 가용 디스크 19GB < 22GB)
        └─ 145,195건 / 24종목
        │
[2] prepare_fnspid.py         주가 → 9종 지표 + 등락률 라벨, 뉴스와 정렬
        │
        ├─ 2020-07-06 이후 주가만 사용 (소스 이어붙임 경계 회피)
        ├─ 뉴스는 2020-09-01부터 (지표 워밍업 + 30일 룩백 확보)
        └─ 82,903건 / 22종목
        │
[3] enrich_records.py         GPT-4o-mini 감성 + text-embedding-3-small 임베딩
        │                     ← 현재 여기서 대기 (OPENAI_API_KEY 필요)
[4] train.py                  학습
```

### [1] 뉴스 스트리밍 추출

22GB를 내려받을 디스크가 없어(가용 19GB) 스트림을 파이프로 흘리며 필터링했다. 원본은 디스크에 남지 않는다.

```bash
curl -sSL https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv \
  | python stream_fnspid_news.py \
        --tickers AAPL,MSFT,NVDA,... \
        --start 2012-01-01 --end 2023-12-31 \
        --out news_subset.csv
```

- 기사 본문에 줄바꿈이 있어 `grep`으로 자르면 행이 깨진다. 반드시 csv 파서로 읽는다.
- 처리량 28MB/s, 전체 1회 스캔에 **31분**.
- 스캔 15,549,299행 → 추출 145,195건 (24종목)

### [2] 주가 → 지표 + 라벨

- 9종 기술적 지표: 종가, 거래량, MA5, MA20, RSI(14), MACD(12,26), 볼린저 상·하한(20, 2σ), ATR(14)
- 라벨: **당일 종가 → 다음 거래일 종가 등락률**
- 뉴스 발생일 기준 과거 30일 지표 시계열이 입력

라벨이 존재하는 날짜(다음 거래일이 있는 날)의 뉴스만 남긴다.

### [3]~[4] 감성·임베딩 후 학습

```bash
export OPENAI_API_KEY=sk-...
python enrich_records.py --input data/fnspid/records_raw.jsonl \
                         --output data/news_cache.jsonl
python train.py --records data/news_cache.jsonl \
                --indicators data/fnspid/indicators.npz \
                --train-end 2022-12-31 --valid-end 2023-06-30
```

---

## 3. 원본 데이터에서 발견한 문제와 대응

### 3.1 주가 CSV 열 이름이 문서와 다름

HuggingFace 배포판은 **소문자**에 **날짜 내림차순**이다.

```
date,open,high,low,close,adj close,volume
2023-12-28,194.13,194.66,193.16,193.58,193.58,34014500
```

GitHub 저장소 예시(`Date,Open,High,...`)와 달라 그대로 읽으면 전부 실패한다. → `load_price_csv`에서 열 이름을 소문자로 정규화해 두 형식을 모두 수용.

또한 파일명 대소문자가 섞여 있다 (`AAPL.csv`, `nvda.csv`). → `_find_price_file`에서 세 가지 형태를 모두 시도.

### 3.2 주가에 소스 이어붙임 경계 존재 (중요)

`2020-07-06`에 서로 무관한 종목이 **동시에** 급등락한다.

| 종목 | 2020-07-02 종가 | 2020-07-06 종가 | 변화 |
|---|---|---|---|
| AAPL | 364.11 | 93.46 | **-74.3%** |
| TSLA | 1208.66 | 91.44 | **-92.4%** |
| GE | 6.82 | 43.72 | **+541.0%** |

실제 분할이 아니라 **두 소스를 이어붙인 흔적**이다. 경계 이전은 당시 시세(미래 분할 미반영), 이후는 분할 소급 조정 시세다. AAPL의 4:1 분할(2020-08), TSLA의 5:1·3:1 분할, GE의 1:8 역분할이 경계 이후 구간에만 반영되어 있다.

**검증**: 각 구간 *내부*는 깨끗하다.

| 구간 | AAPL | NVDA | AMD | MSFT | TSLA |
|---|---|---|---|---|---|
| 2012-01 ~ 2020-07-02 (\|r\|>25%) | 0회 | 1회* | 1회* | 0회 | 0회 |
| 2020-07-06 ~ 2023-12 (\|r\|>25%) | 0회 | 0회 | 0회 | 0회 | 0회 |

\* NVDA +30%(2016-11-11), AMD +52%(2016-04-22)는 실제 실적 급등이다.

**대응**: `--price-start 2020-07-06`으로 경계 이후만 사용. 뉴스의 88%가 2019년 이후라 손실이 작다.

**뉴스는 2020-09-01부터** 사용한다. 지표 워밍업(MA20, RSI14, ATR14)과 30일 룩백이 모두 깨끗한 구간 안에 들어오게 하기 위해서다. 결과적으로 **전 레코드(82,903건)가 30일 룩백을 100% 확보**했다.

### 3.3 뉴스 CSV가 단일 정렬이 아님

종목 심볼로 정렬되어 있으나 **정렬 구간이 여러 개**다(여러 소스를 이어붙인 결과). 부분 스캔으로는 어떤 종목도 완전히 얻을 수 없다.

- 전체 스캔(15.5M행): AAPL 9,338건
- 부분 스캔(1.57M행): AAPL 8,865건 ← 앞쪽 정렬 구간만

→ 두 차례 스캔 결과를 `(심볼, 날짜, URL)` 기준으로 중복 제거해 병합.

### 3.4 티커 별칭

`GOOGL` 주가 파일은 2020-04-01에서 끊긴다. 동일 기업의 `GOOG`(Class C)는 2023-12-28까지 있어 `--price-alias GOOGL=GOOG`로 연결했다.

다만 **GOOGL 뉴스 자체가 2020년에서 끊겨**(2018: 298, 2019: 767, 2020: 687건) 대상 기간에는 0건이다. 최종 22종목에 GOOGL은 포함되지 않는다.

---

## 4. 최종 데이터셋

**82,903건 / 22종목 / 2020-09-01 ~ 2023-12-27**

연도별 분포: 2020년 4,384 · 2021년 13,681 · 2022년 24,245 · 2023년 40,593

요약문 길이 중앙값 579자 (`Textrank_summary` 열 사용)

### 종목별 통계 (라벨 변동성 내림차순)

| 종목 | 뉴스 | 라벨 std% | 평균 \|라벨\|% | 최대 \|라벨\|% |
|---|---:|---:|---:|---:|
| GME | 2,969 | 24.23 | 12.28 | 134.8 |
| AMC | 2,964 | 18.35 | 8.62 | 301.2 |
| FCEL | 483 | 8.15 | 5.78 | 54.3 |
| NKLA | 1,436 | 8.03 | 5.77 | 40.8 |
| BLNK | 606 | 7.50 | 5.23 | 48.3 |
| TSLA | 7,927 | 3.85 | 2.86 | 12.2 |
| NVDA | 7,923 | 3.56 | 2.54 | 24.4 |
| AMD | 4,616 | 3.53 | 2.68 | 14.3 |
| BA | 2,517 | 2.79 | 2.05 | 13.7 |
| MU | 2,107 | 2.73 | 2.05 | 10.5 |
| INTC | 4,650 | 2.61 | 1.88 | 11.7 |
| F | 1,162 | 2.37 | 1.52 | 12.2 |
| DIS | 5,632 | 2.19 | 1.52 | 13.6 |
| GE | 2,115 | 2.17 | 1.61 | 10.7 |
| AMZN | 4,242 | 2.07 | 1.56 | 8.3 |
| MSFT | 7,957 | 2.00 | 1.51 | 8.2 |
| CVX | 4,647 | 1.86 | 1.41 | 8.9 |
| GM | 95 | 1.86 | 1.58 | 6.6 |
| AAPL | 7,970 | 1.77 | 1.30 | 8.9 |
| WMT | 5,342 | 1.66 | 1.03 | 11.4 |
| MRK | 2,915 | 1.48 | 1.01 | 9.9 |
| KO | 2,628 | 1.11 | 0.79 | 7.0 |

전체 라벨: 평균 +0.129%, 표준편차 6.41%
극단값: \|라벨\|>10% 2,222건(2.68%) · >20% 750건(0.90%) · >30% 538건(0.65%)

### 저장 파일

| 파일 | 크기 | 내용 |
|---|---|---|
| `data/fnspid/records_raw.jsonl.gz` | 16MB | 레코드 82,903건 (감성·임베딩 미포함) |
| `data/fnspid/indicators.npz` | 1.2MB | 종목별 (878, 9) 지표 행렬 + 날짜 인덱스 |
| `data/fnspid/ticker_stats.csv` | 1.2KB | 위 통계표 |

레코드 형식:

```json
{"news_id": "NVDA-2023-05-25-0", "ticker": "NVDA", "date": "2023-05-25",
 "summary": "...", "label": 0.0243}
```

`enrich_records.py` 실행 후 `sentiment`(−1~+1)와 `embedding`(1536차원)이 추가된다.

---

## 5. 학습 전 검토가 필요한 사항

### 5.1 종목 간 라벨 스케일 차이 (중요)

라벨 표준편차가 KO 1.11%부터 GME 24.23%까지 **22배** 차이 난다. MSE로 그대로 학습하면 손실이 GME·AMC에 지배되어, 나머지 20종목은 사실상 학습되지 않는다.

대응 선택지:

1. **종목별 라벨 표준화** — 학습 구간 통계로 종목별 z-score 정규화. 종목 간 기여를 균등화한다. 평가 시 역변환.
2. **윈저화(winsorizing)** — \|라벨\|을 상위 1% 지점에서 자른다. 538건(0.65%)이 영향받는다.
3. **고변동 종목 제외** — GME·AMC를 빼면 표준편차가 6.41% → 3.03%로 내려간다.
4. **Huber 손실** — 극단값의 영향을 줄인다.

현재 데이터셋은 **아무 처리도 적용하지 않은 원본 상태**다. 어떤 방식을 쓸지는 실험 설계에 따라 결정한다.

### 5.2 FNSPID의 티커 태깅이 느슨함

업계 일반 기사가 여러 종목에 함께 매핑된다. 예를 들어 인텔 Meteor Lake GPU 기사가 AAPL에도 태깅되어 있다.

필요하면 제목·요약에 종목명이나 티커가 포함된 기사만 남기는 필터를 추가할 수 있다.

### 5.3 알고리즘 생성 필러 기사

`"Interesting A Put And Call Options For August 2024"` 같은 옵션·배당 정보 기사가 다수 포함되어 있다. 가격 영향이 없는 정형 기사로, 저변동 대형주에서 비중이 높다.

### 5.4 시간 순 분할 필수

`train.py`는 `--train-end` / `--valid-end`로 시간 순 분할한다. 무작위 분할은 미래 정보 누수를 일으키므로 사용하지 않는다. 정규화 통계(min/max)도 학습 구간에서만 산출한다.

권장 분할 (2023년에 데이터가 몰려 있는 점을 고려):

```
학습   ~2022-12-31   (42,310건)
검증   ~2023-06-30
평가   2023-07-01~
```

---

## 6. 재현 방법

```bash
# 1) 주가
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
unzip full_history.zip

# 2) 뉴스 (디스크 저장 없이 스트리밍 필터, 약 31분)
curl -sSL https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv \
  | python stream_fnspid_news.py \
        --tickers NVDA,TSLA,KO,AAPL,AMD,MSFT,DIS,CVX,WMT,INTC,GE,MRK,MU,BA,AMZN,GME,AMC,JPM,NKLA,GOOGL,F,FCEL,BLNK,GM \
        --start 2012-01-01 --end 2023-12-31 \
        --out news_subset.csv

# 3) 지표 + 라벨 (API 불필요)
python prepare_fnspid.py \
    --news news_subset.csv --price-dir full_history \
    --tickers NVDA,TSLA,KO,AAPL,AMD,MSFT,DIS,CVX,WMT,INTC,GE,MRK,MU,BA,AMZN,GME,AMC,JPM,NKLA,GOOGL,F,FCEL,BLNK,GM \
    --price-alias GOOGL=GOOG \
    --price-start 2020-07-06 --start 2020-09-01 --end 2023-12-31 \
    --max-per-ticker 50000 \
    --out-records data/fnspid/records_raw.jsonl \
    --out-indicators data/fnspid/indicators.npz \
    --embedding none
```

네트워크가 불안정하면 스트리밍이 중간에 끊길 수 있다(실제로 1회 발생). 종목을 나눠 여러 번 실행한 뒤 `(심볼, 날짜, URL)` 기준으로 병합·중복 제거하면 된다.
