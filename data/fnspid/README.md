# FNSPID 파생 데이터

중간 산출물을 포함해 전부 깃으로 관리한다. 클라우드 컨테이너는 유휴 시
회수되므로, 커밋되지 않은 파일은 세션과 함께 사라진다.

## 파일

| 파일 | 내용 |
|---|---|
| `articles/records_article.part*.jsonl.gz` | **뉴스 본문 포함** 레코드 82,903건 (6조각) |
| `records_next_day.jsonl.gz` | 요약문 기반, 다음날 라벨 82,903건 |
| `records_same_day.jsonl.gz` | 요약문 기반, 당일 라벨 82,913건 |
| `indicators.npz` | 종목별 (878, 9) 지표 행렬 + 날짜 인덱스 |
| `ticker_stats.csv` | 종목별 뉴스 건수와 라벨 분포 |
| `urls_to_crawl.txt.gz` | 발행 시각 복구용 기사 URL 72,659건 |
| `publish_times.jsonl.gz` | 복구한 발행 시각 (크롤링 중 자동 갱신) |

본문 레코드는 깃허브 단일 파일 100MB 한도 때문에 6조각으로 나눴다.
합치려면:

```bash
zcat data/fnspid/articles/records_article.part*.jsonl.gz > records_article.jsonl
```

## 레코드 형식

```json
{"news_id": "NVDA-2023-05-25-0", "ticker": "NVDA", "date": "2023-05-25",
 "anchor": "2023-05-25", "url": "https://...", "text": "뉴스 본문 ...",
 "label": 0.0243}
```

`enrich_records.py` 를 거치면 `sentiment`(−1~+1)와 `embedding`(1536차원)이 붙는다.

본문 레코드는 논문 사양에 맞는 쪽이다. 논문은 GPT-4o-mini 가 **뉴스 본문**을
요약한다고 기술하므로, FNSPID 가 미리 만든 요약문(`Textrank_summary`)을 쓰는
`records_next_day` / `records_same_day` 보다 충실하다.

## 구성 과정

`docs/DATASET.md` 참고.
