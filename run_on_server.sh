#!/bin/bash
# 서버에서 전체 파이프라인을 끝까지 돌린다.
#
# 감성·임베딩 생성(Batch API), 발행 시각 크롤링, 학습을 순서대로 수행하고
# 진행분을 주기적으로 깃에 올린다. 각 단계가 재실행 안전하므로 중간에
# 끊겨도 같은 명령으로 다시 돌리면 이어서 진행된다.
#
#   export OPENAI_API_KEY=sk-...
#   nohup bash run_on_server.sh > run.log 2>&1 &
#
# 환경 변수
#   OPENAI_API_KEY   필수
#   SKIP_CRAWL=1     발행 시각 크롤링을 건너뛴다
#   SKIP_TRAIN=1     학습을 건너뛴다
#   PUSH=0           깃 푸시를 하지 않는다(자격증명이 없는 서버용)

set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
WORK="$ROOT/.work"
TAG="${TAG:-fnspid-server}"
LOGDIR="experiments/$TAG"
PUSH="${PUSH:-1}"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"

mkdir -p "$WORK" "$LOGDIR" checkpoints

log() { echo "[$(date '+%F %T')] $*"; }

if [ -z "${OPENAI_API_KEY:-}" ]; then
    log "OPENAI_API_KEY 가 설정되지 않았습니다."
    exit 1
fi

commit_push() {
    [ "$PUSH" = "1" ] || return 0
    git add -Af "$LOGDIR" data/fnspid 2>/dev/null
    git -c user.email=sunshine31885@gmail.com -c user.name=Claude \
        commit -q -m "$1" 2>/dev/null || return 0
    for d in 2 4 8 16; do
        git push -u origin "$BRANCH" >/dev/null 2>&1 && return 0
        sleep "$d"
    done
    log "푸시 실패 — 로컬 커밋은 남아 있습니다."
}

# ---------------------------------------------------------------- 의존성
log "의존성 확인"
python3 -c "import numpy, openai" 2>/dev/null || {
    log "  numpy, openai 설치"
    pip3 install --quiet --user numpy openai || pip3 install --quiet numpy openai
}

# ------------------------------------------------------- 본문 레코드 복원
RECORDS="$WORK/records_article.jsonl"
if [ ! -s "$RECORDS" ]; then
    log "본문 레코드 복원 (깃 조각 합치기)"
    cat data/fnspid/articles/records_article.part*.jsonl.gz | gunzip -c > "$RECORDS"
fi
log "  레코드 $(wc -l < "$RECORDS")건"

# --------------------------------------------- 이전 진행분 이어받기 (있으면)
if [ ! -s "$WORK/batch/chat_results.jsonl" ] && [ -f "$LOGDIR/chat_results.jsonl.gz" ]; then
    log "이전 요약·감성 진행분 복원"
    mkdir -p "$WORK/batch"
    gunzip -c "$LOGDIR/chat_results.jsonl.gz" > "$WORK/batch/chat_results.jsonl"
    log "  $(wc -l < "$WORK/batch/chat_results.jsonl")건"
fi

# ------------------------------------------------ 발행 시각 크롤링 (병렬)
if [ "${SKIP_CRAWL:-0}" != "1" ]; then
    [ -s "$WORK/urls.txt" ] || gunzip -c data/fnspid/urls_to_crawl.txt.gz > "$WORK/urls.txt"
    [ -s "$WORK/times.jsonl" ] || {
        [ -f data/fnspid/publish_times.jsonl.gz ] && \
            gunzip -c data/fnspid/publish_times.jsonl.gz > "$WORK/times.jsonl"
    }
    log "발행 시각 크롤링 시작 (백그라운드)"
    nohup python3 crawl_publish_time.py --urls "$WORK/urls.txt" --out "$WORK/times.jsonl" \
        --delay 2.0 --workers 8 --shuffle >> "$WORK/crawl.log" 2>&1 &
    CRAWL_PID=$!
fi

# ------------------------------------------------------ 1단계: 요약 + 감성
log "1단계 요약·감성 (Batch API)"
python3 batch_enrich.py run --stage chat --input "$RECORDS" --work "$WORK/batch" --poll 90
gzip -c "$WORK/batch/chat_results.jsonl" > "$LOGDIR/chat_results.jsonl.gz"
commit_push "[$TAG] 요약·감성 $(wc -l < "$WORK/batch/chat_results.jsonl")건"

# -------------------------------------------------------- 2단계: 임베딩
log "2단계 의미 임베딩 (Batch API)"
CACHE="$WORK/news_cache.jsonl"
python3 batch_enrich.py run --stage embed --work "$WORK/batch" --out "$CACHE" --poll 90
gzip -c "$CACHE" > "$LOGDIR/news_cache.jsonl.gz"
commit_push "[$TAG] 임베딩 $(wc -l < "$CACHE")건"

# ---------------------------------------- 3단계: 발행 시각 기반 재라벨링
if [ "${SKIP_CRAWL:-0}" != "1" ]; then
    log "크롤링 완료 대기"
    wait "${CRAWL_PID:-0}" 2>/dev/null
    gzip -c "$WORK/times.jsonl" > data/fnspid/publish_times.jsonl.gz
    commit_push "[$TAG] 발행 시각 $(wc -l < "$WORK/times.jsonl")건"

    if [ ! -d "$WORK/price/full_history" ]; then
        log "주가 데이터 내려받기"
        curl -sSL -o "$WORK/price.zip" \
            "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"
        unzip -q -o "$WORK/price.zip" -d "$WORK/price"
    fi

    log "3단계 발행 시각 기반 재라벨링"
    python3 apply_publish_time.py --records "$CACHE" --times "$WORK/times.jsonl" \
        --price-dir "$WORK/price/full_history" --price-start 2020-07-06 \
        --price-alias GOOGL=GOOG --out "$WORK/news_cache_timed.jsonl"
    CACHE="$WORK/news_cache_timed.jsonl"
    gzip -c "$CACHE" > "$LOGDIR/news_cache_timed.jsonl.gz"
    commit_push "[$TAG] 시각 기반 재라벨링 $(wc -l < "$CACHE")건"
fi

# ------------------------------------------------------------ 4단계: 학습
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    python3 -c "import torch" 2>/dev/null || {
        log "torch 설치"
        pip3 install --quiet --user torch || pip3 install --quiet torch
    }
    log "4단계 학습"
    python3 train.py --records "$CACHE" --indicators data/fnspid/indicators.npz \
        --train-end 2022-12-31 --valid-end 2023-06-30 \
        --checkpoint "checkpoints/${TAG}.pt" --scaler-out "$LOGDIR/scaler.json" \
        --log-dir "$LOGDIR" --tag "$TAG"
    commit_push "[$TAG] 학습 완료"
fi

log "전체 완료"
