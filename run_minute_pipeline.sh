#!/bin/bash
# 분봉 수집이 끝나면 라벨을 다시 만들고 모델 비교 실험을 돌린다.
#
#   ./run_minute_pipeline.sh
#
# 1) 깃에서 분봉(data/fnspid/minute/)을 받는다
# 2) apply_minute_labels.py: 장중 뉴스 라벨을 '발행 다음 분봉 시가 → 종가'로 교체
#    (분봉이 없는 종목·날짜의 장중 뉴스는 깨끗한 라벨을 만들 수 없어 제외)
# 3) run_suite.py 를 작업자 4개로 병렬 실행, 결과는 experiments/suite_minute/
set -u
cd "$(dirname "$0")"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git pull -q --no-rebase origin "$BRANCH"

python3 apply_minute_labels.py --labels data/fnspid/labels_open_close.jsonl.gz \
    --minute-dir data/fnspid/minute --out data/fnspid/labels_minute.jsonl.gz \
    2>&1 | tee experiments/minute_labels.log
git add -f data/fnspid/labels_minute.jsonl.gz experiments/minute_labels.log
git -c user.email=sunshine31885@gmail.com -c user.name=Claude commit -q -m "분봉 기반 라벨 생성: 장중 뉴스 발행 직후 → 종가

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK"
git push -q origin "$BRANCH"

OUT=experiments/suite_minute
mkdir -p "$OUT"
run() {
    python3 run_suite.py --labels data/fnspid/labels_minute.jsonl.gz --out "$OUT" \
        --worker "$1" --threads 1 --commit --groups "$2" --only "$3" > "$OUT/worker_$1.out" 2>&1 &
}
run A linear,gbm,nn L1,L2,L3,L4,L5,G1,G2,N2,N3,N4,N5
run B nn N6,N7,N8,N9,N10,N11
run C nn N1
run D nn N0
wait
echo "완료"
