#!/bin/bash
# 감성·임베딩 생성부터 학습까지 한 번에 돌리고, 결과를 전부 깃에 남긴다.
#
# 클라우드 컨테이너는 유휴 시 회수되므로 중간 산출물을 주기적으로 커밋한다.
# 각 단계는 재실행 안전하다. 끊겨도 같은 명령으로 이어서 돌리면 된다.
#
#   ./run_experiment.sh [실험이름]

set -u
cd /home/user/jolup || exit 1

TAG="${1:-fnspid-nextday}"
SCRATCH="/tmp/claude-0/-home-user-jolup/ca9121ff-be66-518e-bd9b-dfe95c4dbc18/scratchpad"
CACHE="$SCRATCH/news_cache.jsonl"
RECORDS="$SCRATCH/records_article.jsonl"
LOGDIR="experiments/$TAG"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

mkdir -p "$LOGDIR" checkpoints
exec > >(tee -a "$LOGDIR/pipeline.log") 2>&1

commit_push() {
    git add -Af "$LOGDIR" checkpoints 2>/dev/null
    git -c user.email=sunshine31885@gmail.com -c user.name=Claude \
        commit -q -m "$1

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK" 2>/dev/null
    for d in 2 4 8 16; do
        git push -u origin "$BRANCH" >/dev/null 2>&1 && return 0
        sleep "$d"
    done
}

echo "=== [$TAG] 시작 $(date '+%F %T') ==="

# 0) 본문 레코드 복원 (깃의 조각을 합친다)
if [ ! -f "$RECORDS" ]; then
    echo "[0/3] 본문 레코드 복원"
    zcat data/fnspid/articles/records_article.part*.jsonl.gz > "$RECORDS"
fi
echo "  레코드 $(wc -l < "$RECORDS")건"

# 1) 감성 + 의미 임베딩 (재실행 시 처리분은 건너뛴다)
echo "[1/3] 감성·임베딩 생성 $(date '+%T')"
for attempt in 1 2 3; do
    python3 enrich_records.py --input "$RECORDS" --output "$CACHE" \
        --workers 10 --language same 2>&1 | grep -vE "HTTP Request"
    DONE=$(wc -l < "$CACHE" 2>/dev/null || echo 0)
    TOTAL=$(wc -l < "$RECORDS")
    echo "  시도 $attempt: $DONE / $TOTAL"
    gzip -c "$CACHE" > "$LOGDIR/news_cache.jsonl.gz"
    commit_push "[$TAG] 감성·임베딩 ${DONE}/${TOTAL}건"
    [ "$DONE" -ge "$TOTAL" ] && break
done

# 2) 학습
echo "[2/3] 학습 $(date '+%T')"
python3 train.py --records "$CACHE" --indicators data/fnspid/indicators.npz \
    --train-end 2022-12-31 --valid-end 2023-06-30 \
    --checkpoint "checkpoints/${TAG}.pt" --scaler-out "$LOGDIR/scaler.json" \
    --log-dir "$LOGDIR" --tag "$TAG"

# 3) 결과 커밋
echo "[3/3] 결과 저장 $(date '+%T')"
commit_push "[$TAG] 학습 완료"
echo "=== [$TAG] 종료 $(date '+%F %T') ==="
