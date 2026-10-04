#!/bin/bash
# 발행 시각이 확인된 레코드 중 요약·감성이 빠진 것을 Batch API 로 처리하고
# 진행분을 깃에 올린다. 발행 시각을 모르는 뉴스는 학습에 쓰지 않으므로 처리하지 않는다.
#
#   OPENAI_API_KEY=... ./run_remaining.sh
#
# 1) recover_batches.py 로 기존 진행분(깃 + OpenAI 배치 출력)을 복원
# 2) 요약·감성 → 임베딩 순으로 batch_enrich.py run
# 3) 새로 생긴 결과만 experiments/fnspid-full/ 에 나눠 저장하고 커밋
#
# 컨테이너가 중간에 회수돼도 같은 명령으로 다시 실행하면 이어서 처리한다.

set -u
cd "$(dirname "$0")"
WORK=.work
OUT=experiments/fnspid-full
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
mkdir -p "$WORK/batch" "$OUT"
LOG="$OUT/run.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

commit_push() {
    git add -Af "$OUT"
    git -c user.email=sunshine31885@gmail.com -c user.name=Claude commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK" 2>/dev/null || return 0
    for d in 2 4 8 16; do
        git push -q -u origin "$BRANCH" 2>/dev/null && return 0
        sleep "$d"
    done
    log "푸시 실패 — 로컬 커밋은 남아 있습니다."
}

# 새로 처리된 요약·감성만 깃에 둔다(기존분은 다른 실험 폴더에 있다).
save_chat() {
    python3 - "$WORK/batch/chat_results.jsonl" "$OUT/chat_results.jsonl.gz" <<'PY'
import glob, gzip, json, sys
old = set()
for p in glob.glob("experiments/*/chat_results.jsonl.gz"):
    if "fnspid-full" in p:
        continue
    for line in gzip.open(p, "rt", encoding="utf-8"):
        old.add(json.loads(line)["news_id"])
n = 0
with gzip.open(sys.argv[2], "wt", encoding="utf-8") as out:
    for line in open(sys.argv[1], encoding="utf-8"):
        if json.loads(line)["news_id"] not in old:
            out.write(line); n += 1
print(n)
PY
}

# 백그라운드: 30분마다 요약·감성 진행분 커밋
(
    LAST=0
    while true; do
        sleep 1800
        [ -f "$WORK/chat.done" ] && exit 0
        N=$(save_chat)
        if [ "$N" -gt "$LAST" ]; then
            commit_push "[fnspid-full] 추가 요약·감성 ${N}건"
            LAST=$N
        fi
    done
) &
SAVER=$!
trap 'kill $SAVER 2>/dev/null' EXIT

log "기존 진행분 복원"
python3 recover_batches.py --work "$WORK/batch" --out "$WORK/news_cache.jsonl" >> "$LOG" 2>&1 || exit 1
BASE_EMB=$(wc -l < "$WORK/news_cache.jsonl")

# 발행 시각을 확인한 레코드만 남긴다(크롤링 결과 + 기존 캐시에 기록된 시각).
python3 - "$WORK/records_article.jsonl" <<'PY'
import glob, gzip, json, sys
def lines(p):
    f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
    return (json.loads(l) for l in f if l.strip())
timed = set()
for p in glob.glob("data/fnspid/publish_times*.jsonl*"):
    timed |= {r["url"] for r in lines(p) if r.get("published_et")}
for p in glob.glob("experiments/*/news_cache_timed*.jsonl.gz"):
    timed |= {r["url"] for r in lines(p) if r.get("published_et")}
n = 0
with open(sys.argv[1], "w", encoding="utf-8") as out:
    for p in sorted(glob.glob("data/fnspid/articles/records_article.part*.jsonl.gz")):
        for r in lines(p):
            if r.get("url") in timed:
                out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"발행 시각 확인 레코드 {n}건", file=sys.stderr)
PY
TOTAL=$(wc -l < "$WORK/records_article.jsonl")

log "요약·감성 (전체 $TOTAL건)"
for attempt in 1 2 3 4 5; do
    python3 batch_enrich.py run --stage chat --input "$WORK/records_article.jsonl" \
        --work "$WORK/batch" --poll 90 >> "$LOG" 2>&1 && break
    log "  요약·감성 단계 비정상 종료 — 재시도 $attempt/5"
    sleep 300
done
touch "$WORK/chat.done"
DONE=$(wc -l < "$WORK/batch/chat_results.jsonl")
N=$(save_chat)
commit_push "[fnspid-full] 요약·감성 완료: 누적 ${DONE}/${TOTAL}건 (추가 ${N}건)"
if [ "$DONE" -lt $((TOTAL * 95 / 100)) ]; then
    log "요약·감성이 95% 미만(${DONE}/${TOTAL})이라 중단"
    exit 2
fi

log "임베딩"
python3 batch_enrich.py run --stage embed --work "$WORK/batch" \
    --out "$WORK/news_cache.jsonl" --poll 90 >> "$LOG" 2>&1 || exit 3

# 새로 붙은 임베딩만 90MB 단위로 나눠 저장(깃허브 파일 한도 100MB)
rm -f "$OUT"/news_cache_new.part*.jsonl.gz
tail -n +"$((BASE_EMB + 1))" "$WORK/news_cache.jsonl" \
    | split -l 9000 -d -a 2 --additional-suffix=.jsonl --filter='gzip > $FILE.gz' - "$OUT/news_cache_new.part"
FINAL=$(wc -l < "$WORK/news_cache.jsonl")
log "완료: 임베딩 누적 ${FINAL}/${TOTAL}건"
commit_push "[fnspid-full] 감성·임베딩 전량 완료: ${FINAL}/${TOTAL}건"
