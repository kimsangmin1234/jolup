#!/bin/bash
# 크롤링 진행분을 주기적으로 깃에 올린다.
#
# 클라우드 컨테이너는 유휴 시 회수되고 그 안의 파일은 함께 사라진다.
# 장시간 크롤링 결과를 잃지 않도록 일정 주기마다 커밋·푸시한다.
#
#   ./autocommit_progress.sh <감시할 파일> <깃에 둘 경로> [주기(초)]
#
# 예: ./autocommit_progress.sh /tmp/.../times.jsonl data/fnspid/publish_times.jsonl.gz 600

set -u
SRC="${1:?감시할 파일 경로}"
DEST="${2:?저장소 내 경로}"
INTERVAL="${3:-600}"
BRANCH="$(git -C /home/user/jolup rev-parse --abbrev-ref HEAD)"
LAST=0

cd /home/user/jolup || exit 1

while true; do
    sleep "$INTERVAL"
    [ -f "$SRC" ] || continue

    COUNT=$(wc -l < "$SRC" 2>/dev/null || echo 0)
    # 새로 쌓인 게 없으면 커밋하지 않는다.
    [ "$COUNT" -gt "$LAST" ] || continue

    mkdir -p "$(dirname "$DEST")"
    if [[ "$DEST" == *.gz ]]; then
        gzip -c "$SRC" > "$DEST"
    else
        cp "$SRC" "$DEST"
    fi

    git add -f "$DEST" >/dev/null 2>&1
    git -c user.email=sunshine31885@gmail.com -c user.name=Claude \
        commit -q -m "크롤링 진행분 갱신: ${COUNT}건

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QjGCKib6MQ2SozLNAjfniK" >/dev/null 2>&1

    for delay in 2 4 8 16; do
        git push -u origin "$BRANCH" >/dev/null 2>&1 && break
        sleep "$delay"
    done

    echo "$(date '+%F %T')  커밋·푸시 ${COUNT}건"
    LAST="$COUNT"
done
