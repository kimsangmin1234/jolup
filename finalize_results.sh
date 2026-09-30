#!/bin/bash
# 파이프라인이 끝나면 결과와 로그를 모아 깃에 올린다.
#
# run_on_server.sh 가 끝날 때까지 기다렸다가 실행된다. 로그·지표·중간
# 산출물을 experiments/<TAG>/ 에 모으고 RESULTS.md 요약을 만든 뒤 커밋한다.
# 푸시가 실패하면(자격증명 없음 등) 10분마다 최대 48시간 재시도하므로,
# 나중에 토큰을 설정해도 자동으로 올라간다.
#
#   nohup bash finalize_results.sh > finalize.log 2>&1 &

set -u
cd "$(dirname "$0")" || exit 1
TAG="${TAG:-fnspid-server}"
WORK=".work"
OUT="experiments/$TAG"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

log() { echo "[$(date '+%F %T')] $*"; }

# ------------------------------------------------ 파이프라인 종료 대기
log "run_on_server.sh 종료 대기"
while pgrep -f "bash run_on_server.sh" >/dev/null; do
    sleep 300
done
log "파이프라인 종료 확인"

# ------------------------------------------------------- 산출물 수집
mkdir -p "$OUT/logs"
cp -f run.log "$OUT/logs/run.log" 2>/dev/null
[ -f "$WORK/crawl.log" ] && tail -c 5000000 "$WORK/crawl.log" > "$OUT/logs/crawl.log"
cp -f "$WORK"/batch/*_batches.json "$OUT/logs/" 2>/dev/null

[ -s "$WORK/batch/chat_results.jsonl" ] && gzip -c "$WORK/batch/chat_results.jsonl" > "$OUT/chat_results.jsonl.gz"
[ -s "$WORK/times.jsonl" ] && gzip -c "$WORK/times.jsonl" > data/fnspid/publish_times.jsonl.gz

# 임베딩 캐시는 크므로 깃허브 100MB 한도에 맞춰 조각낸다.
for name in news_cache news_cache_timed; do
    src="$WORK/$name.jsonl"
    [ -s "$src" ] || continue
    rm -f "$OUT/$name".part*.jsonl.gz "$OUT/$name.jsonl.gz"
    split -l 12000 -d -a 2 "$src" "$WORK/${name}_part_"
    for part in "$WORK/${name}_part_"*; do
        idx="${part##*_part_}"
        gzip -c "$part" > "$OUT/$name.part$idx.jsonl.gz"
        rm -f "$part"
    done
done

# ------------------------------------------------------- 결과 요약
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
lines = ["# 학습 결과", ""]
metrics_path = out / "metrics.json"
if not metrics_path.exists():
    lines += ["학습 지표 파일이 없습니다. `logs/run.log` 에서 실패 원인을 확인하세요."]
else:
    m = json.loads(metrics_path.read_text())
    t = m.get("test", {})
    lines += [
        f"- 실행: {m.get('started_at')} ~ {m.get('finished_at')} (커밋 `{m.get('git_commit')}`)",
        f"- 데이터: 학습 {m['dataset_size']['train']:,} / 검증 {m['dataset_size']['valid']:,} / 평가 {m['dataset_size']['test']:,}건",
        f"- 분할: 학습 ~{m['split']['train_end']}, 검증 ~{m['split']['valid_end']}",
        f"- 최고 검증 에폭: {m.get('best_epoch')}",
        "",
        "## 평가(test) 지표",
        "",
        "| 지표 | 값 |",
        "|---|---|",
    ]
    for key, label in [("mse", "MSE"), ("mae", "MAE"), ("dir_acc", "방향 적중률"),
                       ("corr", "상관계수"), ("mse_vs_zero", "MSE / 0예측 MSE")]:
        if key in t:
            lines.append(f"| {label} | {t[key]:.6f} |")
    lines += ["", "`MSE / 0예측 MSE` 가 1보다 작아야 '항상 0을 예측'하는 기준선보다 낫다.", "",
              "## 에폭별 검증 지표", "", "| 에폭 | train MSE | valid MSE | valid 방향 |", "|---|---|---|---|"]
    for e in m.get("epochs_log", []):
        lines.append(f"| {e['epoch']} | {e['train']['mse']:.6f} | {e['valid']['mse']:.6f} | {e['valid']['dir_acc']:.3f} |")
(out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:20]))
PY

# ------------------------------------------------------- 커밋·푸시
git add -Af "$OUT" data/fnspid/publish_times.jsonl.gz
git -c user.email=sunshine31885@gmail.com -c user.name=Claude commit -q -m "[$TAG] 학습 결과와 전체 로그

$(head -30 "$OUT/RESULTS.md")

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "커밋 완료: $(git log --oneline -1)"

for attempt in $(seq 1 288); do
    if git push -u origin "$BRANCH" 2>&1; then
        log "푸시 완료"
        exit 0
    fi
    log "푸시 실패 ($attempt/288) — 10분 후 재시도"
    sleep 600
done
log "48시간 동안 푸시하지 못했습니다. 로컬 커밋은 남아 있습니다."
