"""Nasdaq 기사 URL에서 발행 시각을 복구한다.

FNSPID 배포판은 뉴스 날짜만 남기고 시각을 버렸다(전체의 99.98%가 00:00:00).
당일 등락률을 예측하려면 뉴스가 장 마감 전인지 후인지 알아야 하므로,
기사 URL을 다시 방문해 JSON-LD의 ``datePublished`` 를 복구한다.

    "datePublished": "Tue, 12/19/2023 — 15:10"   (미 동부시간)

주의: nasdaq.com 의 robots.txt 는 ``User-agent: *`` 에 ``Crawl-delay: 30`` 을
명시한다. --delay 기본값은 이를 따르며, 낮추는 것은 사용자의 판단이다.

결과는 JSONL로 누적 저장되고 이미 처리한 URL은 건너뛰므로 재실행이 안전하다.

사용 예::

    python crawl_publish_time.py --urls urls.txt --out times.jsonl \\
        --limit 500 --delay 1.5
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# JSON-LD: "datePublished": "Tue, 12/19/2023 — 15:10"
DATE_RE = re.compile(r'"datePublished"\s*:\s*"([^"]{5,80})"')
# "Tue, 12/19/2023 — 15:10" 또는 유사 변형
PARSE_RE = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})\s*[—\-–]\s*(\d{1,2}):(\d{2})')


def parse_published(raw: str) -> str | None:
    """원문 문자열 → 'YYYY-MM-DD HH:MM' (미 동부시간 기준)."""
    m = PARSE_RE.search(raw)
    if m:
        month, day, year, hour, minute = (int(x) for x in m.groups())
        try:
            return datetime(year, month, day, hour, minute).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return None
    # ISO 형식으로 제공되는 경우도 받아둔다
    m = re.search(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})', raw)
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}" if m else None


def fetch(url: str, timeout: int) -> tuple[str | None, str]:
    """(발행시각 원문, 상태) 반환."""
    request = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # 기사 본문 전체를 읽을 필요 없이 head 영역이면 충분하다.
            head = response.read(400_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return None, f"http_{exc.code}"
    except Exception as exc:                      # 네트워크/타임아웃 등
        return None, type(exc).__name__

    m = DATE_RE.search(head)
    return (m.group(1), "ok") if m else (None, "no_date_field")


def load_done(path: Path, retry: tuple[str, ...] = ()) -> set[str]:
    """이미 처리한 URL. ``retry`` 에 든 상태(예: http_403)로 끝난 것은 다시 시도한다."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "url" not in row:
                    continue
                if row.get("status") in retry and not row.get("published_et"):
                    done.discard(row["url"])
                else:
                    done.add(row["url"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="기사 발행 시각 복구 크롤러")
    parser.add_argument("--urls", required=True, help="URL 목록 (한 줄에 하나)")
    parser.add_argument("--out", required=True, help="결과 JSONL (누적)")
    parser.add_argument("--limit", type=int, default=0, help="이번 실행 최대 건수")
    parser.add_argument("--delay", type=float, default=30.0,
                        help="요청 간 대기 초. robots.txt 의 Crawl-delay 는 30이다.")
    parser.add_argument("--jitter", type=float, default=0.3, help="대기 시간 무작위 비율")
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--shuffle", action="store_true",
                        help="URL 순서를 섞는다(표본 조사용)")
    parser.add_argument("--retry-status", default="",
                        help="이 상태로 끝난 URL을 다시 시도한다. 쉼표 구분 (예: http_403)")
    parser.add_argument("--socks", default="",
                        help="SOCKS5 프록시 host:port. 다른 PC 의 IP 로 나가게 할 때 쓴다 "
                             "(그 PC 에서 ssh -R 1080 서버 로 터널을 열어 둔다)")
    parser.add_argument("--block-streak", type=int, default=20,
                        help="403 이 이만큼 연속되면 IP 차단으로 보고 쉰다")
    parser.add_argument("--block-wait", type=int, default=1800,
                        help="차단 감지 시 대기(초)")
    parser.add_argument("--workers", type=int, default=1,
                        help="동시 요청 수. 총 요청률은 workers/delay 가 된다. "
                             "robots.txt 의 Crawl-delay 를 고려해 정할 것.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.socks:
        # 모든 소켓을 SOCKS5 로 보낸다. rdns=True 로 이름 풀이도 프록시 쪽에서 한다.
        import socket

        import socks
        host, port = args.socks.rsplit(":", 1)
        socks.set_default_proxy(socks.SOCKS5, host, int(port), rdns=True)
        socket.socket = socks.socksocket
        logger.info("SOCKS5 프록시 사용: %s", args.socks)

    if args.urls.endswith(".gz"):
        with gzip.open(args.urls, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        text = Path(args.urls).read_text(encoding="utf-8")
    urls = [u.strip() for u in text.splitlines() if u.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    retry = tuple(x.strip() for x in args.retry_status.split(",") if x.strip())
    done = load_done(out_path, retry)
    pending = [u for u in urls if u not in done]
    if args.shuffle:
        random.seed(42)
        random.shuffle(pending)
    if args.limit:
        pending = pending[: args.limit]

    logger.info("전체 %d / 처리됨 %d / 이번 실행 %d (delay %.1fs)",
                len(urls), len(done), len(pending), args.delay)

    stats: dict[str, int] = {}
    started = time.time()
    todo: queue.Queue = queue.Queue()
    for url in pending:
        todo.put(url)

    lock = threading.Lock()
    processed = [0]
    streak = [0]      # 연속 403 횟수. 사이트가 IP 를 차단하면 전부 403 이 된다.

    def worker(out) -> None:
        while True:
            try:
                url = todo.get_nowait()
            except queue.Empty:
                return

            raw, status = fetch(url, args.timeout)

            # 차단된 상태에서 계속 두드리면 차단만 길어진다. 403 이 연속되면
            # 결과를 기록하지 않고 URL 을 되돌린 뒤 한참 쉰다.
            with lock:
                streak[0] = streak[0] + 1 if status == "http_403" else 0
                blocked = streak[0] >= args.block_streak
            if blocked:
                todo.put(url)
                logger.warning("403 이 %d회 연속 — 차단으로 보고 %d분 대기",
                               streak[0], args.block_wait // 60)
                time.sleep(args.block_wait)
                with lock:
                    streak[0] = 0
                continue

            line = json.dumps({
                "url": url,
                "raw": raw,
                "published_et": parse_published(raw) if raw else None,
                "status": status,
            }, ensure_ascii=False)

            with lock:
                stats[status] = stats.get(status, 0) + 1
                out.write(line + "\n")
                out.flush()
                processed[0] += 1
                done_now = processed[0]

            if done_now % 100 == 0:
                rate = done_now / max(time.time() - started, 1e-9)
                remain = (len(pending) - done_now) / max(rate, 1e-9) / 3600
                logger.info("  %d/%d  성공 %d  (%.2f건/초, 잔여 %.1f시간)  %s",
                            done_now, len(pending), stats.get("ok", 0), rate, remain, stats)

            # 워커별 대기. 전체 요청률은 workers/delay 이다.
            time.sleep(max(0.0, args.delay * (1 + random.uniform(-args.jitter, args.jitter))))

    with out_path.open("a", encoding="utf-8") as out:
        threads = [threading.Thread(target=worker, args=(out,), daemon=True)
                   for _ in range(max(1, args.workers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    logger.info("완료 — %s", stats)


if __name__ == "__main__":
    main()
