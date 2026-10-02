"""내 PC에서 발행 시각을 마저 수집하고 결과를 깃에 올린다.

서버 IP 가 nasdaq.com 에 차단되어(전 요청 403) 나머지를 다른 IP 에서 받아야
한다. 표준 라이브러리만 쓰므로 파이썬과 깃만 있으면 윈도우에서도 돈다.

    git clone https://github.com/kimsangmin1234/jolup.git
    cd jolup
    git checkout claude/paper-technique-modular-code-rxj8uw
    python crawl_local.py

500건마다 data/fnspid/publish_times_local.jsonl 을 커밋·푸시한다. 중간에
꺼도 같은 명령으로 이어서 받는다. 요청 간격 기본 4초, 동시 요청 1개다.
빠르게 돌리면 이 PC 의 IP 도 차단된다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URLS = ROOT / "data/fnspid/urls_remaining.txt.gz"
OUT = ROOT / "data/fnspid/publish_times_local.jsonl"


def count(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def git(*args: str) -> int:
    return subprocess.run(["git", *args], cwd=ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--chunk", type=int, default=500, help="이만큼 받을 때마다 푸시")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    while True:
        before = count(OUT)
        code = subprocess.run([
            sys.executable, str(ROOT / "crawl_publish_time.py"),
            "--urls", str(URLS), "--out", str(OUT),
            "--delay", str(args.delay), "--workers", "1",
            "--limit", str(args.chunk), "--shuffle",
        ], cwd=ROOT).returncode
        after = count(OUT)
        if code != 0 or after == before:
            print(f"끝 — 누적 {after}건")
            break
        if not args.no_push:
            git("add", "-f", str(OUT.relative_to(ROOT)))
            git("commit", "-q", "-m", f"로컬 PC 발행 시각 수집 {after}건")
            git("pull", "-q", "--rebase")
            if git("push", "-q") != 0:
                print("푸시 실패 — 결과는 파일에 남아 있습니다. 나중에 git push 하세요.")


if __name__ == "__main__":
    main()
