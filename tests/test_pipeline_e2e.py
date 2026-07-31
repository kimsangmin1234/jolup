"""전체 파이프라인 종단 검증 (API 키·네트워크 불필요).

목 OpenAI 서버를 띄운 뒤 실제 ``openai`` SDK를 통해 다음을 순서대로 실행한다.

    1. preprocess.py       뉴스 본문 → 요약 + 감성 점수 + 1536차원 임베딩
    2. prepare_fnspid.py   FNSPID 스키마 → 학습 형식 (지표 9종 + 등락률 라벨)
    3. train.py            학습 → 검증 → 평가

합성 데이터는 뉴스 감성이 다음 거래일 등락률을 결정하도록 만들어져 있으므로,
파이프라인이 정상이면 손실이 내려가고 방향 적중률이 크게 오른다.

실행::

    python tests/test_pipeline_e2e.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.mock_openai_server import MockOpenAIHandler  # noqa: E402

POSITIVE = ("Shares surge after the company beat earnings estimates and posted "
            "record growth. Analysts issued an upgrade citing strong profit.")
NEGATIVE = ("Shares plunge after the company missed estimates and disclosed a "
            "lawsuit. Analysts issued a downgrade citing a widening loss and decline.")

TICKERS = ("AAPL", "MSFT", "NVDA")


def start_mock_server() -> tuple[HTTPServer, str]:
    """빈 포트에 목 서버를 띄우고 base_url을 반환한다."""
    server = HTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def write_fnspid_fixture(root: Path) -> tuple[Path, Path]:
    """FNSPID와 동일한 스키마의 합성 데이터를 만든다.

    뉴스 방향이 **다음** 거래일 등락률에 반영되도록 가격을 생성하므로,
    미래 정보 누수 없이 학습 가능한 신호가 존재한다.
    """
    rng = np.random.default_rng(11)
    price_dir = root / "full_history"
    price_dir.mkdir(parents=True, exist_ok=True)

    dates = [f"2020-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 26)]
    n = len(dates)
    news_rows: list[list[str]] = []

    for ticker in TICKERS:
        direction = rng.choice([1, -1], size=n)
        ret = direction * 0.012 + rng.normal(0, 0.006, n)

        close = np.empty(n)
        close[0] = 100.0
        for i in range(1, n):
            close[i] = close[i - 1] * (1 + ret[i - 1])  # 전일 뉴스가 당일에 실현

        with (price_dir / f"{ticker}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
            for i, date in enumerate(dates):
                writer.writerow([date, close[i], close[i] * 1.01, close[i] * 0.99,
                                 close[i], close[i], int(rng.uniform(1e6, 9e6))])

        for i, date in enumerate(dates):
            text = POSITIVE if direction[i] > 0 else NEGATIVE
            news_rows.append([f"{date} 00:00:00", f"{ticker} news", ticker,
                              "url", "pub", "auth", text, text])

    news_path = root / "news.csv"
    with news_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Sentiment_gpt 열을 일부러 넣지 않아 GPT 감성 추출 경로를 태운다.
        writer.writerow(["Date", "Article_title", "Stock_symbol", "Url",
                         "Publisher", "Author", "Article", "Textrank_summary"])
        writer.writerows(news_rows)

    return news_path, price_dir


def write_raw_news(root: Path) -> Path:
    """preprocess.py 검증용 뉴스 원문 JSONL."""
    path = root / "news_raw.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(40):
            positive = i % 2 == 0
            f.write(json.dumps({
                "news_id": f"n{i}",
                "ticker": "AAPL" if positive else "MSFT",
                "date": f"2020-01-{(i % 25) + 1:02d}",
                "article": POSITIVE if positive else NEGATIVE,
                "label": 0.01 if positive else -0.01,
            }) + "\n")
    return path


def run(command: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, *command], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"실패: {' '.join(command)}")
    return result.stdout + result.stderr


def main() -> None:
    server, base_url = start_mock_server()
    print(f"[0/3] 목 OpenAI 서버: {base_url}")

    env = {
        **os.environ,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": "dummy-key-for-testing",
        "NO_PROXY": "127.0.0.1,localhost",
    }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1) preprocess.py — 요약 + 감성 + 임베딩
        raw = write_raw_news(root)
        cache = root / "news_cache.jsonl"
        run(["preprocess.py", "--input", str(raw), "--output", str(cache),
             "--batch-size", "8"], env)

        records = [json.loads(line) for line in cache.open(encoding="utf-8")]
        assert len(records) == 40, len(records)
        assert len(records[0]["embedding"]) == 1536
        positive = [r["sentiment"] for r in records if r["label"] > 0]
        negative = [r["sentiment"] for r in records if r["label"] < 0]
        assert np.mean(positive) > np.mean(negative), "감성이 라벨 방향과 반대입니다."
        print(f"[1/3] preprocess.py — {len(records)}건, 임베딩 1536차원, "
              f"감성 상승 {np.mean(positive):+.2f} / 하락 {np.mean(negative):+.2f}")

        # 2) prepare_fnspid.py — FNSPID 스키마 변환
        news_csv, price_dir = write_fnspid_fixture(root)
        fn_records = root / "fn_records.jsonl"
        fn_indicators = root / "fn_indicators.npz"
        run(["prepare_fnspid.py", "--news", str(news_csv), "--price-dir", str(price_dir),
             "--tickers", ",".join(TICKERS), "--start", "2020-01-01", "--end", "2020-12-31",
             "--out-records", str(fn_records), "--out-indicators", str(fn_indicators),
             "--embedding", "openai", "--batch-size", "64"], env)

        converted = [json.loads(line) for line in fn_records.open(encoding="utf-8")]
        assert len(converted) > 800, len(converted)
        assert all(-1.0 <= r["sentiment"] <= 1.0 for r in converted)
        print(f"[2/3] prepare_fnspid.py — {len(converted)}건 변환, 종목 {len(TICKERS)}개")

        # 3) train.py — 학습
        output = run(["train.py", "--records", str(fn_records),
                      "--indicators", str(fn_indicators),
                      "--train-end", "2020-08-25", "--valid-end", "2020-10-25",
                      "--epochs", "12", "--checkpoint", str(root / "best.pt"),
                      "--scaler-out", str(root / "scaler.json")], env)

        last = [l for l in output.splitlines() if "테스트 —" in l][-1]
        accuracy = float(last.split("dir_acc")[1].strip())
        print(f"[3/3] train.py — {last.split('테스트 —')[1].strip()}")

        # 신호가 심어진 합성 데이터이므로 방향 적중률이 확실히 올라야 한다.
        assert accuracy > 0.8, f"방향 적중률이 너무 낮습니다: {accuracy}"

    server.shutdown()
    print("\n전체 파이프라인 검증을 통과했습니다.")


if __name__ == "__main__":
    main()
