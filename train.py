"""학습 / 평가 스크립트.

등락률은 연속값이므로 MSE로 회귀 학습하되, 실무에서 중요한 방향성 적중률
(directional accuracy)도 함께 보고한다.

사용 예::

    python train.py --records data/news_cache.jsonl \\
                    --indicators data/indicators.npz \\
                    --train-end 2025-06-30 --valid-end 2025-09-30
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from data.dataset import (
    NewsStockDataset,
    fit_scaler_on_train,
    load_indicators,
    load_records,
    split_by_date,
)
from model import NewsDrivenStockPredictor

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: NewsDrivenStockPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> dict[str, float]:
    """optimizer가 주어지면 학습, 아니면 평가."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_abs = 0.0
    correct_direction = 0
    count = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            embedding = batch["embedding"].to(device)
            sentiment = batch["sentiment"].to(device)
            indicators = batch["indicators"].to(device)
            label = batch["label"].to(device)

            output = model(embedding, sentiment, indicators)
            loss = criterion(output.prediction, label)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_abs += (output.prediction - label).abs().sum().item()
            # 등락 방향이 일치한 비율 (label이 0인 경우는 오답 처리)
            correct_direction += (
                torch.sign(output.prediction) == torch.sign(label)
            ).sum().item()
            count += batch_size

    if count == 0:
        return {"mse": float("nan"), "mae": float("nan"), "dir_acc": float("nan")}

    return {
        "mse": total_loss / count,
        "mae": total_abs / count,
        "dir_acc": correct_direction / count,
    }


def build_loaders(args, config: ModelConfig):
    records = load_records(args.records)
    indicators, date_index = load_indicators(args.indicators)

    train_rec, valid_rec, test_rec = split_by_date(records, args.train_end, args.valid_end)
    logger.info(
        "레코드 수 — 학습 %d / 검증 %d / 평가 %d", len(train_rec), len(valid_rec), len(test_rec)
    )

    # 스케일러는 반드시 학습 구간만으로 fit (미래 정보 누수 방지)
    scaler = fit_scaler_on_train(train_rec, indicators, date_index)
    if args.scaler_out:
        scaler.save(args.scaler_out)

    def make(recs, shuffle):
        dataset = NewsStockDataset(
            records=recs,
            indicators=indicators,
            date_index=date_index,
            scaler=scaler,
            lookback=config.tcn.lookback,
            embedding_dim=config.news.embedding_dim,
        )
        return DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    return make(train_rec, True), make(valid_rec, False), make(test_rec, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="뉴스 이벤트 기반 주가 예측 모델 학습")
    parser.add_argument("--records", required=True, help="LLM 전처리 캐시 JSONL")
    parser.add_argument("--indicators", required=True, help="종목별 지표 npz")
    parser.add_argument("--train-end", required=True, help="학습 구간 종료일 (YYYY-MM-DD)")
    parser.add_argument("--valid-end", required=True, help="검증 구간 종료일 (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--scaler-out", default="checkpoints/scaler.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = ModelConfig()
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    config.train.device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    set_seed(config.train.seed)
    device = torch.device(config.train.device)

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if args.scaler_out:
        Path(args.scaler_out).parent.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader = build_loaders(args, config)

    model = NewsDrivenStockPredictor(config).to(device)
    logger.info("학습 가능 파라미터: %s", f"{model.num_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    best_valid = float("inf")
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, config.train.grad_clip
        )
        valid_metrics = run_epoch(model, valid_loader, criterion, device)

        logger.info(
            "epoch %02d | train mse %.6f dir %.3f | valid mse %.6f mae %.6f dir %.3f",
            epoch,
            train_metrics["mse"], train_metrics["dir_acc"],
            valid_metrics["mse"], valid_metrics["mae"], valid_metrics["dir_acc"],
        )

        if valid_metrics["mse"] < best_valid:
            best_valid = valid_metrics["mse"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.checkpoint)
            logger.info("  ↳ 최고 성능 갱신, 체크포인트 저장: %s", args.checkpoint)

    if Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    test_metrics = run_epoch(model, test_loader, criterion, device)
    logger.info(
        "테스트 — mse %.6f mae %.6f dir_acc %.3f",
        test_metrics["mse"], test_metrics["mae"], test_metrics["dir_acc"],
    )


if __name__ == "__main__":
    main()
