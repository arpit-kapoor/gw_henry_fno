from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FNO on a single Henry coupling scenario",
    )

    parser.add_argument(
        "--scenario-dir",
        type=Path,
        required=True,
        help="Path to one scenario directory containing run_*/windows.npz",
    )

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")

    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=5,
        help="Learning rate scheduler step size (epochs)",
    )
    parser.add_argument(
        "--scheduler-decay",
        type=float,
        default=0.98,
        help="Learning rate scheduler decay factor (gamma)",
    )
    parser.add_argument(
        "--disable-scheduler",
        action="store_true",
        help="Disable learning rate scheduler (default: enabled with exponential decay)",
    )

    parser.add_argument("--train-ratio", type=float, default=0.8, help="Run-level train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training")

    parser.add_argument("--n-modes-x", type=int, default=16, help="FNO modes in X dimension")
    parser.add_argument("--n-modes-y", type=int, default=16, help="FNO modes in Y dimension")
    parser.add_argument("--hidden-channels", type=int, default=32, help="FNO hidden channels")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of FNO layers")

    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned memory in DataLoaders",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data using mean/std from training set",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device selection mode",
    )

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if not (0.0 < args.train_ratio < 1.0):
        parser.error("--train-ratio must be in (0, 1)")

    return args
