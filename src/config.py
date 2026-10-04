from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def add_scenario_arg(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
    default: Optional[Path] = None,
) -> None:
    """Register the scenarios directory CLI argument."""
    parser.add_argument(
        "--scenario-dir",
        "--scenarios-dir",
        dest="scenario_dir",
        type=Path,
        required=required,
        default=default,
        help="Path to scenarios directory containing scenario_NNN/scenario.npz subdirs",
    )


def add_training_args(
    parser: argparse.ArgumentParser,
    *,
    default_epochs: int = 20,
    default_batch_size: int = 64,
    default_learning_rate: float = 1e-3,
    default_weight_decay: float = 0.0,
) -> None:
    """Register training-loop and optimizer hyperparameters."""
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help=(
            "Split each batch into chunks of this size and accumulate gradients. "
            "Gives the same optimizer update as --batch-size with lower peak memory. "
            "0 disables (default)"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=default_learning_rate, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=default_weight_decay, help="AdamW weight decay")


def add_scheduler_args(
    parser: argparse.ArgumentParser,
    *,
    default_step_size: int = 5,
    default_decay: float = 0.98,
) -> None:
    """Register learning-rate scheduler controls."""
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=default_step_size,
        help="Learning rate scheduler step size (epochs)",
    )
    parser.add_argument(
        "--scheduler-decay",
        type=float,
        default=default_decay,
        help="Learning rate scheduler decay factor (gamma)",
    )
    parser.add_argument(
        "--disable-scheduler",
        action="store_true",
        help="Disable learning rate scheduler (default: enabled with step decay)",
    )


def add_split_and_seed_args(
    parser: argparse.ArgumentParser,
    *,
    default_train_ratio: float = 0.7,
    default_seed: int = 42,
) -> None:
    """Register train/validation split and reproducibility arguments.

    The train/val split is applied at the run-index level with the same
    partition shared across all scenarios (run index k is always in the same
    split regardless of which scenario it belongs to).
    """
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=default_train_ratio,
        help="Fraction of runs used for training (shared across all scenarios); default 0.7",
    )
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed for split and training")


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    default_n_modes_t: int = 8,
    default_n_modes_z: int = 8,
    default_n_modes_x: int = 16,
    default_hidden_channels: int = 32,
    default_n_layers: int = 4,
) -> None:
    """Register 3-D FNO architecture arguments.

    The three mode arguments map to the physical dimensions of the problem:

    - ``--n-modes-t``: Fourier modes along the **time** axis (T_in=25, max ≤ 12)
    - ``--n-modes-z``: Fourier modes along the **vertical** spatial axis (Z=20, max ≤ 10)
    - ``--n-modes-x``: Fourier modes along the **horizontal** spatial axis (X=40, max ≤ 20)
    """
    parser.add_argument(
        "--n-modes-t",
        type=int,
        default=default_n_modes_t,
        help="FNO Fourier modes along the time axis (T_in=25, useful range ≤ 12)",
    )
    parser.add_argument(
        "--n-modes-z",
        type=int,
        default=default_n_modes_z,
        help="FNO Fourier modes along the vertical axis (Z=20, useful range ≤ 10)",
    )
    parser.add_argument(
        "--n-modes-x",
        type=int,
        default=default_n_modes_x,
        help="FNO Fourier modes along the horizontal axis (X=40, useful range ≤ 20)",
    )
    parser.add_argument("--hidden-channels", type=int, default=default_hidden_channels, help="FNO hidden channels")
    parser.add_argument("--n-layers", type=int, default=default_n_layers, help="Number of FNO Fourier layers")


def add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_num_workers: int = 0,
    default_device: str = "auto",
) -> None:
    """Register runtime arguments such as device and loader settings."""
    parser.add_argument("--num-workers", type=int, default=default_num_workers, help="DataLoader workers")
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned memory in DataLoaders (effective only with CUDA)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data using per-channel mean/std from training set",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device selection mode",
    )


def validate_common_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate shared numeric argument constraints for training scripts."""
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if getattr(args, "micro_batch_size", 0) < 0:
        parser.error("--micro-batch-size must be >= 0")
    if not (0.0 < args.train_ratio < 1.0):
        parser.error("--train-ratio must be in (0, 1)")
    if hasattr(args, "scheduler_step_size") and args.scheduler_step_size <= 0:
        parser.error("--scheduler-step-size must be > 0")
    if hasattr(args, "scheduler_decay") and not (0.0 < args.scheduler_decay <= 1.0):
        parser.error("--scheduler-decay must be in (0, 1]")
    if hasattr(args, "hidden_channels") and args.hidden_channels <= 0:
        parser.error("--hidden-channels must be > 0")
    if hasattr(args, "n_layers") and args.n_layers <= 0:
        parser.error("--n-layers must be > 0")


def build_parser() -> argparse.ArgumentParser:
    """Create the standard parser for single-model training."""
    parser = argparse.ArgumentParser(
        description="Train a 3-D FNO across all Henry scenarios with run-level train/val split",
    )

    add_scenario_arg(parser, required=True)
    add_training_args(parser)
    add_scheduler_args(parser)
    add_split_and_seed_args(parser)
    add_model_args(parser)
    add_runtime_args(parser)

    return parser


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args()
    validate_common_args(parser, args)
    return args
