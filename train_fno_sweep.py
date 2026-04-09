from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import torch

from src.data.henry_scenario_dataset import create_henry_dataloaders
from src.neuralop import FNO
from src.neuralop.losses import LpLoss
from src.config import (
    add_model_args,
    add_runtime_args,
    add_scenario_arg,
    add_scheduler_args,
    add_split_and_seed_args,
    add_training_args,
    validate_common_args,
)
from train_fno import evaluate_mse, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train multiple FNO models for one scenario and append results to CSV",
    )

    add_scenario_arg(
        parser,
        required=False,
        default=Path(
            "/Users/akap5486/Projects/groundwater/data/henry_data/all_scenarios/coupling_scenarios/scenario_01"
        ),
    )
    # Keep defaults aligned with current run_training.sh
    add_training_args(parser, default_epochs=100, default_batch_size=512)
    add_scheduler_args(parser, default_step_size=5, default_decay=0.98)
    add_split_and_seed_args(parser, default_train_ratio=0.8, default_seed=42)
    add_model_args(
        parser,
        default_n_modes_x=8,
        default_n_modes_y=16,
        default_hidden_channels=32,
        default_n_layers=4,
    )
    add_runtime_args(parser, default_num_workers=0, default_device="auto")

    parser.add_argument(
        "--hidden-channels-list",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated hidden channel values",
    )

    parser.set_defaults(normalize=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where scenario-specific CSV is stored",
    )

    return parser


def parse_hidden_channels(values: str) -> list[int]:
    # Accept a compact CLI form like "8,16,32,64,128".
    parsed = [int(v.strip()) for v in values.split(",") if v.strip()]
    if not parsed:
        raise ValueError("--hidden-channels-list must contain at least one integer")
    return parsed


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scenario_results_csv(results_dir: Path, scenario_dir: Path) -> Path:
    scenario_group = resolve_scenario_group(scenario_dir)
    scenario_name = scenario_dir.name
    return results_dir / f"{scenario_group}_{scenario_name}_fno_sweep_results.csv"


def resolve_scenario_group(scenario_dir: Path) -> str:
    """Resolve scenario group from scenario_config.json."""
    config_path = scenario_dir / "scenario_config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    scenario_group = str(config["scenario_group"]).strip().lower()
    if not scenario_group:
        raise ValueError(f"scenario_group is empty in {config_path}")

    return scenario_group


def append_result_row(csv_path: Path, row: dict[str, object], fieldnames: Iterable[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Write header only for first write to keep append behavior simple.
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_l2(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> float:
    """Evaluate mean L2 loss across a dataloader using the training LpLoss setup."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)

            loss = criterion(pred, yb)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Dataloader produced zero samples during L2 evaluation")

    return total_loss / total_samples


def train_one_model(
    *,
    scenario_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    train_ratio: float,
    seed: int,
    device: torch.device,
    n_modes_x: int,
    n_modes_y: int,
    hidden_channels: int,
    n_layers: int,
    num_workers: int,
    pin_memory: bool,
    normalize: bool,
    disable_scheduler: bool,
    scheduler_step_size: int,
    scheduler_decay: float,
) -> tuple[int, float, float, float, float]:
    # Build split loaders once per model size; normalization stats are train-split based.
    dataloaders = create_henry_dataloaders(
        scenario_dir=scenario_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
    )

    if normalize:
        train_loader, val_loader, normalizer = dataloaders
    else:
        train_loader, val_loader = dataloaders
        normalizer = None

    sample_x, sample_y = next(iter(train_loader))
    in_channels = int(sample_x.shape[1])
    out_channels = int(sample_y.shape[1])

    model = FNO(
        n_modes=(n_modes_x, n_modes_y),
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
    ).to(device)

    # Parameter count is computed directly from the instantiated model.
    total_params = count_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_criterion = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")

    scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
    if not disable_scheduler:
        # Exponential-style decay via StepLR (gamma applied every step_size epochs).
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_decay,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = train_criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size_local = xb.size(0)
            running_loss += loss.item() * batch_size_local
            total_samples += batch_size_local

        epoch_train_l2 = running_loss / total_samples
        epoch_val_mse = evaluate_mse(model, val_loader, device, normalizer)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} - "
            f"hidden_channels: {hidden_channels}, "
            f"train_l2: {epoch_train_l2:.6f}, "
            f"val_mse: {epoch_val_mse:.6f}, "
            f"lr: {current_lr:.6e}"
        )

        if scheduler is not None:
            scheduler.step()

    final_train_l2 = evaluate_l2(model, train_loader, device)
    final_val_l2 = evaluate_l2(model, val_loader, device)
    final_train_mse = evaluate_mse(model, train_loader, device, normalizer)
    final_val_mse = evaluate_mse(model, val_loader, device, normalizer)

    return total_params, final_train_l2, final_val_l2, final_train_mse, final_val_mse


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_common_args(parser, args)

    hidden_channels_values = parse_hidden_channels(args.hidden_channels_list)

    set_seed(args.seed)
    device = resolve_device(args.device)

    csv_path = scenario_results_csv(args.results_dir, args.scenario_dir)
    fieldnames = [
        "timestamp",
        "scenario",
        "hidden_channels",
        "n_modes_x",
        "n_modes_y",
        "n_layers",
        "total_params",
        "train_l2",
        "val_l2",
        "train_mse",
        "val_mse",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "normalize",
        "scheduler_enabled",
        "scheduler_step_size",
        "scheduler_decay",
        "device",
    ]

    print("Starting multi-model FNO sweep")
    print(f"scenario: {args.scenario_dir}")
    print(f"hidden_channels_list: {hidden_channels_values}")
    print(f"fixed architecture: n_modes=({args.n_modes_x}, {args.n_modes_y}), n_layers={args.n_layers}")
    print(f"results csv: {csv_path}")

    for hidden_channels in hidden_channels_values:
        print("=" * 60)
        print(f"Training model with hidden_channels={hidden_channels}")
        print("=" * 60)

        # Train one architecture variant and collect final metrics.
        total_params, train_l2, val_l2, train_mse, val_mse = train_one_model(
            scenario_dir=args.scenario_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_ratio=args.train_ratio,
            seed=args.seed,
            device=device,
            n_modes_x=args.n_modes_x,
            n_modes_y=args.n_modes_y,
            hidden_channels=hidden_channels,
            n_layers=args.n_layers,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            normalize=args.normalize,
            disable_scheduler=args.disable_scheduler,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_decay=args.scheduler_decay,
        )

        # Append one row per model size for scenario-level comparison.
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": args.scenario_dir.name,
            "hidden_channels": hidden_channels,
            "n_modes_x": args.n_modes_x,
            "n_modes_y": args.n_modes_y,
            "n_layers": args.n_layers,
            "total_params": total_params,
            "train_l2": f"{train_l2:.6f}",
            "val_l2": f"{val_l2:.6f}",
            "train_mse": f"{train_mse:.6f}",
            "val_mse": f"{val_mse:.6f}",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "normalize": args.normalize,
            "scheduler_enabled": (not args.disable_scheduler),
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_decay": args.scheduler_decay,
            "device": str(device),
        }
        append_result_row(csv_path, row, fieldnames)

        print(
            f"Completed hidden_channels={hidden_channels} | "
            f"params={total_params} | "
            f"train_l2={train_l2:.6f} | val_l2={val_l2:.6f} | "
            f"train_mse={train_mse:.6f} | val_mse={val_mse:.6f}"
        )

    print("=" * 60)
    print("Sweep finished")
    print(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()
