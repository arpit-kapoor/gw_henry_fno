from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
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


@dataclass(frozen=True)
class ModelSizeConfig:
    label: str
    hidden_channels: int
    n_modes_x: int
    n_modes_y: int
    n_layers: int


MODEL_SIZE_PRESETS: dict[str, ModelSizeConfig] = {
    # Presets inspired by coordinated scaling used in PDE surrogate benchmarks.
    "tiny": ModelSizeConfig("tiny", hidden_channels=4, n_modes_x=4, n_modes_y=8, n_layers=4),
    "small": ModelSizeConfig("small", hidden_channels=8, n_modes_x=8, n_modes_y=16, n_layers=4),
    "medium": ModelSizeConfig("medium", hidden_channels=16, n_modes_x=8, n_modes_y=16, n_layers=6),
    "large": ModelSizeConfig("large", hidden_channels=32, n_modes_x=12, n_modes_y=24, n_layers=6),
    "huge": ModelSizeConfig("huge", hidden_channels=48, n_modes_x=16, n_modes_y=32, n_layers=6),
    "massive": ModelSizeConfig("massive", hidden_channels=64, n_modes_x=24, n_modes_y=48, n_layers=8),
}


def _channelwise_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute per-sample, per-channel relative L2.

    Returns a tensor of shape (B, C).
    """
    pred_flat = pred.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)
    diff_norm = torch.linalg.norm(pred_flat - target_flat, dim=-1)
    target_norm = torch.linalg.norm(target_flat, dim=-1)
    return diff_norm / (target_norm + eps)


def _channelwise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-sample, per-channel MSE over spatial dimensions.

    Returns a tensor of shape (B, C).
    """
    squared_error = (pred - target) ** 2
    return squared_error.flatten(start_dim=2).mean(dim=-1)


def evaluate_channel_metrics(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    normalizer=None,
    eps: float = 1e-12,
) -> dict[str, list[float]]:
    """Evaluate channel-wise normalized/denormalized relative L2 and MSE."""
    model.eval()
    total_samples = 0

    rel_l2_norm_sum = None
    rel_l2_denorm_sum = None
    mse_norm_sum = None
    mse_denorm_sum = None

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)

            if pred.shape != yb.shape:
                raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {yb.shape}")

            pred_denorm = pred
            yb_denorm = yb
            if normalizer is not None:
                pred_denorm = normalizer.denormalize_output(pred)
                yb_denorm = normalizer.denormalize_output(yb)

            rel_l2_norm = _channelwise_relative_l2(pred, yb, eps=eps)
            rel_l2_denorm = _channelwise_relative_l2(pred_denorm, yb_denorm, eps=eps)
            mse_norm = _channelwise_mse(pred, yb)
            mse_denorm = _channelwise_mse(pred_denorm, yb_denorm)

            if rel_l2_norm_sum is None:
                channels = rel_l2_norm.shape[1]
                rel_l2_norm_sum = torch.zeros(channels, dtype=torch.float64, device=device)
                rel_l2_denorm_sum = torch.zeros(channels, dtype=torch.float64, device=device)
                mse_norm_sum = torch.zeros(channels, dtype=torch.float64, device=device)
                mse_denorm_sum = torch.zeros(channels, dtype=torch.float64, device=device)

            rel_l2_norm_sum += rel_l2_norm.sum(dim=0, dtype=torch.float64)
            rel_l2_denorm_sum += rel_l2_denorm.sum(dim=0, dtype=torch.float64)
            mse_norm_sum += mse_norm.sum(dim=0, dtype=torch.float64)
            mse_denorm_sum += mse_denorm.sum(dim=0, dtype=torch.float64)

            total_samples += xb.size(0)

    if total_samples == 0:
        raise ValueError("Dataloader produced zero samples during channel metric evaluation")

    return {
        "rel_l2_norm_channels": (rel_l2_norm_sum / total_samples).cpu().tolist(),
        "rel_l2_denorm_channels": (rel_l2_denorm_sum / total_samples).cpu().tolist(),
        "mse_norm_channels": (mse_norm_sum / total_samples).cpu().tolist(),
        "mse_denorm_channels": (mse_denorm_sum / total_samples).cpu().tolist(),
    }


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
        help="Comma-separated hidden channel values (used when --sweep-mode=hidden)",
    )

    parser.add_argument(
        "--sweep-mode",
        type=str,
        choices=["hidden", "preset"],
        default="hidden",
        help="Sweep hidden width only or coordinated model-size presets",
    )

    parser.add_argument(
        "--model-size-presets",
        type=str,
        default="tiny,small,medium,large,huge,massive",
        help="Comma-separated preset names for --sweep-mode=preset",
    )

    parser.set_defaults(normalize=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where scenario-specific CSV is stored",
    )

    parser.add_argument(
        "--artifact-dir-name",
        type=str,
        default="validation_final_step_artifacts",
        help="Subdirectory under results-dir for final-step prediction artifacts",
    )

    return parser


def parse_hidden_channels(values: str) -> list[int]:
    # Accept a compact CLI form like "8,16,32,64,128".
    parsed = [int(v.strip()) for v in values.split(",") if v.strip()]
    if not parsed:
        raise ValueError("--hidden-channels-list must contain at least one integer")
    return parsed


def parse_model_size_presets(values: str) -> list[ModelSizeConfig]:
    requested = [v.strip().lower() for v in values.split(",") if v.strip()]
    if not requested:
        raise ValueError("--model-size-presets must contain at least one preset name")

    configs: list[ModelSizeConfig] = []
    for preset_name in requested:
        if preset_name not in MODEL_SIZE_PRESETS:
            valid = ", ".join(sorted(MODEL_SIZE_PRESETS))
            raise ValueError(f"Unknown model size preset '{preset_name}'. Valid presets: {valid}")
        configs.append(MODEL_SIZE_PRESETS[preset_name])

    return configs


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


def _extract_final_validation_sample(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    normalizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return denormalized prediction/target for the final validation sample."""
    dataset = val_loader.dataset
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty; cannot create final-step artifacts")

    # Validation DataLoader is not shuffled; the final element maps to final step ordering.
    xb_single, yb_single = dataset[len(dataset) - 1]
    xb = xb_single.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(xb)

    yb = yb_single.unsqueeze(0).to(device)
    if normalizer is not None:
        pred = normalizer.denormalize_output(pred)
        yb = normalizer.denormalize_output(yb)

    return pred.squeeze(0).detach().cpu(), yb.squeeze(0).detach().cpu()


def _plot_validation_prediction_comparison(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    output_path: Path,
    scenario_name: str,
    model_size_label: str,
    hidden_channels: int,
) -> None:
    """Save a 2 x C panel plot: ground truth row and prediction row."""
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Prediction/ground truth shape mismatch: {prediction.shape} vs {ground_truth.shape}")

    num_channels = int(ground_truth.shape[0])
    channel_names = [f"Channel {idx}" for idx in range(num_channels)]
    if num_channels >= 2:
        channel_names[0] = "Concentration"
        channel_names[1] = "Head"

    fig, axes = plt.subplots(
        2,
        num_channels,
        figsize=(max(6.0, 4.8 * num_channels), 7.8),
        constrained_layout=True,
    )

    if num_channels == 1:
        axes_2d = np.array([[axes[0]], [axes[1]]], dtype=object)
    else:
        axes_2d = axes
    for channel_idx in range(num_channels):
        gt_ch = ground_truth[channel_idx].numpy()
        pred_ch = prediction[channel_idx].numpy()
        vmin = float(min(gt_ch.min(), pred_ch.min()))
        vmax = float(max(gt_ch.max(), pred_ch.max()))

        ax_gt = axes_2d[0, channel_idx]
        im_gt = ax_gt.imshow(gt_ch, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"Ground Truth - {channel_names[channel_idx]}")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        ax_pred = axes_2d[1, channel_idx]
        im_pred = ax_pred.imshow(pred_ch, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"FNO Prediction - {channel_names[channel_idx]}")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        ax_gt.set_xlabel("x-index")
        ax_gt.set_ylabel("z-index")
        ax_pred.set_xlabel("x-index")
        ax_pred.set_ylabel("z-index")

    fig.suptitle(
        (
            f"Scenario: {scenario_name} | "
            f"size_preset={model_size_label} | "
            f"hidden_channels={hidden_channels} | "
            "Validation Final Step"
        ),
        fontsize=11,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_validation_final_step_artifacts(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    normalizer,
    output_dir: Path,
    scenario_name: str,
    model_size_label: str,
    hidden_channels: int,
) -> tuple[Path, Path]:
    """Save final validation ground truth/prediction arrays and comparison plot."""
    prediction, ground_truth = _extract_final_validation_sample(
        model=model,
        val_loader=val_loader,
        device=device,
        normalizer=normalizer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{scenario_name}_{model_size_label}_val_final_step"

    npz_path = output_dir / f"{base_name}.npz"
    np.savez_compressed(
        npz_path,
        prediction=prediction.numpy(),
        ground_truth=ground_truth.numpy(),
        abs_error=(prediction - ground_truth).abs().numpy(),
    )

    fig_path = output_dir / f"{base_name}.png"
    _plot_validation_prediction_comparison(
        prediction=prediction,
        ground_truth=ground_truth,
        output_path=fig_path,
        scenario_name=scenario_name,
        model_size_label=model_size_label,
        hidden_channels=hidden_channels,
    )

    return npz_path, fig_path


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
) -> tuple[
    torch.nn.Module,
    object,
    object,
    int,
    float,
    float,
    float,
    float,
    int,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]:
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
    train_channel_metrics = evaluate_channel_metrics(
        model=model,
        dataloader=train_loader,
        device=device,
        normalizer=normalizer,
    )
    val_channel_metrics = evaluate_channel_metrics(
        model=model,
        dataloader=val_loader,
        device=device,
        normalizer=normalizer,
    )

    return (
        model,
        val_loader,
        normalizer,
        total_params,
        final_train_l2,
        final_val_l2,
        final_train_mse,
        final_val_mse,
        out_channels,
        json.dumps(train_channel_metrics["rel_l2_norm_channels"]),
        json.dumps(val_channel_metrics["rel_l2_norm_channels"]),
        json.dumps(train_channel_metrics["rel_l2_denorm_channels"]),
        json.dumps(val_channel_metrics["rel_l2_denorm_channels"]),
        json.dumps(train_channel_metrics["mse_norm_channels"]),
        json.dumps(val_channel_metrics["mse_norm_channels"]),
        json.dumps(train_channel_metrics["mse_denorm_channels"]),
        json.dumps(val_channel_metrics["mse_denorm_channels"]),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_common_args(parser, args)

    if args.sweep_mode == "preset":
        sweep_configs = parse_model_size_presets(args.model_size_presets)
    else:
        hidden_channels_values = parse_hidden_channels(args.hidden_channels_list)
        sweep_configs = [
            ModelSizeConfig(
                label=f"hidden_{hidden_channels}",
                hidden_channels=hidden_channels,
                n_modes_x=args.n_modes_x,
                n_modes_y=args.n_modes_y,
                n_layers=args.n_layers,
            )
            for hidden_channels in hidden_channels_values
        ]

    set_seed(args.seed)
    device = resolve_device(args.device)

    csv_path = scenario_results_csv(args.results_dir, args.scenario_dir)
    artifact_dir = args.results_dir / args.artifact_dir_name / args.scenario_dir.name
    fieldnames = [
        "timestamp",
        "scenario",
        "model_size_label",
        "hidden_channels",
        "n_modes_x",
        "n_modes_y",
        "n_layers",
        "total_params",
        "train_l2",
        "val_l2",
        "train_mse",
        "val_mse",
        "num_output_channels",
        "train_rel_l2_norm_channels",
        "val_rel_l2_norm_channels",
        "train_rel_l2_denorm_channels",
        "val_rel_l2_denorm_channels",
        "train_mse_norm_channels",
        "val_mse_norm_channels",
        "train_mse_denorm_channels",
        "val_mse_denorm_channels",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "normalize",
        "scheduler_enabled",
        "scheduler_step_size",
        "scheduler_decay",
        "device",
        "val_final_step_npz",
        "val_final_step_plot",
    ]

    print("Starting multi-model FNO sweep")
    print(f"scenario: {args.scenario_dir}")
    print(f"sweep_mode: {args.sweep_mode}")
    if args.sweep_mode == "preset":
        print(f"model_size_presets: {[cfg.label for cfg in sweep_configs]}")
    else:
        print(f"hidden_channels_list: {[cfg.hidden_channels for cfg in sweep_configs]}")
        print(f"fixed architecture: n_modes=({args.n_modes_x}, {args.n_modes_y}), n_layers={args.n_layers}")
    print(f"results csv: {csv_path}")
    print(f"validation artifact dir: {artifact_dir}")

    for config in sweep_configs:
        # Re-seed per configuration so initialization and shuffled batch order
        # do not depend on loop position.
        model_seed = (
            args.seed
            + config.hidden_channels
            + config.n_modes_x
            + config.n_modes_y
            + 10 * config.n_layers
        )
        set_seed(model_seed)

        print("=" * 60)
        print(
            "Training model config "
            f"label={config.label}, hidden_channels={config.hidden_channels}, "
            f"n_modes=({config.n_modes_x}, {config.n_modes_y}), n_layers={config.n_layers}"
        )
        print(f"model_seed: {model_seed}")
        print("=" * 60)

        # Train one architecture variant and collect final metrics.
        (
            model,
            val_loader,
            normalizer,
            total_params,
            train_l2,
            val_l2,
            train_mse,
            val_mse,
            num_output_channels,
            train_rel_l2_norm_channels,
            val_rel_l2_norm_channels,
            train_rel_l2_denorm_channels,
            val_rel_l2_denorm_channels,
            train_mse_norm_channels,
            val_mse_norm_channels,
            train_mse_denorm_channels,
            val_mse_denorm_channels,
        ) = train_one_model(
            scenario_dir=args.scenario_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_ratio=args.train_ratio,
            seed=args.seed,
            device=device,
            n_modes_x=config.n_modes_x,
            n_modes_y=config.n_modes_y,
            hidden_channels=config.hidden_channels,
            n_layers=config.n_layers,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            normalize=args.normalize,
            disable_scheduler=args.disable_scheduler,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_decay=args.scheduler_decay,
        )

        npz_artifact_path, plot_artifact_path = save_validation_final_step_artifacts(
            model=model,
            val_loader=val_loader,
            device=device,
            normalizer=normalizer,
            output_dir=artifact_dir,
            scenario_name=args.scenario_dir.name,
            model_size_label=config.label,
            hidden_channels=config.hidden_channels,
        )

        # Append one row per model size for scenario-level comparison.
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": args.scenario_dir.name,
            "model_size_label": config.label,
            "hidden_channels": config.hidden_channels,
            "n_modes_x": config.n_modes_x,
            "n_modes_y": config.n_modes_y,
            "n_layers": config.n_layers,
            "total_params": total_params,
            "train_l2": f"{train_l2:.6f}",
            "val_l2": f"{val_l2:.6f}",
            "train_mse": f"{train_mse:.6f}",
            "val_mse": f"{val_mse:.6f}",
            "num_output_channels": num_output_channels,
            "train_rel_l2_norm_channels": train_rel_l2_norm_channels,
            "val_rel_l2_norm_channels": val_rel_l2_norm_channels,
            "train_rel_l2_denorm_channels": train_rel_l2_denorm_channels,
            "val_rel_l2_denorm_channels": val_rel_l2_denorm_channels,
            "train_mse_norm_channels": train_mse_norm_channels,
            "val_mse_norm_channels": val_mse_norm_channels,
            "train_mse_denorm_channels": train_mse_denorm_channels,
            "val_mse_denorm_channels": val_mse_denorm_channels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "normalize": args.normalize,
            "scheduler_enabled": (not args.disable_scheduler),
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_decay": args.scheduler_decay,
            "device": str(device),
            "val_final_step_npz": str(npz_artifact_path),
            "val_final_step_plot": str(plot_artifact_path),
        }
        append_result_row(csv_path, row, fieldnames)

        print(
            f"Completed label={config.label}, hidden_channels={config.hidden_channels} | "
            f"params={total_params} | "
            f"train_l2={train_l2:.6f} | val_l2={val_l2:.6f} | "
            f"train_mse={train_mse:.6f} | val_mse={val_mse:.6f} | "
            f"val_rel_l2_denorm_channels={val_rel_l2_denorm_channels} | "
            f"saved_npz={npz_artifact_path.name} | "
            f"saved_plot={plot_artifact_path.name}"
        )

    print("=" * 60)
    print("Sweep finished")
    print(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()
