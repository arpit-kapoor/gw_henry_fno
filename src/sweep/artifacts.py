from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _extract_final_split_sample(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    normalizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return denormalized prediction/target for one sample from a split loader."""
    dataset = data_loader.dataset
    if len(dataset) == 0:
        raise ValueError("Split dataset is empty; cannot create final-step artifacts")

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


def _extract_all_split_samples(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    normalizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return denormalized prediction/target tensors for all samples in a split loader."""
    all_predictions: list[torch.Tensor] = []
    all_ground_truth: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)

            if normalizer is not None:
                pred = normalizer.denormalize_output(pred)
                yb = normalizer.denormalize_output(yb)

            all_predictions.append(pred.detach().cpu())
            all_ground_truth.append(yb.detach().cpu())

    if len(all_predictions) == 0:
        raise ValueError("Split loader produced zero samples; cannot create NPZ artifacts")

    predictions = torch.cat(all_predictions, dim=0)
    ground_truth = torch.cat(all_ground_truth, dim=0)
    return predictions, ground_truth


def _plot_validation_prediction_comparison(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    output_path: Path,
    scenario_name: str,
    split_label: str,
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
            f"Split: {split_label} | "
            f"size_preset={model_size_label} | "
            f"hidden_channels={hidden_channels} | "
            "Final Step"
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
    predictions_all, ground_truth_all = _extract_all_split_samples(
        model=model,
        data_loader=val_loader,
        device=device,
        normalizer=normalizer,
    )
    prediction = predictions_all[-1]
    ground_truth = ground_truth_all[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_base_name = f"{scenario_name}_{model_size_label}_val_all_steps"
    plot_base_name = f"{scenario_name}_{model_size_label}_val_final_step"

    npz_path = output_dir / f"{npz_base_name}.npz"
    np.savez_compressed(
        npz_path,
        predictions_all=predictions_all.numpy(),
        ground_truth_all=ground_truth_all.numpy(),
        abs_error_all=(predictions_all - ground_truth_all).abs().numpy(),
        prediction=prediction.numpy(),
        ground_truth=ground_truth.numpy(),
        abs_error=(prediction - ground_truth).abs().numpy(),
    )

    fig_path = output_dir / f"{plot_base_name}.png"
    _plot_validation_prediction_comparison(
        prediction=prediction,
        ground_truth=ground_truth,
        output_path=fig_path,
        scenario_name=scenario_name,
        split_label="val",
        model_size_label=model_size_label,
        hidden_channels=hidden_channels,
    )

    return npz_path, fig_path


def save_split_final_step_artifacts(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    normalizer,
    output_dir: Path,
    scenario_name: str,
    split_label: str,
    model_size_label: str,
    hidden_channels: int,
) -> tuple[Path, Path]:
    """Save final-step prediction artifacts for a specific split and scenario subset."""
    predictions_all, ground_truth_all = _extract_all_split_samples(
        model=model,
        data_loader=data_loader,
        device=device,
        normalizer=normalizer,
    )
    prediction = predictions_all[-1]
    ground_truth = ground_truth_all[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_base_name = f"{scenario_name}_{model_size_label}_{split_label}_all_steps"
    plot_base_name = f"{scenario_name}_{model_size_label}_{split_label}_final_step"

    npz_path = output_dir / f"{npz_base_name}.npz"
    np.savez_compressed(
        npz_path,
        predictions_all=predictions_all.numpy(),
        ground_truth_all=ground_truth_all.numpy(),
        abs_error_all=(predictions_all - ground_truth_all).abs().numpy(),
        prediction=prediction.numpy(),
        ground_truth=ground_truth.numpy(),
        abs_error=(prediction - ground_truth).abs().numpy(),
    )

    fig_path = output_dir / f"{plot_base_name}.png"
    _plot_validation_prediction_comparison(
        prediction=prediction,
        ground_truth=ground_truth,
        output_path=fig_path,
        scenario_name=scenario_name,
        split_label=split_label,
        model_size_label=model_size_label,
        hidden_channels=hidden_channels,
    )

    return npz_path, fig_path


def save_training_validation_loss_plot(
    train_l2_history: list[float],
    val_l2_history: list[float],
    output_dir: Path,
    scenario_name: str,
    model_size_label: str,
    hidden_channels: int,
) -> Path:
    """Save train/validation L2 loss curves across epochs for convergence inspection."""
    if len(train_l2_history) == 0 or len(val_l2_history) == 0:
        raise ValueError("Loss histories are empty; cannot create convergence plot")
    if len(train_l2_history) != len(val_l2_history):
        raise ValueError(
            "Train/validation history lengths do not match: "
            f"{len(train_l2_history)} vs {len(val_l2_history)}"
        )

    epochs = np.arange(1, len(train_l2_history) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(epochs, train_l2_history, label="Train L2", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_l2_history, label="Validation L2", linewidth=2.0, color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Loss")
    ax.set_title(
        (
            f"Convergence | Scenario: {scenario_name} | "
            f"size_preset={model_size_label} | hidden_channels={hidden_channels}"
        )
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{scenario_name}_{model_size_label}_loss_curve.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return fig_path
