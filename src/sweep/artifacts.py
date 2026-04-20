from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _extract_final_validation_sample(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    normalizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return denormalized prediction/target for one validation sample."""
    dataset = val_loader.dataset
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty; cannot create final-step artifacts")

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
