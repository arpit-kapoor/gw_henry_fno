from __future__ import annotations

import torch

from src.neuralop.losses import LpLoss


def _channelwise_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute per-sample, per-channel relative L2 with shape (B, C)."""
    pred_flat = pred.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)
    diff_norm = torch.linalg.norm(pred_flat - target_flat, dim=-1)
    target_norm = torch.linalg.norm(target_flat, dim=-1)
    return diff_norm / (target_norm + eps)


def _channelwise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-sample, per-channel MSE over spatial dimensions."""
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
                rel_l2_norm_sum = torch.zeros(channels, dtype=torch.float32, device=device)
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
