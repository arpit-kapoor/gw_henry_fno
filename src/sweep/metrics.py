from __future__ import annotations

import torch

from src.neuralop.losses import LpLoss, RelCombinedNormLoss


def evaluate_losses(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    dt: float = 1.0,
    dz: float = 0.05,
    dx: float = 0.05,
    micro_batch_size: int = 0,
) -> tuple[float, float]:
    """Evaluate mean relative L2 loss and RelCombinedNormLoss across a dataloader.

    If ``micro_batch_size`` > 0, each batch is evaluated in chunks of that size
    to bound peak memory; both losses are per-sample means, so the result is
    unchanged.

    Returns
    -------
    float, float
        Mean per-sample relative L2, and Mean per-sample RelCombinedNormLoss across all batches.
    """
    model.eval()
    total_l2_loss = 0.0
    rel_combined_norm_loss = 0.0
    total_samples = 0
    lploss_criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions="mean")
    rel_combined_norm_criterion = RelCombinedNormLoss(dt=dt, dz=dz, dx=dx)

    with torch.no_grad():
        for xb_full, yb_full in dataloader:
            chunk_size = micro_batch_size if micro_batch_size > 0 else xb_full.size(0)
            for xb, yb in zip(xb_full.split(chunk_size), yb_full.split(chunk_size)):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)

                loss_l2 = lploss_criterion(pred, yb)
                loss_norm = rel_combined_norm_criterion(pred, yb)

                total_l2_loss += loss_l2.item() * xb.size(0)
                rel_combined_norm_loss += loss_norm.item() * xb.size(0)
                total_samples += xb.size(0)

    if total_samples == 0:
        raise ValueError("Dataloader produced zero samples during evaluation")

    return total_l2_loss / total_samples, rel_combined_norm_loss / total_samples

def evaluate_l2(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> float:
    # Keeping this for backwards compatibility if used elsewhere
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions="mean")

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

    if total_samples == 0:
        raise ValueError("Dataloader produced zero samples during L2 evaluation")

    return total_loss / total_samples
