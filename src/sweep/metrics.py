from __future__ import annotations

import torch

from src.neuralop.losses import LpLoss


def evaluate_l2(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> float:
    """Evaluate mean relative L2 loss across a dataloader.

    Uses ``LpLoss(d=3)`` to match the 3-D (time × vertical × horizontal)
    training objective.  Computes the per-sample mean relative-L2 in
    normalised space for convergence monitoring.

    Parameters
    ----------
    model : torch.nn.Module
        Trained or partially trained FNO.
    dataloader :
        Validation (or training) DataLoader yielding ``(x, y)`` batches of
        shape ``(B, C, T, Z, X)``.
    device : torch.device
        Inference device.

    Returns
    -------
    float
        Mean per-sample relative L2 across all batches.
    """
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
