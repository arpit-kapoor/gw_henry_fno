from __future__ import annotations

import torch

from src.neuralop.losses import LpLoss


def evaluate_l2(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> float:
    """Evaluate mean relative L2 loss across a dataloader using the training LpLoss setup.

    Computes the per-sample mean relative L2 in normalised (training) space, which
    matches the training objective.  Used for per-epoch convergence tracking only.
    """
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
