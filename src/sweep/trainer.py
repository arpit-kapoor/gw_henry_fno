from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from src.data.henry_scenario_dataset import create_henry_dataloaders
from src.neuralop import FNO
from src.neuralop.losses import LpLoss

from .metrics import evaluate_l2


@dataclass(frozen=True)
class TrainOneModelResult:
    model: torch.nn.Module
    train_loader: object
    val_loader: object
    normalizer: object
    train_loss_history: list[float]
    val_loss_history: list[float]
    total_params: int


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_model(
    *,
    scenarios_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    train_ratio: float,
    seed: int,
    validation_run_name: Optional[str],
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
) -> TrainOneModelResult:
    """Train one model configuration and return final sweep metrics."""
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}")

    # ------------------------------------------------------------------
    # Fix 1: pin_memory and multi-process data loading are only reliable
    # and effective with CUDA.  On MPS (Apple Silicon), pin_memory causes
    # silent data corruption because pinned CPU pages conflict with the
    # unified-memory model; fork-based worker processes can also return
    # stale or garbled batches on macOS.  Force both to their safe
    # defaults when the target device is not CUDA.
    # ------------------------------------------------------------------
    effective_pin_memory = pin_memory and device.type == "cuda"
    effective_num_workers = num_workers if device.type == "cuda" else 0

    dataloaders = create_henry_dataloaders(
        scenarios_dir=scenarios_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        num_workers=effective_num_workers,
        pin_memory=effective_pin_memory,
        normalize=normalize,
        validation_run_name=validation_run_name,
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

    total_params = count_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # Criterion returns the mean relative-L2 over the batch and channel
    # dimensions.  A separate sum-over-batch criterion is used for lossless
    # epoch-level accumulation so we never rely on the implicit
    # "mean × batch_size" cancellation (which silently breaks when the last
    # batch is smaller than batch_size).
    train_criterion = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")
    accum_criterion = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions=["sum", "mean"])

    scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
    if not disable_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_decay,
        )

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = train_criterion(pred, yb)
            loss.backward()
            optimizer.step()

            # ------------------------------------------------------------------
            # Fix 2: accumulate the batch's *sum* of per-sample relative-L2
            # values (mean over channels, sum over batch) so that dividing by
            # total_samples later gives the correct per-sample epoch mean
            # regardless of whether the final batch is a partial batch.
            # accum_criterion uses reductions=["sum", "mean"]: sum over batch
            # (dim 0), mean over channels (dim 1) → scalar batch-sum.
            # ------------------------------------------------------------------
            with torch.no_grad():
                running_loss += accum_criterion(pred, yb).item()
            total_samples += xb.size(0)

        epoch_train_l2 = running_loss / total_samples
        # Always evaluate global validation loss each epoch for convergence plotting.
        epoch_val_l2 = evaluate_l2(model, val_loader, device)
        train_loss_history.append(epoch_train_l2)
        val_loss_history.append(epoch_val_l2)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} - "
            f"hidden_channels: {hidden_channels}, "
            f"train_l2: {epoch_train_l2:.6f}, "
            f"val_l2: {epoch_val_l2:.6f}, "
            f"lr: {current_lr:.6e}"
        )

        if scheduler is not None:
            scheduler.step()

    return TrainOneModelResult(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        total_params=total_params,
    )
