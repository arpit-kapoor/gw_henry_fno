from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.data.henry_scenario_dataset import create_henry_dataloaders
from src.neuralop import FNO
from src.neuralop.losses import LpLoss

from .metrics import evaluate_l2


@dataclass(frozen=True)
class TrainOneModelResult:
    """Return type of :func:`train_one_model`."""

    model: torch.nn.Module
    train_loader: object
    val_loader: object
    normalizer: object
    train_loss_history: list[float]
    val_loss_history: list[float]
    total_params: int


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
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
    device: torch.device,
    n_modes_t: int,
    n_modes_z: int,
    n_modes_x: int,
    hidden_channels: int,
    n_layers: int,
    num_workers: int,
    pin_memory: bool,
    normalize: bool,
    disable_scheduler: bool,
    scheduler_step_size: int,
    scheduler_decay: float,
) -> TrainOneModelResult:
    """Train one 3-D FNO configuration and return results for the sweep.

    The FNO is instantiated with ``n_modes=(n_modes_t, n_modes_z, n_modes_x)``
    so that spectral convolutions operate over the full (time, vertical,
    horizontal) volume simultaneously.  Each training sample is a complete run
    of shape ``(C_in, T_in, Z, X)``; the model predicts
    ``(C_out, T_out, Z, X)`` in a single forward pass.

    Parameters
    ----------
    scenarios_dir : Path
        Parent directory of ``scenario_NNN/scenario.npz`` subdirs.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for both loaders.
    learning_rate : float
        AdamW initial learning rate.
    weight_decay : float
        AdamW weight decay.
    eval_every : int
        Validate every *N* epochs (must be > 0).
    train_ratio : float
        Fraction of runs used for training (shared across scenarios).
    seed : int
        RNG seed for data split and training initialisation.
    device : torch.device
        Training device.
    n_modes_t, n_modes_z, n_modes_x : int
        Fourier modes along time, vertical, and horizontal axes respectively.
    hidden_channels : int
        Hidden channel width in the FNO.
    n_layers : int
        Number of Fourier integral operator layers.
    num_workers : int
        DataLoader worker processes.
    pin_memory : bool
        Enable pinned memory (effective only with CUDA).
    normalize : bool
        Apply per-channel mean/std normalisation.
    disable_scheduler : bool
        Skip learning-rate scheduling when True.
    scheduler_step_size : int
        StepLR step size in epochs.
    scheduler_decay : float
        StepLR gamma (multiplicative decay factor).
    """
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}")

    # pin_memory and multi-process loading are only reliable with CUDA.
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
    )

    if normalize:
        train_loader, val_loader, normalizer = dataloaders
    else:
        train_loader, val_loader = dataloaders
        normalizer = None

    # Infer channel counts from a single batch.
    sample_x, sample_y = next(iter(train_loader))
    in_channels = int(sample_x.shape[1])
    out_channels = int(sample_y.shape[1])

    # Build 3-D FNO: n_modes is a 3-tuple → 3-D spectral convolutions,
    # Conv3d skip connections, and 3-D lifting/projection MLPs.
    model = FNO(
        n_modes=(n_modes_t, n_modes_z, n_modes_x),
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

    # LpLoss with d=3 for the 3-D (T, Z, X) domain.
    # train_criterion returns the mean relative-L2 over the batch.
    # accum_criterion sums over batch for correct per-sample epoch aggregation
    # even when the final batch is smaller than batch_size.
    train_criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions="mean")
    accum_criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions=["sum", "mean"])

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

            with torch.no_grad():
                running_loss += accum_criterion(pred, yb).item()
            total_samples += xb.size(0)

        epoch_train_l2 = running_loss / total_samples
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
