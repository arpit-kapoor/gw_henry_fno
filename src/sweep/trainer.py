from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import datetime

from src.data.henry_scenario_dataset import create_henry_dataloaders
from src.neuralop import FNO
from src.neuralop.losses import LpLoss, RelCombinedNormLoss

from .metrics import evaluate_losses


@dataclass(frozen=True)
class TrainOneModelResult:
    """Return type of :func:`train_one_model`."""

    model: torch.nn.Module
    train_loader: object
    val_loader: object
    normalizer: object
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_rel_combined_norm_history: list[float]
    val_rel_combined_norm_history: list[float]
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
    dt: float = 1.0,
    dz: float = 0.05,
    dx: float = 0.05,
) -> TrainOneModelResult:
    """Train one 3-D FNO configuration and return results for the sweep."""
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

    sample_x, sample_y = next(iter(train_loader))
    in_channels = int(sample_x.shape[1])
    out_channels = int(sample_y.shape[1])

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

    # Use RelCombinedNormLoss for training (backprop)
    train_criterion = RelCombinedNormLoss(dt=dt, dz=dz, dx=dx)
    
    # Old LpLoss for logging
    # lploss_train_criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions="mean")
    lploss_accum_criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions=["sum", "mean"])

    scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
    if not disable_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_decay,
        )

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    train_rel_combined_norm_history: list[float] = []
    val_rel_combined_norm_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_norm_loss = 0.0
        running_lploss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            
            # Forward pass with new loss
            loss = train_criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            with torch.no_grad():
                running_norm_loss += loss.item() * batch_size
                running_lploss += lploss_accum_criterion(pred, yb).item() * batch_size
                
            total_samples += batch_size

        epoch_train_l2 = running_lploss / total_samples
        epoch_train_norm = running_norm_loss / total_samples
        
        epoch_val_l2, epoch_val_norm = evaluate_losses(model, val_loader, device, dt=dt, dz=dz, dx=dx)
        
        train_loss_history.append(epoch_train_l2)
        val_loss_history.append(epoch_val_l2)
        train_rel_combined_norm_history.append(epoch_train_norm)
        val_rel_combined_norm_history.append(epoch_val_norm)
        
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) >> "
            f"Epoch {epoch:03d}/{epochs} - "
            f"hidden_channels: {hidden_channels}, "
            f"train_norm: {epoch_train_norm:.6f}, val_norm: {epoch_val_norm:.6f}, "
            f"train_l2: {epoch_train_l2:.6f}, val_l2: {epoch_val_l2:.6f}, "
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
        train_rel_combined_norm_history=train_rel_combined_norm_history,
        val_rel_combined_norm_history=val_rel_combined_norm_history,
        total_params=total_params,
    )
