from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from src.data.henry_scenario_dataset import create_henry_dataloaders
from src.neuralop import FNO
from src.neuralop.losses import LpLoss

from .metrics import evaluate_channel_metrics, evaluate_l2


@dataclass(frozen=True)
class TrainOneModelResult:
    model: torch.nn.Module
    train_loader: object
    val_loader: object
    normalizer: object
    train_l2_history: list[float]
    val_l2_history: list[float]
    total_params: int
    final_train_l2: float
    final_val_l2: float
    final_train_mse: float
    final_val_mse: float
    out_channels: int
    train_rel_l2_norm_channels: str
    val_rel_l2_norm_channels: str
    train_rel_l2_denorm_channels: str
    val_rel_l2_denorm_channels: str
    train_mse_norm_channels: str
    val_mse_norm_channels: str
    train_mse_denorm_channels: str
    val_mse_denorm_channels: str


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
    evaluate_mse_fn: Callable,
) -> TrainOneModelResult:
    """Train one model configuration and return final sweep metrics."""
    if eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {eval_every}")

    dataloaders = create_henry_dataloaders(
        scenarios_dir=scenarios_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    train_criterion = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")

    scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
    if not disable_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_decay,
        )

    train_l2_history: list[float] = []
    val_l2_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = train_criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size_local = xb.size(0)
            running_loss += loss.item() * batch_size_local
            total_samples += batch_size_local

        epoch_train_l2 = running_loss / total_samples
        # Always evaluate global validation loss each epoch for convergence plotting.
        # This is distinct from per-scenario evaluation, which happens only at the end.
        epoch_val_l2 = evaluate_l2(model, val_loader, device)
        train_l2_history.append(epoch_train_l2)
        val_l2_history.append(epoch_val_l2)
        current_lr = optimizer.param_groups[0]["lr"]

        should_log = (epoch % eval_every == 0) or (epoch == epochs)
        if should_log:
            print(
                f"Epoch {epoch:03d}/{epochs} - "
                f"hidden_channels: {hidden_channels}, "
                f"train_l2: {epoch_train_l2:.6f}, "
                f"val_l2: {epoch_val_l2:.6f}, "
                f"lr: {current_lr:.6e}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{epochs} - "
                f"hidden_channels: {hidden_channels}, "
                f"train_l2: {epoch_train_l2:.6f}, "
                f"val_l2: {epoch_val_l2:.6f}, "
                f"lr: {current_lr:.6e}"
            )

        if scheduler is not None:
            scheduler.step()

    final_train_l2 = evaluate_l2(model, train_loader, device)
    final_val_l2 = evaluate_l2(model, val_loader, device)
    final_train_mse = evaluate_mse_fn(model, train_loader, device, normalizer)
    final_val_mse = evaluate_mse_fn(model, val_loader, device, normalizer)

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

    return TrainOneModelResult(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        train_l2_history=train_l2_history,
        val_l2_history=val_l2_history,
        total_params=total_params,
        final_train_l2=final_train_l2,
        final_val_l2=final_val_l2,
        final_train_mse=final_train_mse,
        final_val_mse=final_val_mse,
        out_channels=out_channels,
        train_rel_l2_norm_channels=json.dumps(train_channel_metrics["rel_l2_norm_channels"]),
        val_rel_l2_norm_channels=json.dumps(val_channel_metrics["rel_l2_norm_channels"]),
        train_rel_l2_denorm_channels=json.dumps(train_channel_metrics["rel_l2_denorm_channels"]),
        val_rel_l2_denorm_channels=json.dumps(val_channel_metrics["rel_l2_denorm_channels"]),
        train_mse_norm_channels=json.dumps(train_channel_metrics["mse_norm_channels"]),
        val_mse_norm_channels=json.dumps(val_channel_metrics["mse_norm_channels"]),
        train_mse_denorm_channels=json.dumps(train_channel_metrics["mse_denorm_channels"]),
        val_mse_denorm_channels=json.dumps(val_channel_metrics["mse_denorm_channels"]),
    )
