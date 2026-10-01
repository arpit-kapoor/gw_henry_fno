"""Train a single 3-D FNO model across Henry scenarios."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import datetime as dt

from src.config import parse_args

from src.data.henry_scenario_dataset import create_henry_dataloaders
from src.data.normalizer import Normalizer
from src.neuralop.losses import LpLoss
from src.neuralop import FNO


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve requested runtime device into a concrete torch device."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' is not available")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    # Auto mode priority: CUDA -> MPS -> CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_mse(
    model: nn.Module,
    dataloader,
    device: torch.device,
    normalizer: Optional[Normalizer] = None,
) -> float:
    """Evaluate model MSE on a dataloader in denormalised space.

    Parameters
    ----------
    model : nn.Module
        Trained 3-D FNO.
    dataloader :
        DataLoader yielding ``(x, y)`` batches of shape ``(B, C, T, Z, X)``.
    device : torch.device
        Inference device.
    normalizer : Normalizer, optional
        If provided, predictions and targets are denormalised before MSE
        computation so metrics are reported in the original data scale.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            pred = model(xb)

            if normalizer is not None:
                pred = normalizer.denormalize_output(pred)
                yb = normalizer.denormalize_output(yb)

            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

    if total_samples == 0:
        raise ValueError("Dataloader produced zero samples during evaluation")

    return total_loss / total_samples


def main() -> None:
    """Run end-to-end 3-D FNO training and print final metrics."""
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    dataloaders = create_henry_dataloaders(
        scenarios_dir=args.scenario_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        normalize=args.normalize,
    )

    if args.normalize:
        train_loader, val_loader, normalizer = dataloaders
    else:
        train_loader, val_loader = dataloaders
        normalizer = None

    sample_x, sample_y = next(iter(train_loader))
    in_channels = int(sample_x.shape[1])
    out_channels = int(sample_y.shape[1])

    # 3-D FNO: n_modes is a 3-tuple → spectral convolutions over (T, Z, X).
    model = FNO(
        n_modes=(args.n_modes_t, args.n_modes_z, args.n_modes_x),
        hidden_channels=args.hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_criterion = LpLoss(d=3, p=2, reduce_dims=[0, 1], reductions="mean")

    scheduler = None
    if not args.disable_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_decay,
        )

    for epoch in range(1, args.epochs + 1):
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

            running_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

        epoch_train_l2 = running_loss / total_samples
        epoch_val_mse = evaluate_mse(model, val_loader, device, normalizer)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"({dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) :: "
            f"Epoch {epoch:03d}/{args.epochs} - "
            f"train_l2: {epoch_train_l2:.6f}, "
            f"val_mse: {epoch_val_mse:.6f}, "
            f"lr: {current_lr:.6e}"
        )

        if scheduler is not None:
            scheduler.step()

    final_train_mse = evaluate_mse(model, train_loader, device, normalizer)
    final_val_mse = evaluate_mse(model, val_loader, device, normalizer)

    print("\nFinal metrics")
    print(f"scenarios_dir: {args.scenario_dir}")
    print(f"device: {device}")
    print(f"normalize: {args.normalize}")
    print(f"n_modes: (t={args.n_modes_t}, z={args.n_modes_z}, x={args.n_modes_x})")
    print(f"scheduler_enabled: {not args.disable_scheduler}")
    if not args.disable_scheduler:
        print(f"scheduler_step_size: {args.scheduler_step_size}")
        print(f"scheduler_decay: {args.scheduler_decay}")
    if normalizer is not None:
        print(f"input_mean: {normalizer.input_mean}")
        print(f"input_std: {normalizer.input_std}")
        print(f"output_mean: {normalizer.output_mean}")
        print(f"output_std: {normalizer.output_std}")
    print(f"train_mse: {final_train_mse:.6f}")
    print(f"val_mse: {final_val_mse:.6f}")


if __name__ == "__main__":
    main()
