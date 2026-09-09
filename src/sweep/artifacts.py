from __future__ import annotations

"""Artifact-saving utilities for the FNO sweep.

Three responsibilities:
  1. save_model_weights     — persist model state_dict + arch config as a .pt file
  2. save_predictions_npz   — persist all prediction arrays for one (scenario, model) pair
  3. save_loss_history_json — persist train/val loss histories for offline plotting
"""

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def save_model_weights(
    model: torch.nn.Module,
    config,
    output_dir: Path,
) -> Path:
    """Save model state_dict and architecture config to a .pt file.

    The file is placed directly in ``output_dir/model_weights/{label}.pt`` so
    that different model sizes are differentiated by their label only (the same
    model is trained across all scenarios together, so no scenario prefix).

    Parameters
    ----------
    model:
        Trained FNO model.
    config:
        :class:`~src.sweep.config.ModelSizeConfig` dataclass instance.
    output_dir:
        Top-level results directory (``--results-dir``).

    Returns
    -------
    Path
        Path to the saved ``.pt`` file.
    """
    weights_dir = output_dir / "model_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / f"{config.label}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(config),
        },
        out_path,
    )
    return out_path


def save_predictions_npz(
    *,
    train_targets: Optional[np.ndarray],
    train_preds: Optional[np.ndarray],
    train_preds_rollout: Optional[np.ndarray],
    val_targets: Optional[np.ndarray],
    val_preds: Optional[np.ndarray],
    val_preds_rollout: Optional[np.ndarray],
    output_dir: Path,
    scenario_name: str,
    model_size_label: str,
) -> Path:
    """Save all prediction arrays for one (scenario, model) pair to a .npz file.

    All arrays must have shape ``(N, T, H, W, 2)`` where:
      - N = number of runs (train or val)
      - T = number of timesteps per run
      - H, W = spatial grid dimensions
      - 2 = output channels (concentration, hydraulic head)

    Parameters
    ----------
    train_targets, train_preds, train_preds_rollout:
        Arrays for the training split.
    val_targets, val_preds, val_preds_rollout:
        Arrays for the validation split.
    output_dir:
        Top-level results directory (``--results-dir``).
    scenario_name:
        Scenario name used in the file name.
    model_size_label:
        Model size label used in the file name.

    Returns
    -------
    Path
        Path to the saved ``.npz`` file.
    """
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    out_path = preds_dir / f"{scenario_name}_{model_size_label}.npz"
    np.savez_compressed(
        out_path,
        train_targets=train_targets,
        train_preds=train_preds,
        train_preds_rollout=train_preds_rollout,
        val_targets=val_targets,
        val_preds=val_preds,
        val_preds_rollout=val_preds_rollout,
    )
    return out_path


def save_loss_history_json(
    *,
    train_loss_history: list[float],
    val_loss_history: list[float],
    model_size_label: str,
    output_dir: Path,
) -> Path:
    """Save per-epoch train/val loss histories to a JSON file for offline plotting.

    The histories reflect **one-step-ahead losses** in normalised space (the
    training objective), logged every epoch.  They are intended for comparing
    overfitting across model sizes by loading and plotting the JSON files.

    File is placed at ``output_dir/figures/{model_size_label}_loss_history.json``.

    Parameters
    ----------
    train_loss_history:
        Per-epoch training loss (relative L2, normalised space).
    val_loss_history:
        Per-epoch validation loss (relative L2, normalised space).
    model_size_label:
        Used in the file name.
    output_dir:
        Top-level results directory (``--results-dir``).

    Returns
    -------
    Path
        Path to the saved ``.json`` file.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{model_size_label}_loss_history.json"
    payload = {
        "model_size_label": model_size_label,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path
