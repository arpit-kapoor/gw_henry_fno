from __future__ import annotations

"""Prediction collection for trained 3-D FNO models.

The 3-D FNO predicts the full output trajectory ``(C_out, T_out, Z, X)`` from
a single input ``(C_in, T_in, Z, X)`` in **one forward pass** — there is no
step-by-step autoregressive rollout.  This module collects per-scenario
predictions grouped by scenario name and computes per-channel relative L2
metrics.
"""

import numpy as np
import torch

from src.data.henry_scenario_dataset import HenryScenarioDataset


def collect_predictions_by_scenario(
    model: torch.nn.Module,
    dataset: HenryScenarioDataset,
    device: torch.device,
    normalizer=None,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect one-shot predictions grouped by scenario.

    For each run in the dataset, performs a single forward pass through the
    3-D FNO to obtain the full predicted output trajectory.  Results are
    grouped by scenario name and stacked into arrays of shape
    ``(N_runs, T_out, Z, X, C_out)``.

    Parameters
    ----------
    model : torch.nn.Module
        Trained 3-D FNO (eval mode is set internally).
    dataset : HenryScenarioDataset
        Train or validation split.
    device : torch.device
        Inference device.
    normalizer :
        Optional :class:`~src.data.normalizer.Normalizer` used during
        training.  Predictions and targets are denormalised before storage.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Mapping ``scenario_name -> {"targets": arr, "preds": arr}`` where each
        array has shape ``(N_runs, T_out, Z, X, C_out)``.
    """
    model.eval()

    # Group sample dataset-indices by scenario.
    scenario_sample_indices: dict[str, list[int]] = {}
    for sample_idx, ref in enumerate(dataset._sample_refs):
        scenario_name = dataset.scenario_dirs[ref.scenario_index].name
        scenario_sample_indices.setdefault(scenario_name, []).append(sample_idx)

    results: dict[str, dict[str, np.ndarray]] = {}

    for scenario_name, sample_indices in scenario_sample_indices.items():
        targets_list: list[np.ndarray] = []
        preds_list: list[np.ndarray] = []

        for sample_idx in sample_indices:
            x, y = dataset[sample_idx]
            # x: (C_in, T_in, Z, X), y: (C_out, T_out, Z, X) — normalised if normalizer is set.

            with torch.no_grad():
                x_dev = x.unsqueeze(0).to(device)          # (1, C_in, T_in, Z, X)
                pred_norm = model(x_dev).squeeze(0)         # (C_out, T_out, Z, X)

                if normalizer is not None:
                    pred_denorm = normalizer.denormalize_output(pred_norm.unsqueeze(0)).squeeze(0)
                    gt_denorm = normalizer.denormalize_output(y.unsqueeze(0).to(device)).squeeze(0)
                else:
                    pred_denorm = pred_norm
                    gt_denorm = y.to(device)

            # Permute to (T_out, Z, X, C_out) for storage.
            preds_np = pred_denorm.detach().cpu().permute(1, 2, 3, 0).numpy()    # (T, Z, X, C)
            targets_np = gt_denorm.detach().cpu().permute(1, 2, 3, 0).numpy()    # (T, Z, X, C)

            preds_list.append(preds_np)
            targets_list.append(targets_np)

        # Stack runs: list of (T, Z, X, C) → (N_runs, T, Z, X, C)
        results[scenario_name] = {
            "targets": np.stack(targets_list, axis=0),
            "preds":   np.stack(preds_list,   axis=0),
        }

    return results


def compute_rel_l2_per_channel(
    targets: np.ndarray,
    preds: np.ndarray,
) -> tuple[float, float]:
    """Compute per-channel relative L2 error averaged over all runs.

    Parameters
    ----------
    targets : np.ndarray
        Denormalised ground-truth array, shape ``(N, T, Z, X, C_out)``.
    preds : np.ndarray
        Denormalised predicted array, shape ``(N, T, Z, X, C_out)``.

    Returns
    -------
    rel_l2_ch0, rel_l2_ch1
        Scalar relative L2 error for channel 0 (concentration) and
        channel 1 (hydraulic head), averaged over all N runs.
    """
    N = targets.shape[0]
    C_out = targets.shape[-1]

    # Flatten all spatial/temporal dims: (N, T*Z*X, C_out)
    flat_targets = targets.reshape(N, -1, C_out)
    flat_preds = preds.reshape(N, -1, C_out)

    diff = flat_preds - flat_targets                         # (N, T*Z*X, C_out)
    diff_norm = np.linalg.norm(diff, axis=1)                 # (N, C_out)
    gt_norm = np.linalg.norm(flat_targets, axis=1)           # (N, C_out)
    rel_l2 = diff_norm / (gt_norm + 1e-12)                   # (N, C_out)

    mean_rel_l2 = rel_l2.mean(axis=0)                        # (C_out,)
    return float(mean_rel_l2[0]), float(mean_rel_l2[1])
