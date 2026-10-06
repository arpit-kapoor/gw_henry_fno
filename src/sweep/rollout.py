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
    batch_size: int = 64,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect one-shot predictions grouped by scenario.

    For each run in the dataset, performs a single forward pass through the
    3-D FNO to obtain the full predicted output trajectory.  Runs are pushed
    through the model ``batch_size`` at a time (in the main process, no
    DataLoader workers).  Results are grouped by scenario name and stacked
    into arrays of shape ``(N_runs, T_out, Z, X, C_out)``.

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
    batch_size : int, optional
        Number of runs per forward pass (default 64).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Mapping ``scenario_name -> {"targets": arr, "preds": arr}`` where each
        array has shape ``(N_runs, T_out, Z, X, C_out)``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

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

        for start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[start:start + batch_size]
            samples = [dataset[i] for i in batch_indices]
            # x: (B, C_in, T_in, Z, X), y: (B, C_out, T_out, Z, X) — normalised if normalizer is set.
            x = torch.stack([s[0] for s in samples])
            y = torch.stack([s[1] for s in samples])

            with torch.no_grad():
                x_dev = x.to(device)
                y_dev = y.to(device)
                pred_norm = model(x_dev)                    # (B, C_out, T_out, Z, X)

                if normalizer is not None:
                    pred_denorm = normalizer.denormalize_output(pred_norm)
                    gt_denorm = normalizer.denormalize_output(y_dev)
                else:
                    pred_denorm = pred_norm
                    gt_denorm = y_dev

            # Permute to (B, T_out, Z, X, C_out) for storage.
            preds_list.append(pred_denorm.detach().cpu().permute(0, 2, 3, 4, 1).numpy())
            targets_list.append(gt_denorm.detach().cpu().permute(0, 2, 3, 4, 1).numpy())

        # Concatenate batches: list of (B, T, Z, X, C) → (N_runs, T, Z, X, C)
        results[scenario_name] = {
            "targets": np.concatenate(targets_list, axis=0),
            "preds":   np.concatenate(preds_list,   axis=0),
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

def compute_linf_l2_norm(u, dx, dz, dt):
    area = dz*dx

    # Squared Norm
    l2_sq_u_space = np.sum(u**2, axis=(2, 3)) * area 
    l_inf_l2_sq_u = np.max(l2_sq_u_space, axis=1)  

    # Sum over time of spatial gradient L2 norm squared
    # Compute spatial gradients over Z (axis 2) and X (axis 3) using actual spacing
    grad_u_z, grad_u_x = np.gradient(u, dz, dx, axis=(2, 3))
    grad_u_l2_sq_space = np.sum(grad_u_z**2 + grad_u_x**2, axis=(2, 3)) * area 
    grad_u_l2_l2_sq = np.sum(grad_u_l2_sq_space, axis=1) * dt

    # Final u norm for each run
    norm_u = np.sqrt(l_inf_l2_sq_u + grad_u_l2_l2_sq) 

    return norm_u

def compute_l2_norm(u, dx, dz, dt):
    area = dz*dx

    # Spatial L2 norm squared at each time step
    l2_sq_u_space = np.sum(u**2, axis=(2, 3)) * area 

    # Spatial gradient L2 norm squared at each time step
    grad_u_z, grad_u_x = np.gradient(u, dz, dx, axis=(2, 3))
    grad_u_l2_sq_space = np.sum(grad_u_z**2 + grad_u_x**2, axis=(2, 3)) * area 

    # ||p(t)||_{H^1}^2 = L2^2 + \nabla L2^2
    h1_sq_u = l2_sq_u_space + grad_u_l2_sq_space 

    # Final p norm for each run: Max over time of the H^1 norm
    norm_u = np.max(np.sqrt(h1_sq_u), axis=1)

    return norm_u

def compute_rel_combined_norm_error(targets: np.ndarray, preds: np.ndarray, dt: float=1.0, dz: float=0.05, dx: float=0.05) -> float:
    """Compute physics-informed relative combined norm error averaged over all runs."""
    # Compute area of the domain
    area = dz * dx

    # Residual
    diff = preds - targets
    
    # Assuming Channel 0 = Concentration (C), Channel 1 = Pressure/Head (p)
    diff_c = diff[..., 0]  # shape: (N, T, Z, X)
    diff_h = diff[..., 1]  # shape: (N, T, Z, X)

    # =====================================================================
    # 1. Concentration Norm: ||C||_{\mathcal{C}_T}
    # =====================================================================
    norm_c_diff = compute_linf_l2_norm(diff_c, dx, dz, dt)
    norm_c_true = compute_linf_l2_norm(targets[...,0], dx, dz, dt)
    norm_c = norm_c_diff / (norm_c_true + 1e-12)

    # =====================================================================
    # 2. Pressure/Head Norm: ||p||_{L^\infty H^1}
    # =====================================================================
    norm_h_diff = compute_l2_norm(diff_h, dx, dz, dt)
    norm_h_true = compute_l2_norm(targets[...,1], dx, dz, dt)
    norm_h = norm_h_diff / (norm_h_true + 1e-12)
    
    # =====================================================================
    # 3. Combined Total Metric: ||(C, p)||_{\mathcal{Z}_T}
    # =====================================================================
    rel_combined_norm = norm_c + norm_h 

    return float(np.mean(rel_combined_norm))
