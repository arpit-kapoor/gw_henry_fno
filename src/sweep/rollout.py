from __future__ import annotations

"""Autoregressive rollout inference for trained FNO models.

The FNO is trained on one-step windows: given fields at time t (plus optional
forcings/parameters), predict fields at time t+lag.  At evaluation time this
module replaces teacher-forcing with **autoregressive rollout**: the model's
own output is fed back as the next-step state while non-state channels
(boundary forcings and static parameters) are taken from the ground-truth
dataset inputs.

Channel classification is **fully data-driven**: channels whose names end with
``_t`` are treated as state fields that will be replaced by the model output at
each rollout step.  All other channels (time-varying forcings, static physical
parameters) are kept from the dataset.  This makes the code compatible with any
channel layout stored in ``windows.npz``:

- ``simple_henry`` (4 input channels):  ``concentration_t``, ``head_t``,
  ``beta_c``, ``diffc``  → state = channels 0–1, static = channels 2–3.
- ``henry_data`` with dynamic forcings (7+ channels): same state indices plus
  time-varying forcing channels 2–4 (and optional ``tidal_phase``).
"""

from pathlib import Path

import numpy as np
import torch

from src.data.henry_scenario_dataset import HenryScenarioDataset


# ---------------------------------------------------------------------------
# Channel index helpers
# ---------------------------------------------------------------------------


def channel_indices_from_names(
    channel_names: list[str],
) -> tuple[list[int], list[int]]:
    """Classify input channels as state or non-state by name convention.

    Channels whose names end with ``_t`` are state fields at time t; the FNO
    output provides those same fields at time t+lag and they are fed back at
    the next rollout step.  All other channels (forcings, static parameters)
    are taken from the dataset.

    Parameters
    ----------
    channel_names:
        Ordered list of input channel names as stored in ``windows.npz``
        under the key ``input_channel_names``.

    Returns
    -------
    state_indices:
        Indices of state channels (names ending in ``_t``).
    non_state_indices:
        Indices of all remaining channels.
    """
    state_indices = [i for i, name in enumerate(channel_names) if name.endswith("_t")]
    non_state_indices = [i for i in range(len(channel_names)) if i not in state_indices]
    return state_indices, non_state_indices


def _load_channel_names_from_run(run_dir: Path) -> list[str]:
    """Read ``input_channel_names`` from a run's ``windows.npz``.

    Raises
    ------
    KeyError
        If the NPZ file does not contain the ``input_channel_names`` key.
    """
    npz_path = run_dir / "windows.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        if "input_channel_names" not in data.files:
            raise KeyError(
                f"'input_channel_names' key not found in {npz_path}. "
                f"Available keys: {data.files}"
            )
        names = [str(n) for n in data["input_channel_names"]]
    return names


def resolve_channel_indices(
    dataset: HenryScenarioDataset,
) -> tuple[list[int], list[int], list[str]]:
    """Resolve state/non-state channel indices from a dataset's first run.

    Parameters
    ----------
    dataset:
        A :class:`HenryScenarioDataset` instance.  The channel names are read
        from the first run's ``windows.npz``.

    Returns
    -------
    state_indices, non_state_indices, channel_names
    """
    if len(dataset.run_refs) == 0:
        raise ValueError("Dataset has no run references; cannot resolve channel names.")
    _, first_run_dir = dataset.run_refs[0]
    channel_names = _load_channel_names_from_run(first_run_dir)
    state_indices, non_state_indices = channel_indices_from_names(channel_names)

    if not state_indices:
        raise ValueError(
            f"No state channels found in {channel_names}. "
            "State channels must have names ending in '_t'."
        )

    return state_indices, non_state_indices, channel_names


# ---------------------------------------------------------------------------
# Per-run rollout
# ---------------------------------------------------------------------------


def autoregressive_rollout(
    model: torch.nn.Module,
    run_inputs: torch.Tensor,
    run_outputs: torch.Tensor,
    state_channel_indices: list[int],
    device: torch.device,
    normalizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive rollout over a single run's ordered windows.

    Starting from the ground-truth state in the first window, the model's
    prediction is fed back as the state for every subsequent step.  Non-state
    channels (forcings, static parameters) are always taken from the
    ground-truth input at the corresponding step.

    Parameters
    ----------
    model:
        Trained FNO (or any ``nn.Module`` with the same interface).
    run_inputs:
        Input tensor for all windows in one run, shape ``(T, C_in, H, W)``.
        Expected to already be normalized when ``normalizer`` is not None.
    run_outputs:
        Ground-truth output tensor for all windows in one run,
        shape ``(T, C_out, H, W)``.  Expected to already be normalized.
    state_channel_indices:
        Indices of the input channels that are state fields (to be replaced
        by the model output at each step).
    device:
        Torch device to run inference on.
    normalizer:
        Optional :class:`~src.data.normalizer.Normalizer`.  When provided,
        predictions are denormalized for metric computation and then
        re-normalized before being injected as the next-step state.

    Returns
    -------
    rollout_preds:
        Predicted states in **denormalized** space, shape ``(T, C_out, H, W)``.
    ground_truth:
        Ground-truth outputs in **denormalized** space, shape ``(T, C_out, H, W)``.
    """
    T = run_inputs.shape[0]
    model.eval()

    rollout_preds_list: list[torch.Tensor] = []

    # Seed with the first window's ground-truth input (already normalized).
    x_current = run_inputs[0].clone().to(device)  # (C_in, H, W)

    with torch.no_grad():
        for i in range(T):
            pred = model(x_current.unsqueeze(0))  # (1, C_out, H, W)
            pred_squeezed = pred.squeeze(0)        # (C_out, H, W)

            # Denormalize prediction for metric accumulation.
            if normalizer is not None:
                pred_denorm = normalizer.denormalize_output(pred_squeezed.unsqueeze(0)).squeeze(0)
            else:
                pred_denorm = pred_squeezed

            rollout_preds_list.append(pred_denorm.detach().cpu())

            if i + 1 < T:
                # Build next input: start from ground-truth at step i+1
                # (gives correct forcings/static channels), then overwrite
                # the state channels with the model's prediction.
                x_next = run_inputs[i + 1].clone().to(device)  # (C_in, H, W)

                if normalizer is not None:
                    # Re-normalize the predicted state back into input space.
                    # output_mean/std[k] <-> input_mean/std[state_channel_indices[k]]
                    # because both refer to the same physical fields.
                    pred_denorm_dev = pred_denorm.to(device)
                    for out_ch, in_ch in enumerate(state_channel_indices):
                        ch_mean = normalizer.input_mean[in_ch].to(device)
                        ch_std = (normalizer.input_std[in_ch] + normalizer.epsilon).to(device)
                        x_next[in_ch] = (pred_denorm_dev[out_ch] - ch_mean) / ch_std
                else:
                    for out_ch, in_ch in enumerate(state_channel_indices):
                        x_next[in_ch] = pred_squeezed[out_ch]

                x_current = x_next

    rollout_preds = torch.stack(rollout_preds_list, dim=0)  # (T, C_out, H, W)

    # Denormalize ground-truth outputs for consistent metric computation.
    if normalizer is not None:
        gt_denorm = normalizer.denormalize_output(run_outputs.to(device)).detach().cpu()
    else:
        gt_denorm = run_outputs.cpu()

    return rollout_preds, gt_denorm


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------


def _collect_run_windows(
    dataset: HenryScenarioDataset,
    run_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all windows for a given run index in temporal order.

    The dataset ``__getitem__`` applies normalization on access, so the
    returned tensors are already normalized when ``dataset.normalizer`` is set.

    Parameters
    ----------
    dataset:
        A :class:`HenryScenarioDataset` instance.
    run_index:
        Index into ``dataset.run_refs``.

    Returns
    -------
    inputs:
        Shape ``(T, C_in, H, W)``.
    outputs:
        Shape ``(T, C_out, H, W)``.
    """
    scenario_idx, _ = dataset.run_refs[run_index]
    inputs_np, outputs_np = dataset._get_run_tensors(scenario_idx, run_index)

    inputs = torch.from_numpy(inputs_np).float()   # (T, C_in, H, W)
    outputs = torch.from_numpy(outputs_np).float()  # (T, C_out, H, W)

    # Apply normalization consistently with dataset.__getitem__.
    if dataset.normalizer is not None:
        inputs = dataset.normalizer.normalize_input(inputs)
        outputs = dataset.normalizer.normalize_output(outputs)

    return inputs, outputs


def _accumulate_step_metrics(
    rollout_preds: torch.Tensor,
    gt_denorm: torch.Tensor,
    mse_sum: torch.Tensor,
    l2_sum: torch.Tensor,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate per-channel MSE and relative L2 from one rollout trajectory.

    Parameters
    ----------
    rollout_preds, gt_denorm:
        Denormalized tensors, shape ``(T, C_out, H, W)``.
    mse_sum, l2_sum:
        Running per-channel accumulators, shape ``(C_out,)``.
    n_steps:
        Running step count.

    Returns
    -------
    Updated ``(mse_sum, l2_sum, n_steps)``.
    """
    sq_err = (rollout_preds - gt_denorm) ** 2               # (T, C_out, H, W)
    mse_per_step = sq_err.flatten(start_dim=2).mean(dim=-1)  # (T, C_out)
    mse_sum = mse_sum + mse_per_step.sum(dim=0).double()

    pred_flat = rollout_preds.flatten(start_dim=2)          # (T, C_out, H*W)
    gt_flat = gt_denorm.flatten(start_dim=2)
    diff_norm = torch.linalg.norm(pred_flat - gt_flat, dim=-1)  # (T, C_out)
    gt_norm = torch.linalg.norm(gt_flat, dim=-1)                # (T, C_out)
    rel_l2 = diff_norm / (gt_norm + 1e-12)                      # (T, C_out)
    l2_sum = l2_sum + rel_l2.sum(dim=0).double()

    return mse_sum, l2_sum, n_steps + rollout_preds.shape[0]


def _metrics_from_sums(
    mse_sum: torch.Tensor,
    l2_sum: torch.Tensor,
    n_steps: int,
    n_runs: int,
) -> dict[str, object]:
    """Compute final scalar metrics from accumulated per-channel sums."""
    mse_channels = (mse_sum / n_steps).tolist()
    l2_channels = (l2_sum / n_steps).tolist()
    return {
        "rollout_mse": float(sum(mse_channels) / len(mse_channels)),
        "rollout_l2": float(sum(l2_channels) / len(l2_channels)),
        "rollout_mse_channels": mse_channels,
        "rollout_l2_channels": l2_channels,
        "n_runs": n_runs,
        "n_steps_total": n_steps,
    }


def evaluate_rollout_metrics(
    model: torch.nn.Module,
    dataset: HenryScenarioDataset,
    device: torch.device,
    normalizer=None,
) -> dict[str, object]:
    """Evaluate autoregressive rollout metrics over all runs in a dataset.

    Iterates each run in ``dataset`` in temporal window order, performs a full
    autoregressive rollout (no teacher-forcing), and accumulates MSE and
    relative L2 error in denormalized space.

    Both **overall** and **per-scenario** breakdowns are computed in a single
    pass over all runs.

    Parameters
    ----------
    model:
        Trained FNO.
    dataset:
        A :class:`HenryScenarioDataset` instance (train or val split).
    device:
        Torch device.
    normalizer:
        Optional :class:`~src.data.normalizer.Normalizer` used during
        training.  Must match what was applied when creating ``dataset``.

    Returns
    -------
    dict with keys:

    Overall metrics
        - ``rollout_mse``           – scalar float, mean over all steps/runs.
        - ``rollout_l2``            – scalar float, mean relative L2.
        - ``rollout_mse_channels``  – list[float], per output channel.
        - ``rollout_l2_channels``   – list[float], per output channel.
        - ``n_runs``                – int, total runs evaluated.
        - ``n_steps_total``         – int, total rollout steps across all runs.
        - ``state_channel_indices`` – list[int].
        - ``channel_names``         – list[str].

    Per-scenario breakdown (``per_scenario``)
        A ``dict[str, dict]`` keyed by scenario name, where each inner dict
        contains the same metric keys as above.  Also includes
        ``first_run_index`` – the ``dataset.run_refs`` index of the first run
        in that scenario (useful for generating per-scenario artifacts).
    """
    state_indices, _, channel_names = resolve_channel_indices(dataset)

    n_runs = len(dataset.run_refs)
    if n_runs == 0:
        raise ValueError("Dataset has no runs; cannot evaluate rollout metrics.")

    # Build scenario -> run_index mapping in a single scan of run_refs.
    scenario_run_map: dict[str, list[int]] = {}
    for run_idx, (scenario_idx, _) in enumerate(dataset.run_refs):
        scenario_name = dataset.scenario_dirs[scenario_idx].name
        scenario_run_map.setdefault(scenario_name, []).append(run_idx)

    C_out: int | None = None
    total_mse_sum: torch.Tensor | None = None
    total_l2_sum: torch.Tensor | None = None
    total_steps = 0
    total_runs = 0

    # Per-scenario accumulators: [mse_sum, l2_sum, steps, runs]
    scenario_accum: dict[str, list] = {
        name: [None, None, 0, 0] for name in scenario_run_map
    }

    model.eval()

    for run_idx in range(n_runs):
        run_inputs, run_outputs = _collect_run_windows(dataset, run_idx)
        T = run_inputs.shape[0]
        if T == 0:
            continue

        rollout_preds, gt_denorm = autoregressive_rollout(
            model=model,
            run_inputs=run_inputs,
            run_outputs=run_outputs,
            state_channel_indices=state_indices,
            device=device,
            normalizer=normalizer,
        )
        # rollout_preds, gt_denorm: (T, C_out, H, W) in denormalized space.

        if C_out is None:
            C_out = rollout_preds.shape[1]
            total_mse_sum = torch.zeros(C_out, dtype=torch.float64)
            total_l2_sum = torch.zeros(C_out, dtype=torch.float64)

        # --- Overall accumulation ---
        total_mse_sum, total_l2_sum, total_steps = _accumulate_step_metrics(
            rollout_preds, gt_denorm, total_mse_sum, total_l2_sum, total_steps
        )
        total_runs += 1

        # --- Per-scenario accumulation ---
        scenario_idx_ref, _ = dataset.run_refs[run_idx]
        scenario_name = dataset.scenario_dirs[scenario_idx_ref].name
        s_mse, s_l2, s_steps, s_runs = scenario_accum[scenario_name]
        if s_mse is None:
            s_mse = torch.zeros(C_out, dtype=torch.float64)
            s_l2 = torch.zeros(C_out, dtype=torch.float64)
        s_mse, s_l2, s_steps = _accumulate_step_metrics(
            rollout_preds, gt_denorm, s_mse, s_l2, s_steps
        )
        scenario_accum[scenario_name] = [s_mse, s_l2, s_steps, s_runs + 1]

    if total_steps == 0:
        raise ValueError("All runs in dataset had zero windows; cannot compute rollout metrics.")

    # Build per-scenario metrics dict
    per_scenario: dict[str, dict[str, object]] = {}
    for scenario_name, (s_mse, s_l2, s_steps, s_runs) in scenario_accum.items():
        if s_steps == 0:
            per_scenario[scenario_name] = {
                "rollout_mse": None,
                "rollout_l2": None,
                "rollout_mse_channels": None,
                "rollout_l2_channels": None,
                "n_runs": 0,
                "n_steps_total": 0,
                "first_run_index": scenario_run_map[scenario_name][0],
            }
        else:
            s_metrics = _metrics_from_sums(s_mse, s_l2, s_steps, s_runs)
            s_metrics["first_run_index"] = scenario_run_map[scenario_name][0]
            per_scenario[scenario_name] = s_metrics

    overall = _metrics_from_sums(total_mse_sum, total_l2_sum, total_steps, total_runs)
    overall["state_channel_indices"] = state_indices
    overall["channel_names"] = channel_names
    overall["per_scenario"] = per_scenario
    return overall

