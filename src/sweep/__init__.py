"""Utilities for FNO sweep training, evaluation, and artifact generation."""

from .config import (
    MODEL_SIZE_PRESETS,
    SWEEP_CSV_FIELDNAMES,
    ModelSizeConfig,
    build_parser,
    parse_hidden_channels,
    parse_model_size_presets,
    scenario_results_csv,
)
from .results import append_result_row
from .trainer import TrainOneModelResult, train_one_model
from .artifacts import (
    save_loss_history_json,
    save_model_weights,
    save_predictions_npz,
)
from .rollout import (
    autoregressive_rollout,
    channel_indices_from_names,
    collect_predictions_by_scenario,
    compute_rollout_rel_l2_per_channel,
    evaluate_rollout_metrics,
    resolve_channel_indices,
)

__all__ = [
    "MODEL_SIZE_PRESETS",
    "SWEEP_CSV_FIELDNAMES",
    "ModelSizeConfig",
    "TrainOneModelResult",
    "append_result_row",
    "autoregressive_rollout",
    "build_parser",
    "channel_indices_from_names",
    "collect_predictions_by_scenario",
    "compute_rollout_rel_l2_per_channel",
    "evaluate_rollout_metrics",
    "parse_hidden_channels",
    "parse_model_size_presets",
    "resolve_channel_indices",
    "save_loss_history_json",
    "save_model_weights",
    "save_predictions_npz",
    "scenario_results_csv",
    "train_one_model",
]
