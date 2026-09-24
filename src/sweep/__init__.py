"""Utilities for FNO sweep training, evaluation, and artifact generation."""

from .config import (
    MODEL_SIZE_PRESETS,
    SWEEP_CSV_FIELDNAMES,
    ModelSizeConfig,
    build_parser,
    parse_hidden_channels,
    parse_model_size_presets,
)
from .results import append_result_row
from .trainer import TrainOneModelResult, train_one_model
from .artifacts import (
    save_loss_history_json,
    save_model_weights,
    save_predictions_npz,
)
from .rollout import (
    collect_predictions_by_scenario,
    compute_rel_l2_per_channel,
)

__all__ = [
    "MODEL_SIZE_PRESETS",
    "SWEEP_CSV_FIELDNAMES",
    "ModelSizeConfig",
    "TrainOneModelResult",
    "append_result_row",
    "build_parser",
    "collect_predictions_by_scenario",
    "compute_rel_l2_per_channel",
    "parse_hidden_channels",
    "parse_model_size_presets",
    "save_loss_history_json",
    "save_model_weights",
    "save_predictions_npz",
    "train_one_model",
]
