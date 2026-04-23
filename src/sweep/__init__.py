"""Utilities for FNO sweep training, evaluation, and artifact generation."""

from .config import (
    MODEL_SIZE_PRESETS,
    SWEEP_PER_SCENARIO_RESULT_FIELDNAMES,
    SWEEP_RESULT_FIELDNAMES,
    ModelSizeConfig,
    build_parser,
    parse_hidden_channels,
    parse_model_size_presets,
    scenario_results_csv,
)
from .results import append_result_row
from .trainer import TrainOneModelResult, train_one_model
from .artifacts import (
    save_split_final_step_artifacts,
    save_training_validation_loss_plot,
    save_validation_final_step_artifacts,
)

__all__ = [
    "MODEL_SIZE_PRESETS",
    "SWEEP_PER_SCENARIO_RESULT_FIELDNAMES",
    "SWEEP_RESULT_FIELDNAMES",
    "ModelSizeConfig",
    "TrainOneModelResult",
    "append_result_row",
    "build_parser",
    "parse_hidden_channels",
    "parse_model_size_presets",
    "save_split_final_step_artifacts",
    "save_training_validation_loss_plot",
    "save_validation_final_step_artifacts",
    "scenario_results_csv",
    "train_one_model",
]
