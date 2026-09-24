from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    add_model_args,
    add_runtime_args,
    add_scenario_arg,
    add_scheduler_args,
    add_split_and_seed_args,
    add_training_args,
)


@dataclass(frozen=True)
class ModelSizeConfig:
    """Architecture configuration for one point in the sweep grid.

    Attributes
    ----------
    label : str
        Human-readable name (e.g. ``"small"``).
    hidden_channels : int
        Number of hidden channels in the FNO lifting, Fourier, and projection layers.
    n_modes_t : int
        Fourier modes along the **time** axis.
    n_modes_z : int
        Fourier modes along the **vertical** spatial axis.
    n_modes_x : int
        Fourier modes along the **horizontal** spatial axis.
    n_layers : int
        Number of Fourier integral operator layers.
    """

    label: str
    hidden_channels: int
    n_modes_t: int
    n_modes_z: int
    n_modes_x: int
    n_layers: int


# ---------------------------------------------------------------------------
# Model-size presets
# ---------------------------------------------------------------------------
# Grid:  T_in=25 (max usable modes ≤ 12), Z=20 (≤ 10), X=40 (≤ 20).
# Modes are scaled together with hidden_channels to keep compute balanced.

MODEL_SIZE_PRESETS: dict[str, ModelSizeConfig] = {
    "tiny":    ModelSizeConfig("tiny",    hidden_channels=4,  n_modes_t=4,  n_modes_z=4,  n_modes_x=4,  n_layers=4),
    "small":   ModelSizeConfig("small",   hidden_channels=8,  n_modes_t=4,  n_modes_z=6,  n_modes_x=8,  n_layers=4),
    "medium":  ModelSizeConfig("medium",  hidden_channels=16, n_modes_t=6,  n_modes_z=8,  n_modes_x=12, n_layers=6),
    "large":   ModelSizeConfig("large",   hidden_channels=32, n_modes_t=8,  n_modes_z=10, n_modes_x=16, n_layers=6),
    "huge":    ModelSizeConfig("huge",    hidden_channels=48, n_modes_t=10, n_modes_z=10, n_modes_x=20, n_layers=6),
    "massive": ModelSizeConfig("massive", hidden_channels=48, n_modes_t=12, n_modes_z=10, n_modes_x=20, n_layers=8),
}


SWEEP_CSV_FIELDNAMES = [
    "run_timestamp",
    "scenario_name",
    "model_size_label",
    "total_params",
    "rel_l2_error_concentration_train",
    "rel_l2_error_hydraulic_head_train",
    "rel_l2_error_concentration_val",
    "rel_l2_error_hydraulic_head_val",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the sweep entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train multiple 3-D FNO models across all Henry scenarios and append results to CSV",
    )

    add_scenario_arg(
        parser,
        required=False,
        default=Path(
            "/Users/akap5486/Projects/groundwater/data/simple_henry_data/"
            "grid_scenarios_random_skip2_20x40"
        ),
    )
    add_training_args(parser, default_epochs=100, default_batch_size=512)
    add_scheduler_args(parser, default_step_size=5, default_decay=0.98)
    add_split_and_seed_args(parser, default_train_ratio=0.7, default_seed=42)
    add_model_args(
        parser,
        default_n_modes_t=8,
        default_n_modes_z=8,
        default_n_modes_x=16,
        default_hidden_channels=32,
        default_n_layers=4,
    )
    add_runtime_args(parser, default_num_workers=0, default_device="auto")

    parser.add_argument(
        "--hidden-channels-list",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated hidden channel values (used when --sweep-mode=hidden)",
    )

    parser.add_argument(
        "--sweep-mode",
        type=str,
        choices=["hidden", "preset"],
        default="hidden",
        help="'hidden' sweeps hidden width at fixed modes; 'preset' uses coordinated model-size presets",
    )

    parser.add_argument(
        "--model-size-presets",
        type=str,
        default="tiny,small,medium,large,huge,massive",
        help="Comma-separated preset names for --sweep-mode=preset",
    )

    parser.set_defaults(normalize=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where sweep results (CSV, weights, predictions) are stored",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs during training",
    )

    return parser


def parse_hidden_channels(values: str) -> list[int]:
    """Parse a comma-separated list of hidden-channel counts."""
    parsed = [int(v.strip()) for v in values.split(",") if v.strip()]
    if not parsed:
        raise ValueError("--hidden-channels-list must contain at least one integer")
    return parsed


def parse_model_size_presets(values: str) -> list[ModelSizeConfig]:
    """Parse a comma-separated list of preset names into :class:`ModelSizeConfig` objects."""
    requested = [v.strip().lower() for v in values.split(",") if v.strip()]
    if not requested:
        raise ValueError("--model-size-presets must contain at least one preset name")

    configs: list[ModelSizeConfig] = []
    for name in requested:
        if name not in MODEL_SIZE_PRESETS:
            valid = ", ".join(sorted(MODEL_SIZE_PRESETS))
            raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
        configs.append(MODEL_SIZE_PRESETS[name])

    return configs
