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
    label: str
    hidden_channels: int
    n_modes_x: int
    n_modes_y: int
    n_layers: int


MODEL_SIZE_PRESETS: dict[str, ModelSizeConfig] = {
    # Presets inspired by coordinated scaling used in PDE surrogate benchmarks.
    "tiny": ModelSizeConfig("tiny", hidden_channels=4, n_modes_x=4, n_modes_y=4, n_layers=4),
    "small": ModelSizeConfig("small", hidden_channels=8, n_modes_x=8, n_modes_y=8, n_layers=4),
    "medium": ModelSizeConfig("medium", hidden_channels=16, n_modes_x=8, n_modes_y=8, n_layers=6),
    "large": ModelSizeConfig("large", hidden_channels=32, n_modes_x=12, n_modes_y=12, n_layers=6),
    "huge": ModelSizeConfig("huge", hidden_channels=48, n_modes_x=16, n_modes_y=16, n_layers=6),
    "massive": ModelSizeConfig("massive", hidden_channels=48, n_modes_x=24, n_modes_y=24, n_layers=8),
}

# # Scaling factors for the model size presets.
# scaling_factors = {
#     "tiny": 0.25,
#     "small": 0.50,
#     "medium": 0.75,
#     "base": 1.00,
#     "large": 1.25,
#     "huge": 1.50,
#     "massive": 1.75,
# }

# # Base model size configuration.
# base_model_size_config = ModelSizeConfig("base", 
#                                          hidden_channels=32, 
#                                          n_modes_x=8, 
#                                          n_modes_y=16, 
#                                          n_layers=4)

##  Model size presets.
# MODEL_SIZE_PRESETS: dict[str, ModelSizeConfig] = {
#     name: ModelSizeConfig(
#         label=name,
#         hidden_channels=int(base_model_size_config.hidden_channels * scaling_factors[name]),
#         n_modes_x=int(base_model_size_config.n_modes_x * scaling_factors[name]),
#         n_modes_y=int(base_model_size_config.n_modes_y * scaling_factors[name]),
#         n_layers=int(base_model_size_config.n_layers * scaling_factors[name]),
#     )
#     for name in scaling_factors.keys()
# }


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
    parser = argparse.ArgumentParser(
        description="Train multiple FNO models across all scenarios and append results to CSV",
    )

    add_scenario_arg(
        parser,
        required=False,
        default=Path(
            "/Users/akap5486/Projects/groundwater/data/henry_data/all_scenarios/coupling_scenarios"
        ),
    )
    # Keep defaults aligned with current run_training.sh.
    add_training_args(parser, default_epochs=100, default_batch_size=512)
    add_scheduler_args(parser, default_step_size=5, default_decay=0.98)
    add_split_and_seed_args(parser, default_train_ratio=0.8, default_seed=42)
    add_model_args(
        parser,
        default_n_modes_x=8,
        default_n_modes_y=16,
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
        help="Sweep hidden width only or coordinated model-size presets",
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
        help="Directory where scenario-specific CSV is stored",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs during training (always validates on final epoch)",
    )

    return parser


def parse_hidden_channels(values: str) -> list[int]:
    # Accept a compact CLI form like "8,16,32,64,128".
    parsed = [int(v.strip()) for v in values.split(",") if v.strip()]
    if not parsed:
        raise ValueError("--hidden-channels-list must contain at least one integer")
    return parsed


def parse_model_size_presets(values: str) -> list[ModelSizeConfig]:
    requested = [v.strip().lower() for v in values.split(",") if v.strip()]
    if not requested:
        raise ValueError("--model-size-presets must contain at least one preset name")

    configs: list[ModelSizeConfig] = []
    for preset_name in requested:
        if preset_name not in MODEL_SIZE_PRESETS:
            valid = ", ".join(sorted(MODEL_SIZE_PRESETS))
            raise ValueError(f"Unknown model size preset '{preset_name}'. Valid presets: {valid}")
        configs.append(MODEL_SIZE_PRESETS[preset_name])

    return configs


def resolve_scenario_group(scenario_dir: Path) -> str:
    """Resolve scenario group from scenario_config.json."""
    config_path = scenario_dir / "scenario_config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    scenario_group = str(config["scenario_group"]).strip().lower()
    if not scenario_group:
        raise ValueError(f"scenario_group is empty in {config_path}")

    return scenario_group


def scenario_results_csv(results_dir: Path, scenario_dir: Path) -> Path:
    scenario_group = resolve_scenario_group(scenario_dir)
    scenario_name = scenario_dir.name
    return results_dir / f"{scenario_group}_{scenario_name}_fno_sweep_results.csv"
