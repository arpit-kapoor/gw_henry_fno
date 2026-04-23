from __future__ import annotations

"""CLI entrypoint for running FNO architecture sweeps."""

import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
from src.config import validate_common_args
from src.sweep import (
    SWEEP_PER_SCENARIO_RESULT_FIELDNAMES,
    SWEEP_RESULT_FIELDNAMES,
    ModelSizeConfig,
    append_result_row,
    build_parser,
    parse_hidden_channels,
    parse_model_size_presets,
    save_split_final_step_artifacts,
    save_training_validation_loss_plot,
    save_validation_final_step_artifacts,
    train_one_model,
)
from src.sweep.metrics import evaluate_l2
from train_fno import evaluate_mse, resolve_device, set_seed


def _build_subset_loader(dataset, indices: list[int], batch_size: int, pin_memory: bool) -> DataLoader:
    """Build a lightweight subset loader for per-scenario evaluation/artifacts.

    These subsets are small, so using worker processes adds startup overhead.
    """
    if len(indices) == 0:
        raise ValueError("Cannot create subset loader with zero indices")

    return DataLoader(
        Subset(dataset, indices),
        batch_size=min(batch_size, len(indices)),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


def main() -> None:
    """Parse sweep CLI arguments and orchestrate multi-model training."""
    # Thin CLI entrypoint that delegates core work to src/sweep modules.
    parser = build_parser()
    args = parser.parse_args()

    validate_common_args(parser, args)

    if args.sweep_mode == "preset":
        sweep_configs = parse_model_size_presets(args.model_size_presets)
    else:
        hidden_channels_values = parse_hidden_channels(args.hidden_channels_list)
        sweep_configs = [
            ModelSizeConfig(
                label=f"hidden_{hidden_channels}",
                hidden_channels=hidden_channels,
                n_modes_x=args.n_modes_x,
                n_modes_y=args.n_modes_y,
                n_layers=args.n_layers,
            )
            for hidden_channels in hidden_channels_values
        ]

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        # Input shapes are fixed, so autotuning usually improves conv throughput.
        torch.backends.cudnn.benchmark = True

    scenarios_dir = args.scenario_dir
    validation_tag = args.validation_run_name if args.validation_run_name is not None else "random_split"
    csv_path = args.results_dir / f"{scenarios_dir.name}_{validation_tag}_fno_sweep_results.csv"
    per_scenario_csv_path = args.results_dir / f"{scenarios_dir.name}_{validation_tag}_fno_sweep_per_scenario_results.csv"
    artifact_dir = args.results_dir / args.artifact_dir_name / scenarios_dir.name / validation_tag
    fieldnames = SWEEP_RESULT_FIELDNAMES
    per_scenario_fieldnames = SWEEP_PER_SCENARIO_RESULT_FIELDNAMES

    print("Starting multi-model FNO sweep")
    print(f"scenarios_dir: {scenarios_dir}")
    print(f"validation_run_name: {args.validation_run_name}")
    print(f"sweep_mode: {args.sweep_mode}")
    if args.sweep_mode == "preset":
        print(f"model_size_presets: {[cfg.label for cfg in sweep_configs]}")
    else:
        print(f"hidden_channels_list: {[cfg.hidden_channels for cfg in sweep_configs]}")
        print(f"fixed architecture: n_modes=({args.n_modes_x}, {args.n_modes_y}), n_layers={args.n_layers}")
    print(f"results csv: {csv_path}")
    print(f"per-scenario results csv: {per_scenario_csv_path}")
    print(f"validation artifact dir: {artifact_dir}")

    for config in sweep_configs:
        # Re-seed per configuration so initialization and shuffled batch order
        # do not depend on loop position.
        model_seed = (
            args.seed
            + config.hidden_channels
            + config.n_modes_x
            + config.n_modes_y
            + 10 * config.n_layers
        )
        set_seed(model_seed)

        print("=" * 60)
        print(
            "Training model config "
            f"label={config.label}, hidden_channels={config.hidden_channels}, "
            f"n_modes=({config.n_modes_x}, {config.n_modes_y}), n_layers={config.n_layers}"
        )
        print(f"model_seed: {model_seed}")
        print("=" * 60)

        result = train_one_model(
            scenarios_dir=scenarios_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            train_ratio=args.train_ratio,
            seed=args.seed,
            validation_run_name=args.validation_run_name,
            device=device,
            n_modes_x=config.n_modes_x,
            n_modes_y=config.n_modes_y,
            hidden_channels=config.hidden_channels,
            n_layers=config.n_layers,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            normalize=args.normalize,
            disable_scheduler=args.disable_scheduler,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_decay=args.scheduler_decay,
            evaluate_mse_fn=evaluate_mse,
        )

        npz_artifact_path, plot_artifact_path = save_validation_final_step_artifacts(
            model=result.model,
            val_loader=result.val_loader,
            device=device,
            normalizer=result.normalizer,
            output_dir=artifact_dir,
            scenario_name=f"{scenarios_dir.name}_{validation_tag}",
            model_size_label=config.label,
            hidden_channels=config.hidden_channels,
        )

        loss_curve_plot_path = save_training_validation_loss_plot(
            train_l2_history=result.train_l2_history,
            val_l2_history=result.val_l2_history,
            output_dir=artifact_dir,
            scenario_name=f"{scenarios_dir.name}_{validation_tag}",
            model_size_label=config.label,
            hidden_channels=config.hidden_channels,
        )

        per_scenario_train_metrics: dict[str, dict[str, float]] = {}
        per_scenario_val_metrics: dict[str, dict[str, float]] = {}
        per_scenario_train_plots: dict[str, str] = {}
        per_scenario_val_plots: dict[str, str] = {}

        train_dataset = result.train_loader.dataset
        val_dataset = result.val_loader.dataset
        scenario_names = train_dataset.scenario_names

        for scenario_name in scenario_names:
            train_indices = train_dataset.sample_indices_for_scenario(scenario_name)
            if len(train_indices) > 0:
                train_subset_loader = _build_subset_loader(
                    train_dataset,
                    train_indices,
                    args.batch_size,
                    args.pin_memory,
                )
                scenario_train_l2 = evaluate_l2(result.model, train_subset_loader, device)
                scenario_train_mse = evaluate_mse(result.model, train_subset_loader, device, result.normalizer)
                per_scenario_train_metrics[scenario_name] = {
                    "train_l2": scenario_train_l2,
                    "train_mse": scenario_train_mse,
                }
                _, scenario_train_plot = save_split_final_step_artifacts(
                    model=result.model,
                    data_loader=train_subset_loader,
                    device=device,
                    normalizer=result.normalizer,
                    output_dir=artifact_dir / "per_scenario",
                    scenario_name=scenario_name,
                    split_label="train",
                    model_size_label=config.label,
                    hidden_channels=config.hidden_channels,
                )
                per_scenario_train_plots[scenario_name] = str(scenario_train_plot)

            val_indices = val_dataset.sample_indices_for_scenario(scenario_name)
            if len(val_indices) > 0:
                val_subset_loader = _build_subset_loader(
                    val_dataset,
                    val_indices,
                    args.batch_size,
                    args.pin_memory,
                )
                scenario_val_l2 = evaluate_l2(result.model, val_subset_loader, device)
                scenario_val_mse = evaluate_mse(result.model, val_subset_loader, device, result.normalizer)
                per_scenario_val_metrics[scenario_name] = {
                    "val_l2": scenario_val_l2,
                    "val_mse": scenario_val_mse,
                }
                _, scenario_val_plot = save_split_final_step_artifacts(
                    model=result.model,
                    data_loader=val_subset_loader,
                    device=device,
                    normalizer=result.normalizer,
                    output_dir=artifact_dir / "per_scenario",
                    scenario_name=scenario_name,
                    split_label="val",
                    model_size_label=config.label,
                    hidden_channels=config.hidden_channels,
                )
                per_scenario_val_plots[scenario_name] = str(scenario_val_plot)

        for scenario_name in scenario_names:
            train_msg = per_scenario_train_metrics.get(scenario_name)
            val_msg = per_scenario_val_metrics.get(scenario_name)
            print(
                f"scenario={scenario_name} | "
                f"train={train_msg if train_msg is not None else 'n/a'} | "
                f"val={val_msg if val_msg is not None else 'n/a'}"
            )

        # Append one row per model config so downstream analysis can compare
        # architecture choices within a scenario.
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": f"{scenarios_dir.name}_{validation_tag}",
            "model_size_label": config.label,
            "hidden_channels": config.hidden_channels,
            "n_modes_x": config.n_modes_x,
            "n_modes_y": config.n_modes_y,
            "n_layers": config.n_layers,
            "total_params": result.total_params,
            "train_l2": f"{result.final_train_l2:.6f}",
            "val_l2": f"{result.final_val_l2:.6f}",
            "train_mse": f"{result.final_train_mse:.6f}",
            "val_mse": f"{result.final_val_mse:.6f}",
            "num_output_channels": result.out_channels,
            "train_rel_l2_norm_channels": result.train_rel_l2_norm_channels,
            "val_rel_l2_norm_channels": result.val_rel_l2_norm_channels,
            "train_rel_l2_denorm_channels": result.train_rel_l2_denorm_channels,
            "val_rel_l2_denorm_channels": result.val_rel_l2_denorm_channels,
            "train_mse_norm_channels": result.train_mse_norm_channels,
            "val_mse_norm_channels": result.val_mse_norm_channels,
            "train_mse_denorm_channels": result.train_mse_denorm_channels,
            "val_mse_denorm_channels": result.val_mse_denorm_channels,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "normalize": args.normalize,
            "scheduler_enabled": (not args.disable_scheduler),
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_decay": args.scheduler_decay,
            "device": str(device),
            "val_final_step_npz": str(npz_artifact_path),
            "val_final_step_plot": str(plot_artifact_path),
            "loss_curve_plot": str(loss_curve_plot_path),
            "per_scenario_train_metrics": json.dumps(per_scenario_train_metrics),
            "per_scenario_val_metrics": json.dumps(per_scenario_val_metrics),
            "per_scenario_train_plots": json.dumps(per_scenario_train_plots),
            "per_scenario_val_plots": json.dumps(per_scenario_val_plots),
        }
        append_result_row(csv_path, row, fieldnames)

        for scenario_name in scenario_names:
            scenario_train = per_scenario_train_metrics.get(scenario_name, {})
            scenario_val = per_scenario_val_metrics.get(scenario_name, {})
            per_scenario_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "scenario_collection": scenarios_dir.name,
                "validation_tag": validation_tag,
                "scenario_name": scenario_name,
                "model_size_label": config.label,
                "hidden_channels": config.hidden_channels,
                "n_modes_x": config.n_modes_x,
                "n_modes_y": config.n_modes_y,
                "n_layers": config.n_layers,
                "eval_every": args.eval_every,
                "train_l2": scenario_train.get("train_l2", ""),
                "train_mse": scenario_train.get("train_mse", ""),
                "val_l2": scenario_val.get("val_l2", ""),
                "val_mse": scenario_val.get("val_mse", ""),
                "train_plot": per_scenario_train_plots.get(scenario_name, ""),
                "val_plot": per_scenario_val_plots.get(scenario_name, ""),
            }
            append_result_row(per_scenario_csv_path, per_scenario_row, per_scenario_fieldnames)

        print(
            f"Completed label={config.label}, hidden_channels={config.hidden_channels} | "
            f"params={result.total_params} | "
            f"train_l2={result.final_train_l2:.6f} | val_l2={result.final_val_l2:.6f} | "
            f"train_mse={result.final_train_mse:.6f} | val_mse={result.final_val_mse:.6f} | "
            f"val_rel_l2_denorm_channels={result.val_rel_l2_denorm_channels} | "
            f"saved_npz={npz_artifact_path.name} | "
            f"saved_plot={plot_artifact_path.name} | "
            f"saved_loss_curve={loss_curve_plot_path.name}"
        )

    print("=" * 60)
    print("Sweep finished")
    print(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()
