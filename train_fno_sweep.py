from __future__ import annotations

"""CLI entrypoint for running FNO architecture sweeps."""

import gc
from datetime import datetime

import torch

from src.config import validate_common_args
from src.sweep import (
    SWEEP_CSV_FIELDNAMES,
    ModelSizeConfig,
    append_result_row,
    build_parser,
    collect_predictions_by_scenario,
    compute_rollout_rel_l2_per_channel,
    parse_hidden_channels,
    parse_model_size_presets,
    save_loss_history_json,
    save_model_weights,
    save_predictions_npz,
    train_one_model,
)
from train_fno import resolve_device, set_seed


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
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "sweep_results.csv"

    print("Starting multi-model FNO sweep")
    print(f"scenarios_dir: {scenarios_dir}")
    print(f"sweep_mode: {args.sweep_mode}")
    if args.sweep_mode == "preset":
        print(f"model_size_presets: {[cfg.label for cfg in sweep_configs]}")
    else:
        print(f"hidden_channels_list: {[cfg.hidden_channels for cfg in sweep_configs]}")
        print(f"fixed architecture: n_modes=({args.n_modes_x}, {args.n_modes_y}), n_layers={args.n_layers}")
    print(f"results csv: {csv_path}")

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
        )

        # 1. Save model weights + architecture config.
        weights_path = save_model_weights(
            model=result.model,
            config=config,
            output_dir=results_dir,
        )
        print(f"Saved model weights: {weights_path}")

        # 2. Save per-epoch loss histories (train/val one-step-ahead, normalised space).
        loss_json_path = save_loss_history_json(
            train_loss_history=result.train_loss_history,
            val_loss_history=result.val_loss_history,
            model_size_label=config.label,
            output_dir=results_dir,
        )
        print(f"Saved loss history: {loss_json_path}")

        train_dataset = result.train_loader.dataset
        val_dataset = result.val_loader.dataset

        # 3. Collect per-scenario predictions for both splits.
        print(f"Collecting train predictions for {config.label} ...")
        train_preds_by_scenario = collect_predictions_by_scenario(
            model=result.model,
            dataset=train_dataset,
            device=device,
            normalizer=result.normalizer,
        )

        print(f"Collecting val predictions for {config.label} ...")
        val_preds_by_scenario = collect_predictions_by_scenario(
            model=result.model,
            dataset=val_dataset,
            device=device,
            normalizer=result.normalizer,
        )

        # 4. For each scenario: save NPZ and append CSV row.
        all_scenario_names = sorted(
            set(train_preds_by_scenario.keys()) | set(val_preds_by_scenario.keys())
        )

        for scenario_name in all_scenario_names:
            train_data = train_preds_by_scenario.get(scenario_name)
            val_data = val_preds_by_scenario.get(scenario_name)

            # Compute rollout relative L2 per channel (denormalized).
            if train_data is not None:
                train_rel_l2_conc, train_rel_l2_head = compute_rollout_rel_l2_per_channel(
                    targets=train_data["targets"],
                    preds_rollout=train_data["preds_rollout"],
                )
            else:
                train_rel_l2_conc, train_rel_l2_head = float("nan"), float("nan")

            if val_data is not None:
                val_rel_l2_conc, val_rel_l2_head = compute_rollout_rel_l2_per_channel(
                    targets=val_data["targets"],
                    preds_rollout=val_data["preds_rollout"],
                )
            else:
                val_rel_l2_conc, val_rel_l2_head = float("nan"), float("nan")

            print(
                f"  scenario={scenario_name} | "
                f"train_conc={train_rel_l2_conc:.6f}, train_head={train_rel_l2_head:.6f} | "
                f"val_conc={val_rel_l2_conc:.6f}, val_head={val_rel_l2_head:.6f}"
            )

            # Save predictions NPZ: all six arrays with shape (N, T, H, W, 2).
            # N may differ between train and val splits.
            _empty = lambda: None  # sentinel; replaced below if data absent
            npz_path = save_predictions_npz(
                train_targets=train_data["targets"] if train_data else None,
                train_preds=train_data["preds"] if train_data else None,
                train_preds_rollout=train_data["preds_rollout"] if train_data else None,
                val_targets=val_data["targets"] if val_data else None,
                val_preds=val_data["preds"] if val_data else None,
                val_preds_rollout=val_data["preds_rollout"] if val_data else None,
                output_dir=results_dir,
                scenario_name=scenario_name,
                model_size_label=config.label,
            )
            print(f"  Saved predictions: {npz_path}")

            # Append one CSV row per (scenario, model) pair.
            row = {
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                "scenario_name": scenario_name,
                "model_size_label": config.label,
                "total_params": result.total_params,
                "rel_l2_error_concentration_train": f"{train_rel_l2_conc:.6f}",
                "rel_l2_error_hydraulic_head_train": f"{train_rel_l2_head:.6f}",
                "rel_l2_error_concentration_val": f"{val_rel_l2_conc:.6f}",
                "rel_l2_error_hydraulic_head_val": f"{val_rel_l2_head:.6f}",
            }
            append_result_row(csv_path, row, SWEEP_CSV_FIELDNAMES)

        print(
            f"Completed label={config.label}, hidden_channels={config.hidden_channels} | "
            f"params={result.total_params}"
        )

        # Clean up model, loaders, and memory before the next configuration.
        del result, train_dataset, val_dataset
        del train_preds_by_scenario, val_preds_by_scenario
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("=" * 60)
    print("Sweep finished")
    print(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()
