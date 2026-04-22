from __future__ import annotations

"""CLI entrypoint for running FNO architecture sweeps."""

from datetime import datetime
from src.config import validate_common_args
from src.sweep import (
    SWEEP_RESULT_FIELDNAMES,
    ModelSizeConfig,
    append_result_row,
    build_parser,
    parse_hidden_channels,
    parse_model_size_presets,
    save_validation_final_step_artifacts,
    scenario_results_csv,
    train_one_model,
)
from train_fno import evaluate_mse, resolve_device, set_seed


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

    csv_path = scenario_results_csv(args.results_dir, args.scenario_dir)
    artifact_dir = args.results_dir / args.artifact_dir_name / args.scenario_dir.name
    fieldnames = SWEEP_RESULT_FIELDNAMES

    print("Starting multi-model FNO sweep")
    print(f"scenario: {args.scenario_dir}")
    print(f"sweep_mode: {args.sweep_mode}")
    if args.sweep_mode == "preset":
        print(f"model_size_presets: {[cfg.label for cfg in sweep_configs]}")
    else:
        print(f"hidden_channels_list: {[cfg.hidden_channels for cfg in sweep_configs]}")
        print(f"fixed architecture: n_modes=({args.n_modes_x}, {args.n_modes_y}), n_layers={args.n_layers}")
    print(f"results csv: {csv_path}")
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
            scenario_dir=args.scenario_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_ratio=args.train_ratio,
            seed=args.seed,
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
            scenario_name=args.scenario_dir.name,
            model_size_label=config.label,
            hidden_channels=config.hidden_channels,
        )

        # Append one row per model config so downstream analysis can compare
        # architecture choices within a scenario.
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": args.scenario_dir.name,
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
        }
        append_result_row(csv_path, row, fieldnames)

        print(
            f"Completed label={config.label}, hidden_channels={config.hidden_channels} | "
            f"params={result.total_params} | "
            f"train_l2={result.final_train_l2:.6f} | val_l2={result.final_val_l2:.6f} | "
            f"train_mse={result.final_train_mse:.6f} | val_mse={result.final_val_mse:.6f} | "
            f"val_rel_l2_denorm_channels={result.val_rel_l2_denorm_channels} | "
            f"saved_npz={npz_artifact_path.name} | "
            f"saved_plot={plot_artifact_path.name}"
        )

    print("=" * 60)
    print("Sweep finished")
    print(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()
