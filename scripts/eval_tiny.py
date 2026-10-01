import json
import torch
from pathlib import Path
from datetime import datetime

import sys
sys.path.append("/scratch/yl75/ak4177/src/gw_henry_fno")

from src.data.henry_scenario_dataset import HenryScenarioDataset
from src.data.normalizer import Normalizer
from src.sweep import (
    SWEEP_CSV_FIELDNAMES,
    ModelSizeConfig,
    append_result_row,
    collect_predictions_by_scenario,
    compute_rel_l2_per_channel,
    save_predictions_npz,
)
from src.sweep.rollout import compute_rel_combined_norm_error
from src.neuralop.fno import FNO
from src.data.henry_scenario_dataset import create_henry_dataloaders

def main():
    scenarios_dir = Path("/scratch/yl75/ak4177/data/simple_henry/grid_scenarios_random_skip2_20x40")
    results_dir = Path("/scratch/yl75/ak4177/results/groundwater/fno_3d_henry_sweep/grid_scenarios_random_skip2_20x40_new_rel_loss")
    csv_path = results_dir / "sweep_results.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} for tiny model prediction collection.")
    
    config = ModelSizeConfig(
        label="tiny",
        hidden_channels=4,
        n_modes_t=4,
        n_modes_z=4,
        n_modes_x=4,
        n_layers=4
    )
    
    model = FNO(
        n_modes=(config.n_modes_t, config.n_modes_z, config.n_modes_x),
        hidden_channels=config.hidden_channels,
        in_channels=4,
        out_channels=2,
        n_layers=config.n_layers
    ).to(device)
    
    weights_path = results_dir / "model_weights" / "tiny.pt"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print("Loading datasets...")
    train_loader, val_loader, normalizer = create_henry_dataloaders(
        scenarios_dir=scenarios_dir,
        train_ratio=0.7,
        seed=31,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        normalize=True,
    )
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    print(f"Collecting train predictions for {config.label} ...")
    train_preds_by_scenario = collect_predictions_by_scenario(
        model=model,
        dataset=train_dataset,
        device=device,
        normalizer=normalizer,
    )

    print(f"Collecting val predictions for {config.label} ...")
    val_preds_by_scenario = collect_predictions_by_scenario(
        model=model,
        dataset=val_dataset,
        device=device,
        normalizer=normalizer,
    )
    
    dt = 0.04
    dz = 0.05
    dx = 0.05

    all_scenario_names = sorted(
        set(train_preds_by_scenario.keys()) | set(val_preds_by_scenario.keys())
    )

    total_params = sum(p.numel() for p in model.parameters())

    for scenario_name in all_scenario_names:
        train_data = train_preds_by_scenario.get(scenario_name)
        val_data = val_preds_by_scenario.get(scenario_name)

        if train_data is not None:
            train_rel_l2_conc, train_rel_l2_head = compute_rel_l2_per_channel(
                targets=train_data["targets"],
                preds=train_data["preds"],
            )
            train_rel_combined_norm = compute_rel_combined_norm_error(
                targets=train_data["targets"],
                preds=train_data["preds"],
                dt=dt,
                dz=dz,
                dx=dx,
            )
        else:
            train_rel_l2_conc, train_rel_l2_head = float("nan"), float("nan")
            train_rel_combined_norm = float("nan")

        if val_data is not None:
            val_rel_l2_conc, val_rel_l2_head = compute_rel_l2_per_channel(
                targets=val_data["targets"],
                preds=val_data["preds"],
            )
            val_rel_combined_norm = compute_rel_combined_norm_error(
                targets=val_data["targets"],
                preds=val_data["preds"],
                dt=dt,
                dz=dz,
                dx=dx,
            )
        else:
            val_rel_l2_conc, val_rel_l2_head = float("nan"), float("nan")
            val_rel_combined_norm = float("nan")

        print(
            f"  scenario={scenario_name} | "
            f"train_norm={train_rel_combined_norm:.6f}, train_conc={train_rel_l2_conc:.6f}, train_head={train_rel_l2_head:.6f} | "
            f"val_norm={val_rel_combined_norm:.6f}, val_conc={val_rel_l2_conc:.6f}, val_head={val_rel_l2_head:.6f}"
        )

        npz_path = save_predictions_npz(
            train_targets=train_data["targets"] if train_data else None,
            train_preds=train_data["preds"] if train_data else None,
            val_targets=val_data["targets"] if val_data else None,
            val_preds=val_data["preds"] if val_data else None,
            output_dir=results_dir,
            scenario_name=scenario_name,
            model_size_label=config.label,
        )
        print(f"  Saved predictions: {npz_path}")

        row = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario_name": scenario_name,
            "model_size_label": config.label,
            "total_params": total_params,
            "rel_combined_norm_train": f"{train_rel_combined_norm:.6f}",
            "rel_combined_norm_val": f"{val_rel_combined_norm:.6f}",
            "rel_l2_error_concentration_train": f"{train_rel_l2_conc:.6f}",
            "rel_l2_error_hydraulic_head_train": f"{train_rel_l2_head:.6f}",
            "rel_l2_error_concentration_val": f"{val_rel_l2_conc:.6f}",
            "rel_l2_error_hydraulic_head_val": f"{val_rel_l2_head:.6f}",
        }
        append_result_row(csv_path, row, SWEEP_CSV_FIELDNAMES)

    print("Finished evaluating tiny model.")

if __name__ == '__main__':
    main()
