# FNO for the Henry Problem in Coastal Aquifers

Surrogate modelling experiments with 3-D Fourier Neural Operators (FNO) for coupled density-dependent groundwater flow and solute transport (the Henry saltwater intrusion problem). The FNO maps an input trajectory `(C_in, T_in, Z, X)` to the full output trajectory of salt concentration and hydraulic head `(C_out, T_out, Z, X)` in a single forward pass, with spectral convolutions over time and both spatial axes.

## Installation

Requires Python `>=3.10,<3.13` (developed with 3.12). Dependencies are pinned in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).

```bash
uv sync                       # recommended
# or
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

On Linux, `uv` installs PyTorch 2.5.1 with CUDA 11.8. On macOS, the Apple MPS backend is supported.

## Data

The datasets are not included in this repository. Each scenario directory holds a single `scenario.npz` that packs all simulation runs for that scenario:

```text
scenarios_dir/
├── scenario_001/scenario.npz
├── scenario_002/scenario.npz
└── ...
```

| Key | Shape |
|---|---|
| `input_tensor` | `(N_runs, C_in, T_in, Z, X)` |
| `output_tensor` | `(N_runs, C_out, T_out, Z, X)` (channel 0: concentration, channel 1: hydraulic head) |
| `input_channel_names`, `output_channel_names` | optional |

Runs are split into train and validation sets at the run-index level (`--train-ratio`, default 0.7). The same partition is shared by every scenario. Inputs and outputs are normalised with per-channel mean and standard deviation computed on the training set.

## Training

### Model-size sweep (main experiments)

[`train_fno_sweep.py`](train_fno_sweep.py) trains one model per configuration on all scenarios. Training minimises `RelCombinedNormLoss`, the relative error in a combined norm of concentration and hydraulic head (see [`src/neuralop/losses.py`](src/neuralop/losses.py)).

```bash
python train_fno_sweep.py \
  --scenario-dir /path/to/scenarios_dir \
  --sweep-mode preset \
  --model-size-presets tiny,small,medium,large,huge,massive \
  --epochs 500 --batch-size 152 --learning-rate 1e-4 --weight-decay 1e-5 \
  --scheduler-step-size 50 --scheduler-decay 0.75 \
  --dt 0.04 --dz 0.05 --dx 0.05 \
  --seed 31 --results-dir results/
```

- `--sweep-mode preset` uses the presets in [`src/sweep/config.py`](src/sweep/config.py), which scale hidden channels, Fourier modes `(t, z, x)` and depth together. `--sweep-mode hidden` varies `--hidden-channels-list` at fixed modes and depth instead.
- `--dt/--dz/--dx` set the grid spacing used by the combined norm.
- `--micro-batch-size N` accumulates gradients over chunks of `N` samples. The optimiser update is the same as with the full batch, but peak memory is lower.
- `--device` accepts `auto` (CUDA → MPS → CPU), `cuda`, `mps` or `cpu`.
- Run `python train_fno_sweep.py --help` for all options.

Launcher scripts:

- [`scripts/sweep_fno_size.sh`](scripts/sweep_fno_size.sh) runs a sweep on a local machine. It detaches into the background and logs to `logs/`; set `FOREGROUND=1` to run attached. Hyperparameters can be overridden with environment variables, e.g. `MODEL_SIZE_PRESETS=small EPOCHS=500 scripts/sweep_fno_size.sh`.
- [`scripts/sweep_fno_size.pbs`](scripts/sweep_fno_size.pbs) runs one sweep job on a PBS cluster. Set the project, storage and paths at the top of the script.
- [`scripts/submit_sweep_parallel.sh`](scripts/submit_sweep_parallel.sh) submits one PBS job per preset, e.g. `scripts/submit_sweep_parallel.sh small medium`.

### Single model

[`train_fno.py`](train_fno.py) trains a single FNO with a relative L2 loss and prints the final train and validation MSE. It does not save any outputs.

```bash
python train_fno.py --scenario-dir /path/to/scenarios_dir --normalize \
  --n-modes-t 8 --n-modes-z 8 --n-modes-x 16 --hidden-channels 32 --n-layers 4
```

## Outputs

The sweep writes the following to `--results-dir`:

| Path | Contents |
|---|---|
| `sweep_results.csv` | One row per (scenario, model): parameter count, relative combined-norm error and per-channel relative L2 error on the train and validation sets. Rows are appended across runs. |
| `model_weights/<preset>.pt` | Model `state_dict` and architecture configuration |
| `figures/<preset>_loss_history.json` | Per-epoch train and validation loss histories |
| `predictions/<scenario>_<preset>.npz` | Denormalised `train_preds`, `train_targets`, `val_preds`, `val_targets`, shaped `(N_runs, T, Z, X, C)` |

The notebooks in [`notebooks/`](notebooks/) read these outputs for error analysis and plotting. Set `BASE_RESULTS_PATH` and `BASE_DATA_PATH` in the first cells.

## Repository layout

```text
train_fno_sweep.py     model-size sweep entry point
train_fno.py           single-model training
src/config.py          shared CLI arguments
src/data/              scenario dataset, run-level split, normaliser
src/neuralop/          FNO, spectral convolution and losses (adapted from neuraloperator)
src/sweep/             sweep presets, training loop, metrics and output writers
scripts/               local and PBS launchers
tests/                 unit tests
notebooks/             result analysis
```

Run the tests from the repository root with `uv run --with pytest python -m pytest tests`.

## Third-party code

The code in [`src/neuralop/`](src/neuralop/) is adapted from the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library (Kossaifi et al., 2025, *A Library for Learning Neural Operators*, arXiv:2412.10354), released under the MIT License ([`src/neuralop/LICENSE`](src/neuralop/LICENSE)). Our additions are `RelCombinedNormLoss`, a real-valued truncated-DFT spectral convolution path and pointwise layers optimised for Apple MPS. Each file's header lists its changes.
