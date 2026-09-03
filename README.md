# FNO for Henry Problem in Coastal Aquifers

Surrogate modeling and approximation experiments using Fourier Neural Operators (FNO) for coupled density-dependent groundwater flow and solute transport (the Henry saltwater intrusion problem).

---

## Dependencies & Installation

### Requirements
- **Python**: `>= 3.10, < 3.13`
- **Key Libraries**: PyTorch (`2.5.1`), TensorLy, TensorLy-Torch, NumPy, Pandas, Matplotlib

### Setup
The project configuration is defined in [`pyproject.toml`](pyproject.toml). You can install dependencies using either `uv` (recommended) or standard `pip`:

```bash
# Using uv (recommended)
uv sync

# Or using standard venv and pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Note**: For GPU execution on Linux, PyTorch with CUDA 11.8+ support is configured in `pyproject.toml`.

---

## Data Structure

The dataset loaders expect spatio-temporal window arrays stored as `.npz` files organized by scenario and run:

```text
scenarios_dir/
├── scenario_01/
│   ├── run_000001/windows.npz
│   ├── run_000002/windows.npz
│   └── ...
└── scenario_02/
    └── ...
```

---

## Model Training

### Single Model Training
Run [`train_fno.py`](train_fno.py) or use the helper script [`scripts/run_training.sh`](scripts/run_training.sh):

```bash
# Direct CLI execution
python train_fno.py \
  --scenario-dir /path/to/scenario_01 \
  --epochs 100 \
  --batch-size 512 \
  --learning-rate 1e-3 \
  --normalize

# Or via the shell script (supports environment variable overrides)
SCENARIO_DIR=/path/to/scenario_01 EPOCHS=100 ./scripts/run_training.sh
```

### Multi-Model Sweeps & Scaling Experiments
To run sweeps across model size presets (`tiny`, `small`, `medium`, `base`, `large`, `huge`, `massive`) or hidden dimension lists, use [`train_fno_sweep.py`](train_fno_sweep.py):

```bash
python train_fno_sweep.py \
  --scenarios-dir /path/to/scenarios \
  --sweep-mode preset \
  --model-size-presets "tiny,small,medium,base" \
  --epochs 500 \
  --batch-size 512 \
  --normalize \
  --results-dir ./results
```

For PBS/Torque HPC environments (e.g. NCI Gadi), submit the job script [`scripts/sweep_fno_size.pbs`](scripts/sweep_fno_size.pbs):

```bash
qsub scripts/sweep_fno_size.pbs
```

---

## What to Expect

### Training Feedback
- **Epoch Logs**: During training, the script reports relative training $L_2$ loss, validation MSE, and the active learning rate per epoch:
  ```text
  Epoch 010/100 - train_l2: 0.034512, val_mse: 0.001245, lr: 9.605960e-04
  ```
- **Final Metrics**: At completion, denormalized train and validation MSE metrics are displayed alongside normalization parameters.

### Outputs & Artifacts
- **Sweep Summaries**: Result tables saved as CSV files (`*_fno_sweep_results.csv` and `*_per_scenario_results.csv`) detailing model parameters, loss values, and runtimes.
- **Loss Plots**: Training and validation loss curves exported as `.png` images.
- **Evaluation Arrays**: Model predictions and ground truth targets exported as `.npz` files for downstream analysis and visualization in [`notebooks/compare_preds_targets.ipynb`](notebooks/compare_preds_targets.ipynb).
