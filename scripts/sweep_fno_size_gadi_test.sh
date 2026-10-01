#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Direct shell test for 3-D FNO training on an HPC GPU node (Gadi)
# Mirrors the configuration from scripts/sweep_fno_size.pbs
#
# Dataset layout  : scenario_NNN/scenario.npz (one packed file per scenario)
# Model           : 3-D FNO n_modes=(n_modes_t, n_modes_z, n_modes_x)
# Split           : 70/30 random at run-index level, shared across all scenarios
# =============================================================================

# Determine project root directory dynamically (one level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Prevent cluster-provided Python packages from overriding the venv packages.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Configuration & Paths (from sweep_fno_size.pbs)
# ---------------------------------------------------------------------------

# Name of the dataset archive on scratch (without extension)
SETTING_NAME="grid_scenarios_random_skip2_20x40"

# Source archive on RDS/scratch
DATA_ARCHIVE="/scratch/yl75/ak4177/data/simple_henry/${SETTING_NAME}.tar.gz"

# Node-local fast storage: use $PBS_JOBFS if in an interactive PBS job,
# otherwise fallback to scratch temp directory
if [ -n "${PBS_JOBFS:-}" ] && [ -d "${PBS_JOBFS}" ]; then
    JOBFS_DIR="${PBS_JOBFS}"
else
    JOBFS_DIR="/scratch/yl75/ak4177/tmp/jobfs"
    mkdir -p "${JOBFS_DIR}"
fi

# Scenarios root: the extracted folder contains scenario_NNN/ subdirs directly
SCENARIOS_ROOT="${JOBFS_DIR}/${SETTING_NAME}"

# Results directory on persistent scratch
RESULTS_DIR="${RESULTS_DIR:-/scratch/yl75/ak4177/results/groundwater/fno_3d_henry_sweep/${SETTING_NAME}}"

# Python executable in the virtual environment
PYTHON_BIN="/scratch/yl75/ak4177/src/GW_SciML/.venv/bin/python"

# Model presets to sweep (tiny, small, medium, large, huge, massive)
# Default matches sweep_fno_size.pbs ("massive"); can be overridden from shell:
#   MODEL_SIZE_PRESETS="tiny" ./scripts/sweep_fno_size_gadi_test.sh
MODEL_SIZE_PRESETS="${MODEL_SIZE_PRESETS:-massive}"

# Number of training epochs (default 500 from sweep_fno_size.pbs; override e.g. EPOCHS=5 for fast test)
EPOCHS="${EPOCHS:-500}"

# ---------------------------------------------------------------------------
# Sanity & Pre-flight Checks
# ---------------------------------------------------------------------------

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python executable not found or not executable: ${PYTHON_BIN}"
    exit 1
fi

if [ ! -f "${DATA_ARCHIVE}" ]; then
    echo "ERROR: Data archive not found: ${DATA_ARCHIVE}"
    exit 1
fi

# Extract dataset to fast storage if not already present
if [ ! -d "${SCENARIOS_ROOT}" ]; then
    echo "Extracting ${DATA_ARCHIVE} to ${JOBFS_DIR}/..."
    tar -xf "${DATA_ARCHIVE}" -C "${JOBFS_DIR}/"
else
    echo "Found existing scenarios directory: ${SCENARIOS_ROOT}"
fi

if [ ! -d "${SCENARIOS_ROOT}" ]; then
    echo "ERROR: Scenarios root directory not found: ${SCENARIOS_ROOT}"
    exit 1
fi

echo "============================================================"
echo "3-D FNO GPU Shell Test — ${SETTING_NAME}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Scenarios root : ${SCENARIOS_ROOT}"
echo "Results dir    : ${RESULTS_DIR}"
echo "Model presets  : ${MODEL_SIZE_PRESETS}"
echo "Epochs         : ${EPOCHS}"
echo "============================================================"

# # Fail early if Python environment or CUDA is not functioning
# "${PYTHON_BIN}" -c 'import torch; print(f"Python: {__import__(\"sys\").executable}"); print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}"); print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None (CPU only)\"}")'

# ---------------------------------------------------------------------------
# Run model training
# ---------------------------------------------------------------------------

set +e
"${PYTHON_BIN}" train_fno_sweep.py \
    --scenario-dir        "${SCENARIOS_ROOT}"        \
    --sweep-mode          preset                     \
    --model-size-presets  "${MODEL_SIZE_PRESETS}"    \
    --epochs              "${EPOCHS}"                \
    --batch-size          152                         \
    --learning-rate       1e-4                       \
    --weight-decay        1e-5                       \
    --train-ratio         0.7                        \
    --seed                321                        \
    --num-workers         4                          \
    --pin-memory                                     \
    --normalize                                      \
    --scheduler-step-size 50                        \
    --scheduler-decay     0.75                       \
    --results-dir         "${RESULTS_DIR}"           \
    --device              "${DEVICE:-cuda}"          \
    "$@"

status=$?
set -e

if [ "${status}" -ne 0 ]; then
    echo "Training failed with exit code ${status}"
    exit "${status}"
fi

echo "============================================================"
echo "Training completed successfully."
echo "Results written to: ${RESULTS_DIR}"
echo "============================================================"