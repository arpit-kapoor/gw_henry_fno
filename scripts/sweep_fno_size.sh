#!/usr/bin/env bash
set -euo pipefail

# Resolve project root directory regardless of invocation location
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Ensure stdin is decoupled from any controlling terminal (prevents multiprocessing spawn failure if terminal closes)
exec 0</dev/null

# Ensure Python flushes stdout/stderr immediately when redirected to a log file
export PYTHONUNBUFFERED=1

echo "Starting FNO model size sweep script..."
echo "Project directory: ${PROJECT_DIR}"
echo "Current working directory: $(pwd)"

# Problem setting name for organizing results
SETTING_NAME="grid_scenarios_20x40"

# Set up the scenarios root directory
SCENARIOS_ROOT="${SCENARIOS_ROOT:-$HOME/Projects/groundwater/data/simple_henry_data/${SETTING_NAME}/scenarios}"

# Results directory for storing the outputs of the sweep
RESULTS_DIR="${RESULTS_DIR:-$HOME/Projects/groundwater/results/fno_simple_henry_sweep/${SETTING_NAME}}"
mkdir -p "${RESULTS_DIR}"

# Activate the virtual environment if it exists, otherwise use the system Python
if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

MODEL_SIZE_PRESETS="${MODEL_SIZE_PRESETS:-tiny,small,medium,base,large,huge,massive}"
# MODEL_SIZE_PRESETS="massive"
DEVICE="${DEVICE:-mps}"  # Change to "cuda" if using a CUDA-capable GPU
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-8e-4}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-321}"
VALIDATION_RUN_NAME="${VALIDATION_RUN_NAME:-run_000003}"

if [ ! -d "${SCENARIOS_ROOT}" ]; then
  echo "Scenarios root directory not found: ${SCENARIOS_ROOT}"
  exit 1
fi

echo "============================================================"
echo "Running sweep for scenarios root: ${SCENARIOS_ROOT}"
echo "Model size presets: ${MODEL_SIZE_PRESETS}"
echo "Device: ${DEVICE}"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================================"

# Run the training script with the specified parameters
"${PYTHON_BIN}" "${PROJECT_DIR}/train_fno_sweep.py" \
  --scenarios-dir "${SCENARIOS_ROOT}" \
  --sweep-mode preset \
  --model-size-presets "${MODEL_SIZE_PRESETS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --train-ratio "${TRAIN_RATIO}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory \
  --scheduler-step-size 50 \
  --scheduler-decay 0.80 \
  --seed "${SEED}" \
  --validation-run-name "${VALIDATION_RUN_NAME}" \
  --results-dir "${RESULTS_DIR}" \
  --device "${DEVICE}"


status=$?
if [ ${status} -ne 0 ]; then
      echo "Sweep failed with exit code ${status}"
      exit ${status}
fi
