#!/usr/bin/env bash
set -euo pipefail

# Training configuration
# Override any value with environment variables, for example:
#   EPOCHS=50 BATCH_SIZE=32 NORMALIZE=1 ./scripts/run_training.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Required: path to one scenario directory containing run_*/windows.npz
SCENARIO_DIR="${SCENARIO_DIR:-/Users/akap5486/Projects/groundwater/data/henry_data/all_scenarios/coupling_scenarios/scenario_01}"

# Core training options
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

# FNO architecture options
N_MODES_X="${N_MODES_X:-16}"
N_MODES_Y="${N_MODES_Y:-24}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
N_LAYERS="${N_LAYERS:-4}"

# Data loader options
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEMORY="${PIN_MEMORY:-0}"
NORMALIZE="${NORMALIZE:-1}"

# Scheduler options
SCHEDULER_ENABLED="${SCHEDULER_ENABLED:-1}"
SCHEDULER_STEP_SIZE="${SCHEDULER_STEP_SIZE:-5}"
SCHEDULER_DECAY="${SCHEDULER_DECAY:-0.98}"

# Execution options
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SCENARIO_DIR}" ]]; then
  echo "Error: SCENARIO_DIR is required."
  echo "Example: SCENARIO_DIR=/path/to/scenario_01 ./scripts/run_training.sh"
  exit 1
fi

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/train_fno.py"
  --scenario-dir "${SCENARIO_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --train-ratio "${TRAIN_RATIO}"
  --seed "${SEED}"
  --n-modes-x "${N_MODES_X}"
  --n-modes-y "${N_MODES_Y}"
  --hidden-channels "${HIDDEN_CHANNELS}"
  --n-layers "${N_LAYERS}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)

if [[ "${PIN_MEMORY}" == "1" ]]; then
  CMD+=(--pin-memory)
fi

if [[ "${NORMALIZE}" == "1" ]]; then
  CMD+=(--normalize)
fi

if [[ "${SCHEDULER_ENABLED}" == "1" ]]; then
  CMD+=(--scheduler-step-size "${SCHEDULER_STEP_SIZE}" --scheduler-decay "${SCHEDULER_DECAY}")
else
  CMD+=(--disable-scheduler)
fi

# Pass through any additional CLI args from the shell script invocation.
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed."
  exit 0
fi

exec "${CMD[@]}"
