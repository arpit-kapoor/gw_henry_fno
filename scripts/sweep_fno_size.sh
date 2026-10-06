#!/usr/bin/env bash
set -euo pipefail

# Resolve project root directory regardless of invocation location
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
SCRIPT_PATH="${PROJECT_DIR}/scripts/$(basename "${BASH_SOURCE[0]:-$0}")"
cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Background launch (equivalent of PBS -o logs/... -j oe)
#
# By default the script re-launches itself detached with nohup, sending
# stdout+stderr to a dedicated log file, and returns immediately.
#   Follow the log : tail -f logs/fno_sweep_<timestamp>.out
#   Run attached   : FOREGROUND=1 scripts/sweep_fno_size.sh
#   Custom log     : LOG_FILE=logs/my_run.out scripts/sweep_fno_size.sh
# ---------------------------------------------------------------------------
if [[ "${FOREGROUND:-0}" != "1" && -z "${_SWEEP_DETACHED:-}" ]]; then
  mkdir -p "${PROJECT_DIR}/logs"
  LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/fno_sweep_$(date +%Y%m%d_%H%M%S).out}"
  _SWEEP_DETACHED=1 nohup bash "${SCRIPT_PATH}" "$@" >"${LOG_FILE}" 2>&1 </dev/null &
  pid=$!
  echo "Sweep launched in background (PID ${pid})"
  echo "Log file : ${LOG_FILE}"
  echo "Follow   : tail -f \"${LOG_FILE}\""
  echo "Stop     : kill ${pid}"
  exit 0
fi

# Ensure stdin is decoupled from any controlling terminal (prevents multiprocessing spawn failure if terminal closes)
exec 0</dev/null

# Keep macOS from idle-sleeping while this script is alive
if [[ -n "${_SWEEP_DETACHED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &
fi

echo "Job started at: $(date)"
echo "Host: $(hostname)  PID: $$"

# Ensure Python flushes stdout/stderr immediately when redirected to a log file
export PYTHONUNBUFFERED=1

echo "Starting FNO model size sweep script..."
echo "Project directory: ${PROJECT_DIR}"
echo "Current working directory: $(pwd)"

# Problem setting name for organizing results
SETTING_NAME="grid_scenarios_forced_random_skip4_20x40"

SCENARIOS_ROOT="${SCENARIOS_ROOT:-$HOME/Projects/groundwater/data/henry_forced_data/${SETTING_NAME}}"

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

MODEL_SIZE_PRESETS="${MODEL_SIZE_PRESETS:-small}"
DEVICE="${DEVICE:-auto}"  # "auto" detects CUDA on Linux, MPS on macOS, or CPU fallback
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-152}"
# Each batch is processed in chunks of this size with gradient accumulation:
# same optimizer update as BATCH_SIZE, but lower peak memory (needed on MPS).
# Set to 0 to disable.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-38}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-31}"

# Spatial and temporal grid spacing for the RelCombinedNormLoss metric
DT="${DT:-0.04}"
DZ="${DZ:-0.05}"
DX="${DX:-0.05}"

if [ ! -d "${SCENARIOS_ROOT}" ]; then
  echo "Scenarios root directory not found: ${SCENARIOS_ROOT}"
  exit 1
fi

echo "============================================================"
echo "Running sweep for scenarios root: ${SCENARIOS_ROOT}"
echo "Model size presets: ${MODEL_SIZE_PRESETS}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE} (micro-batch: ${MICRO_BATCH_SIZE})"
echo "Results dir: ${RESULTS_DIR}"
echo "Grid spacing: dt=${DT}, dz=${DZ}, dx=${DX}"
echo "============================================================"

set +e
"${PYTHON_BIN}" "${PROJECT_DIR}/train_fno_sweep.py" \
  --scenario-dir "${SCENARIOS_ROOT}" \
  --sweep-mode preset \
  --model-size-presets "${MODEL_SIZE_PRESETS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --train-ratio "${TRAIN_RATIO}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory \
  --normalize \
  --scheduler-step-size 50 \
  --scheduler-decay 0.75 \
  --seed "${SEED}" \
  --results-dir "${RESULTS_DIR}" \
  --device "${DEVICE}" \
  --dt "${DT}" \
  --dz "${DZ}" \
  --dx "${DX}" &

# Forward stop signals to the training process so `kill <PID>` stops the whole run
TRAIN_PID=$!
trap 'echo "Stop requested; terminating training (PID ${TRAIN_PID})"; kill -TERM "${TRAIN_PID}" 2>/dev/null' INT TERM HUP
wait "${TRAIN_PID}"
status=$?
# wait is interrupted when a trapped signal arrives; wait again for the real exit code
if kill -0 "${TRAIN_PID}" 2>/dev/null; then
  wait "${TRAIN_PID}"
  status=$?
fi

if [ ${status} -ne 0 ]; then
      echo "Sweep failed with exit code ${status} at $(date)"
      exit ${status}
fi

echo "============================================================"
echo "Sweep completed successfully for preset(s): ${MODEL_SIZE_PRESETS}"
echo "Results written to: ${RESULTS_DIR}"
echo "Finished at: $(date)"
echo "============================================================"
