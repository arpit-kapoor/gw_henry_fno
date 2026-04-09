#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO_DIR="/Users/akap5486/Projects/groundwater/data/henry_data/all_scenarios/coupling_scenarios/scenario_01"


if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

exec "${PYTHON_BIN}" "${PROJECT_DIR}/train_fno_sweep.py" \
      --scenario-dir "${SCENARIO_DIR}" \
      --hidden-channels-list "8,16" \
      --epochs 5 \
      --batch-size 512 \
      --learning-rate 1e-3 \
      --train-ratio 0.7 \
      --n-modes-x 8 \
      --n-modes-y 16 \
      --n-layers 4 \
      --num-workers 4 
