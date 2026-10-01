#!/usr/bin/env bash
set -euo pipefail

# Move to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Model presets to run: pass as arguments or default to all presets
# Example: ./submit_sweep_parallel.sh small medium massive
PRESETS=("$@")
if [ ${#PRESETS[@]} -eq 0 ]; then
    PRESETS=(huge large medium small tiny)
    # PRESETS=(small)
fi

mkdir -p logs

for preset in "${PRESETS[@]}"; do
    echo "Submitting 3-D FNO job for preset: ${preset}"
    qsub \
        -N "fno_${preset}" \
        -o "logs/fno_sweep_${preset}.out" \
        -v "MODEL_SIZE_PRESETS=${preset}" \
        sweep_fno_size.pbs
done
