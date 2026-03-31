#!/usr/bin/env bash
# Parallel ROI-complex loss sweep for beam-reduced D2NN (15cm → 5.12mm)
# 9 runs in 3 batches of 3, 50 epochs each
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SWEEP_DIR="runs/09_d2nn_br15cm_roi-complex-sweep"
PYTHON="${ROOT}/../miniconda3/bin/python"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "$LOG_DIR"

run_one() {
    local name="$1"
    local cfg="${SWEEP_DIR}/${name}/config.yaml"
    echo "[$(date +%H:%M:%S)] START  ${name}"
    $PYTHON -m kim2026.cli.train_beam_cleanup --config "$cfg" \
        > "${LOG_DIR}/${name}_train.log" 2>&1
    $PYTHON -m kim2026.cli.evaluate_beam_cleanup --config "$cfg" \
        > "${LOG_DIR}/${name}_eval.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE   ${name}"
}

echo "=== Batch 1/3: roi50 ==="
run_one roi50_lw05 &
run_one roi50_lw10 &
run_one roi50_lw20 &
wait
echo "=== Batch 1 complete ==="

echo "=== Batch 2/3: roi70 ==="
run_one roi70_lw05 &
run_one roi70_lw10 &
run_one roi70_lw20 &
wait
echo "=== Batch 2 complete ==="

echo "=== Batch 3/3: roi90 ==="
run_one roi90_lw05 &
run_one roi90_lw10 &
run_one roi90_lw20 &
wait
echo "=== Batch 3 complete ==="

echo "=== All 9 runs complete. Generating summary... ==="
$PYTHON -c "
import json
from pathlib import Path
from scripts.run_d2nn_br15cm_roi_complex_sweep import build_runs, aggregate_summary
runs = build_runs()
summary = aggregate_summary(runs=runs)
print(f\"Best: {summary['best_run_name']} (roi={summary['best_roi_threshold']}, lw={summary['best_leakage_weight']})\")
"
echo "=== SWEEP FINISHED ==="
