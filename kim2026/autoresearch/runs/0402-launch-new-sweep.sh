#!/bin/bash
# Wait for 0401 co_pib_hybrid to finish, then launch 0402 sweep

SWEEP_DIR="/root/dj/D2NN/kim2026/autoresearch/runs/0401-focal-pib-sweep-clean-4loss-cn2-5e14"
LOG="/root/dj/D2NN/kim2026/autoresearch/runs/0402-new-losses-sweep.log"

echo "$(date) Watching for 0401 sweep completion..." > "$LOG"

while true; do
  if [ -f "$SWEEP_DIR/focal_co_pib_hybrid/results.json" ]; then
    echo "$(date) 0401 sweep complete! Launching 0402 new losses sweep..." >> "$LOG"
    break
  fi
  sleep 60
done

# Launch new sweep
cd /root/dj/D2NN/kim2026
PYTHONPATH=src python -m autoresearch.d2nn_focal_pib_sweep \
  --config autoresearch/configs/0402-focal-sweep-new-losses-30ep.yaml \
  >> "$LOG" 2>&1

echo "$(date) 0402 sweep COMPLETE!" >> "$LOG"
