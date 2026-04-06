#!/bin/bash
# Wait for 0403 co_hard_tp/abs_bucket_plus_co/pib_hard_tp sweep to finish,
# then launch TP weight sweep + raw RP

SWEEP_DIR="/root/dj/D2NN/kim2026/autoresearch/runs/0403-focal-pib-vacuum-target-pitchrescale-3strat-cn2-5e14"
LOG="/root/dj/D2NN/kim2026/autoresearch/runs/0403-tp-sweep.log"

echo "$(date) Watching for 0403 first sweep completion..." > "$LOG"

while true; do
  DONE=0
  for s in co_hard_tp abs_bucket_plus_co pib_hard_tp; do
    [ -f "$SWEEP_DIR/$s/results.json" ] && DONE=$((DONE+1))
  done
  if [ "$DONE" -ge 3 ]; then
    echo "$(date) All 3 strategies complete! Launching TP sweep..." >> "$LOG"
    break
  fi
  sleep 120
done

cd /root/dj/D2NN/kim2026
PYTHONPATH=src python -m autoresearch.d2nn_focal_pib_sweep \
  --config autoresearch/configs/0403-tp-sweep-and-raw-rp-30ep.yaml \
  >> "$LOG" 2>&1

echo "$(date) 0403 TP sweep COMPLETE!" >> "$LOG"
