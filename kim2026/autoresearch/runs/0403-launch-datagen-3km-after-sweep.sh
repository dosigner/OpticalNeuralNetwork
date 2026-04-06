#!/bin/bash
# Wait for 0403 combined sweep to finish, then launch 3km data generation

SWEEP_DIR="/root/dj/D2NN/kim2026/autoresearch/runs/0403-combined-6strat-pitchrescale-cn2-5e14"
LOG="/root/dj/D2NN/kim2026/autoresearch/runs/0403-datagen-3km.log"

echo "$(date) Watching for 0403 sweep completion..." > "$LOG"

while true; do
  DONE=0
  for s in abs_bucket_plus_co pib_hard_tp focal_tp_pib_w05 focal_tp_pib_w2 focal_tp_pib_w5 focal_raw_received_power; do
    [ -f "$SWEEP_DIR/$s/results.json" ] && DONE=$((DONE+1))
  done
  if [ "$DONE" -ge 6 ]; then
    echo "$(date) All 6 strategies complete! Starting 3km data generation..." >> "$LOG"
    break
  fi
  sleep 120
done

cd /root/dj/D2NN/kim2026
PYTHONPATH=src python scripts/generate_data_pitch_rescale_3km.py >> "$LOG" 2>&1

echo "$(date) 3km data generation COMPLETE!" >> "$LOG"
