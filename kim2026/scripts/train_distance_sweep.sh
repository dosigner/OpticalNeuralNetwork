#!/bin/bash
# 0405: Train D2NN for each distance with optimal settings
# Loss: focal_raw_received_power (best from 0403 combined sweep)
# f = 6.5mm (SMF-28 optimal coupling)
# Data: distance_sweep_cn2_5e-14/L{distance}m

set -e
cd /root/dj/D2NN/kim2026
LOG="autoresearch/runs/0405-distance-train.log"
echo "$(date) Starting distance sweep training..." > "$LOG"

for L in 100 500 1000 2000 3000; do
  DATA_PATH="data/kim2026/distance_sweep_cn2_5e-14/L${L}m"
  OUT_DIR="autoresearch/runs/0405-distance-sweep-rawrp-f6p5mm/L${L}m"

  # Check data exists
  if [ ! -f "$DATA_PATH/split_manifest.json" ]; then
    echo "$(date) SKIP L=${L}m — no data yet" >> "$LOG"
    continue
  fi

  echo "$(date) === Training L=${L}m ===" >> "$LOG"

  PYTHONPATH=src python -u -c "
import sys
sys.argv = ['sweep']
import autoresearch.d2nn_focal_pib_sweep as sweep

cfg = {
    'training': {'seed': 20260405, 'lr': 8e-3, 'epochs': 30, 'batch_size': 32, 'tv_weight': 0.05, 'warmup_epochs': 5},
    'architecture': {'num_layers': 5, 'layer_spacing_m': 0.01, 'detector_distance_m': 0.01, 'propagation_pad_factor': 2},
    'physics': {'wavelength_m': 1.55e-6, 'receiver_window_m': 0.002048, 'aperture_diameter_m': 0.002, 'focus_f_m': 6.5e-3, 'pib_bucket_radius_um': 10.0},
    'data': {'path': '${DATA_PATH}', 'plane_selector': 'stored'},
    'output': {'dir': '${OUT_DIR}'},
    'strategies': ['focal_raw_received_power'],
}
sweep.apply_config_overrides(cfg)

import torch
from torch.utils.data import DataLoader
from kim2026.data.dataset import CachedFieldDataset

device = torch.device('cuda')
print(f'L=${L}m, f={sweep.FOCUS_F_M*1e3:.1f}mm, loss=focal_raw_received_power')

train_ds = CachedFieldDataset(cache_dir=str(sweep.DATA_DIR), manifest_path=str(sweep.MANIFEST), split='train', plane_selector=sweep.DATA_PLANE_SELECTOR)
val_ds = CachedFieldDataset(cache_dir=str(sweep.DATA_DIR), manifest_path=str(sweep.MANIFEST), split='val', plane_selector=sweep.DATA_PLANE_SELECTOR)
test_ds = CachedFieldDataset(cache_dir=str(sweep.DATA_DIR), manifest_path=str(sweep.MANIFEST), split='test', plane_selector=sweep.DATA_PLANE_SELECTOR)
print(f'Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}')

train_loader = DataLoader(train_ds, batch_size=sweep.TRAIN['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=16, num_workers=0)

active = {k: v for k, v in sweep.LOSS_CONFIGS.items() if k in cfg['strategies']}
for name, config in active.items():
    sweep.train_one(name, config, train_loader, val_loader, test_loader, device, {})
print('Done: L=${L}m')
" >> "$LOG" 2>&1

  echo "$(date) Finished L=${L}m" >> "$LOG"
done

echo "$(date) ALL DISTANCE TRAINING COMPLETE!" >> "$LOG"
