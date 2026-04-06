#!/bin/bash
# Wait for 0404 f-sweep to finish, then generate distance sweep data + train

LOG="/root/dj/D2NN/kim2026/autoresearch/runs/0405-distance-sweep.log"
echo "$(date) Waiting for 0404 to finish..." > "$LOG"

while true; do
  if ! ps aux | grep -v grep | grep "0404-focal-length" > /dev/null 2>&1; then
    echo "$(date) 0404 done. Starting distance sweep data generation..." >> "$LOG"
    break
  fi
  sleep 60
done

cd /root/dj/D2NN/kim2026

# Step 1: Generate data for all distances
echo "$(date) === DATA GENERATION ===" >> "$LOG"
PYTHONPATH=src python -u scripts/generate_data_distance_sweep.py >> "$LOG" 2>&1

# Step 2: Train abs_bucket for each distance
echo "$(date) === TRAINING ===" >> "$LOG"
for L in 100 500 1000 2000 5000; do
  DATA_PATH="data/kim2026/distance_sweep_cn2_5e-14/L${L}m"
  OUT_DIR="autoresearch/runs/0405-distance-sweep-absbucket/L${L}m"
  
  echo "$(date) Training L=${L}m..." >> "$LOG"
  
  PYTHONPATH=src python -u -c "
import sys, yaml
sys.argv = ['sweep', '--config', 'autoresearch/configs/0404-focal-length-sweep-absbucket.yaml']
import autoresearch.d2nn_focal_pib_sweep as sweep

cfg = {
    'training': {'seed': 20260405, 'lr': 8e-3, 'epochs': 30, 'batch_size': 32, 'tv_weight': 0.05, 'warmup_epochs': 5},
    'architecture': {'num_layers': 5, 'layer_spacing_m': 0.01, 'detector_distance_m': 0.01, 'propagation_pad_factor': 2},
    'physics': {'wavelength_m': 1.55e-6, 'receiver_window_m': 0.002048, 'aperture_diameter_m': 0.002, 'focus_f_m': 4.5e-3, 'pib_bucket_radius_um': 10.0},
    'data': {'path': '${DATA_PATH}', 'plane_selector': 'stored'},
    'output': {'dir': '${OUT_DIR}'},
    'strategies': ['focal_absolute_bucket'],
}
sweep.apply_config_overrides(cfg)

import torch
from torch.utils.data import DataLoader
from kim2026.data.dataset import CachedFieldDataset

device = torch.device('cuda')
print(f'L=${L}m, f={sweep.FOCUS_F_M*1e3:.1f}mm')

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

echo "$(date) ALL DISTANCE SWEEP COMPLETE!" >> "$LOG"
