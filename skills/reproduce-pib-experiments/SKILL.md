---
name: reproduce-pib-experiments
description: Reproduce D2NN focal-plane PIB experiment checkpoints and run the full Train→Eval→Analyze pipeline. Use when asked to reproduce experiments, regenerate checkpoints, re-run training, evaluate trained models, or generate paper figures from kim2026/autoresearch.
metadata:
  short-description: Reproduce PIB experiments end-to-end
---

# Reproduce PIB Experiments

Regenerate gitignored binary files (.pt, .npy, .npz) and run the full experiment pipeline for the D2NN focal-plane PIB sweep.

## Prerequisites

1. **Data**: `kim2026/data/kim2026/1km_cn2_5e-14_tel15cm_n1024_br75/cache/` must exist
   - If missing: `cd kim2026 && PYTHONPATH=src python -m kim2026.data.generate`
2. **GPU**: Check with `nvidia-smi`
3. **Python**: `/root/dj/D2NN/miniconda3/envs/d2nn/bin/python`

## Full Pipeline

```bash
cd kim2026 && ./scripts/run_pipeline.sh --gpu 0
```

| Phase | What | Script | Time |
|-------|------|--------|------|
| 1 | Train 4 loss strategies | `d2nn_focal_pib_sweep.py` | ~10h |
| 2 | Eval + 6 viz figures | `eval_focal_pib_only.py` + `visualize_focal_pib_report.py` | ~20min |
| 3 | Bucket sweep + paper figs | `eval_bucket_radius_sweep.py` + `generate_focal_paper_figures.py` | ~30min |

Run single phase:
```bash
./scripts/run_pipeline.sh --phase 2 --gpu 0
```

## Individual Experiments

```bash
./scripts/reproduce_experiments.sh list
./scripts/reproduce_experiments.sh focal_pib --gpu 0
```

| Experiment | Module | Time |
|------------|--------|------|
| `telescope_sweep` | `autoresearch.d2nn_sweep_telescope` | ~2h |
| `loss_sweep_prelens` | `autoresearch.d2nn_loss_strategy_sweep` | ~3h |
| `theorem_verify` | `autoresearch.visualize_optical_path` | ~5min |
| `co_sweep_strong` | `autoresearch.d2nn_strong_turb_sweep` | ~2h |
| `paper_figures` | `autoresearch.visualize_optical_path` | ~10min |
| `focal_pib` | `autoresearch.d2nn_focal_pib_sweep` | ~10h |

## Config

Experiment parameters in `autoresearch/configs/focal_pib_sweep.yaml`. Override:
```bash
PYTHONPATH=src python -m autoresearch.d2nn_focal_pib_sweep --config autoresearch/configs/focal_pib_sweep.yaml
```

## Physics Sanity Checks (automatic)

Runs before/after training via `kim2026.eval.sanity_check`:
- Vacuum Strehl ≈ 1.0
- Throughput ∈ [0.90, 1.10]
- Unitary CO |Δ| < 0.01
- Strehl ≤ 1.05

## Output Location

All results: `kim2026/autoresearch/runs/d2nn_focal_pib_sweep/`
- `{strategy}/checkpoint.pt` — model weights (gitignored)
- `{strategy}/results.json` — metrics
- `{strategy}/phases_wrapped.npy` — learned phase masks (gitignored)
- `summary.json` — all strategies compared
- `bucket_radius_sweep/` — Phase 3 analysis
- `paper_figures/` — publication figures

## Shared Utilities

New eval/viz scripts should import from `kim2026.eval.focal_utils`:
```python
from kim2026.eval.focal_utils import (
    prepare_field, apply_focal_lens, load_checkpoint,
    compute_pib_torch, compute_ee_curve, load_test_dataset,
    FOCAL_STRATEGIES, STRATEGY_LABELS, STRATEGY_COLORS,
)
```
