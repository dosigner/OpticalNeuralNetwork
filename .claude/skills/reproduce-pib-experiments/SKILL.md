---
name: reproduce-pib-experiments
description: Reproduce kim2026 experiment checkpoints and run the full Train→Eval→Analyze pipeline
user_invocable: true
---

# Reproduce PIB Experiments

Regenerate gitignored binary files (.pt, .npy, .npz) and run the full experiment pipeline.

## Two Ways to Run

### 1. Full Pipeline (recommended)
Train → Evaluate → Analyze in one command:
```bash
cd kim2026 && ./scripts/run_pipeline.sh --gpu 0
```

Run a single phase:
```bash
./scripts/run_pipeline.sh --phase 1 --gpu 0   # Train only
./scripts/run_pipeline.sh --phase 2 --gpu 0   # Evaluate + Visualize
./scripts/run_pipeline.sh --phase 3 --gpu 0   # Bucket radius sweep + paper figures
```

### 2. Individual Experiments
```bash
./scripts/reproduce_experiments.sh list                    # list all
./scripts/reproduce_experiments.sh focal_pib --gpu 0       # single experiment
./scripts/reproduce_experiments.sh --gpu 0                 # all experiments
```

### Available experiments
| Name | Description | Estimated Time |
|------|-------------|---------------|
| `telescope_sweep` | D2NN loss sweep on telescope data (5 configs) | ~2h |
| `loss_sweep_prelens` | Pre-lens PIB/Strehl/IO/hybrid sweep | ~3h |
| `theorem_verify` | Unitary theorem verification (no training) | ~5min |
| `co_sweep_strong` | Complex overlap sweep, strong turbulence | ~2h |
| `paper_figures` | Static D2NN paper figure generation | ~10min |
| `focal_pib` | Focal-plane PIB sweep (4 loss strategies) | ~10h |

## Config-Driven Experiments
Experiment parameters are in `autoresearch/configs/focal_pib_sweep.yaml`:
```bash
cd kim2026 && PYTHONPATH=src python -m autoresearch.d2nn_focal_pib_sweep \
    --config autoresearch/configs/focal_pib_sweep.yaml
```

## Physics Sanity Checks
Automatically run before/after training:
- Vacuum Strehl ~1.0 (optical pipeline integrity)
- Energy throughput [0.90, 1.10] (conservation)
- Unitary CO preservation |Δ| < 0.01 (theorem verification)
- Strehl upper bound ≤ 1.05 (passive device constraint)

## Prerequisites
1. Conda environment `d2nn` activated
2. Training data in `kim2026/data/` (generate: `cd kim2026 && PYTHONPATH=src python -m kim2026.data.generate`)

## Implementation
When the user asks to reproduce experiments:
1. Check if `kim2026/data/` exists — if not, warn that data generation is needed first
2. Check GPU availability with `nvidia-smi`
3. Recommend `run_pipeline.sh` for full workflow, `reproduce_experiments.sh` for individual reruns
4. For long-running experiments, provide copy-paste commands per CLAUDE.md guidelines
5. After completion, point to results in `autoresearch/runs/d2nn_focal_pib_sweep/`
