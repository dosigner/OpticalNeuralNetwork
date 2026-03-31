# D2NN Project — Codex / Agent Instructions

## Python Environment
- Conda env: `d2nn` at `miniconda3/envs/d2nn/bin/python` relative to the repo root
- Always use `PYTHONPATH=src` when running project scripts.

## Experiment Reproduction
Binary files (.pt, .npy, .npz) in `kim2026/autoresearch/runs/` are gitignored.
To regenerate checkpoints and data from committed configs:

```bash
# List available experiments
./scripts/reproduce_experiments.sh list

# Run a specific experiment
./scripts/reproduce_experiments.sh <name> --gpu <GPU_ID>

# Available experiments:
#   telescope_sweep      — D2NN loss sweep on telescope data (~2h)
#   loss_sweep_prelens   — Pre-lens PIB/Strehl/IO/hybrid sweep (~3h)
#   theorem_verify       — Unitary theorem verification (~5min)
#   co_sweep_strong      — Complex overlap sweep, strong turbulence (~2h)
#   paper_figures        — Static D2NN paper figure generation (~10min)
#   focal_pib            — Focal-plane PIB sweep, 4 loss strategies (~10h)

# Run all experiments
./scripts/reproduce_experiments.sh --gpu 0

# Dry run (show commands without executing)
./scripts/reproduce_experiments.sh --dry-run
```

### Prerequisites
1. Training data in `kim2026/data/` (generate: `cd kim2026 && PYTHONPATH=src python -m kim2026.data.generate`)
2. GPU available (check: `nvidia-smi`)

## Project Structure
- `kim2026/src/` — Core D2NN/FD2NN models, optics, training code
- `kim2026/autoresearch/` — Experiment sweep scripts and results
- `kim2026/docs/` — Reports, plans, design docs
- `luo2022_random_diffusers_d2nn/` — Luo 2022 reproduction
- `lin2018_all_optical_d2nn/` — Lin 2018 reproduction
- `tao2019_fourier_space_d2nn/` — Tao 2019 (FD2NN) reproduction
- `docs/optica_paper/` — Optica journal submission
- `scripts/` — Utility scripts

## Physics Notes
- Propagation: BL-ASM (Band-Limited Angular Spectrum Method)
- Standard optics: f=25mm Thorlabs AC127-025-C, dx_fourier=37.8μm
- Energy conservation: flag if >10% loss at any stage
- Baselines (CO/IO, Strehl): compute from raw input beams, NOT D2NN outputs
