# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproducible simulation/training stack for diffractive deep neural networks (D2NN), aligned with the 2018 Science paper "All-optical machine learning using diffractive deep neural networks." Python 3.10+, PyTorch-based.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e .[dev]

# Run all tests
pytest

# Run a single test file
pytest tests/test_asm_energy.py

# Train classifier (MNIST or FashionMNIST)
d2nn-train-classifier --config configs/mnist_phase_only_5l.yaml

# Train imager
d2nn-train-imager --config configs/imaging_lens_5l.yaml

# Evaluate a checkpoint
d2nn-eval --config configs/mnist_phase_only_5l.yaml --checkpoint path/to/final.pt

# Export height maps from checkpoint
d2nn-export-heightmap --config configs/mnist_phase_only_5l.yaml --checkpoint path/to/final.pt

# Generate figures from a run directory
d2nn-make-figures --run-dir runs/...
```

## Architecture

The package lives in `src/d2nn/` with strict module separation:

- **physics/** — Grid construction (`grid.py`), Angular Spectrum Method propagation (`asm.py`) with cached transfer functions, material properties (`materials.py`), aperture padding (`apertures.py`). All computations use SI units (meters, radians).
- **models/** — `DiffractionLayer` (trainable phase mask with sigmoid constraint to [0, φ_max], optional amplitude), `PropagationLayer` (output-plane propagation), `D2NNModel` (stacks layers). Factory: `build_d2nn_model()`.
- **detectors/** — `DetectorLayout` (physical detector regions from JSON), `build_region_masks()` (regions → pixel boolean masks), `integrate_regions()` (sum intensity per detector).
- **training/** — `losses.py` (cross-entropy + leakage penalty for classification, MSE for imaging), `loops.py` (train_classifier/train_imager epoch loops), `callbacks.py` (checkpoint/metrics/config saving), `sweeps.py` (architecture sweeps).
- **data/** — Dataset wrappers converting images to complex optical fields: `MNISTFieldDataset` (amplitude encoding), `FashionMNISTFieldDataset` (phase encoding), `ImageFolderFieldDataset`. Preprocessing in `preprocess.py` (resize, encode, pad).
- **viz/** — Plotting functions for phase masks, intensity fields, confusion matrices, energy distribution heatmaps, detector overlays, propagation cross-sections. Uses deterministic matplotlib styling (`style.py`).
- **export/** — Height map conversion (phase → physical height via Δn), Lumerical FDTD builder for FSP geometry files.
- **cli/** — Entry points. Shared setup in `common.py` (config loading, device selection, model/dataloader/detector construction).
- **utils/** — Seed management (`set_global_seed`), YAML/JSON/NPY I/O, run directory hashing (`resolve_run_dir`), math helpers.
- **types.py** — Core dataclasses: `DetectorRegionConfig`, `RunConfig`.

### Data Flow

Config (YAML) → `RunConfig` → seed + model + dataloaders + detector masks → training loop → checkpoint + metrics + figures saved to `runs/{exp_name}/{hash}/`.

### Key Physics

- Propagation via Angular Spectrum Method: FFT → multiply by transfer function H(fx,fy) → IFFT
- Transfer functions are cached by `(N, dx, wavelength, z, n, bandlimit)` tuple
- Default: complex64 (float32 components); complex128 optional via config
- Classification: detector regions integrate output intensity → logits via temperature scaling → cross-entropy + leakage penalty
- Imaging: MSE loss on output intensity, evaluated with SSIM

## Non-negotiables (from AGENT.md)

1. **SI units internally** — all lengths in meters, angles in radians
2. **Document tensor shapes** in public API docstrings
3. **Deterministic output** for fixed config + seed
4. **Module separation** — physics, training, visualization, and export stay separate
5. **Reusable plotting functions** — figures generated via `viz/` module, not inline

## Code Style

- Full type hints on all public APIs
- Docstrings must include units and tensor shapes
- Inline comments explain *why*, not *what*
- Phase wrapping conventions and FFT ordering must be documented where used
- Frozen dataclasses for configuration and detector definitions

## Configuration

YAML configs in `configs/` drive all experiments. Detector layouts are separate JSON files in `configs/layouts/`. Key config sections: `experiment`, `physics`, `model`, `data`, `training`, `loss`, `detector_layout`, `error_model`, `runtime`, `export`.

## Run Outputs

```
runs/{exp_name}/{run_id_hash}/
├── config_resolved.yaml
├── checkpoints/final.pt
├── metrics.json
├── figures/   (phase masks, confusion matrices, energy heatmaps, etc.)
└── exports/   (height map .npy files, optional .fsp)
```