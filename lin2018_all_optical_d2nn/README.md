# D2NN Reproducible Simulation

This repository implements a reproducible simulation/training stack for diffractive deep neural networks (D2NN), aligned with the 2018 Science D2NN paper.

## Scope
- Physics core: scalar wave + Angular Spectrum propagation
- Tasks: MNIST/Fashion classification and imaging lens training
- Reproducibility: config-driven runs, deterministic seeds, fixed artifacts
- Export: phase-to-height conversion and optional Lumerical pipeline

## Install
```bash
pip install -e .[dev]
```

## Run
```bash
d2nn-train-classifier --config configs/mnist_phase_only_5l.yaml
d2nn-train-imager --config configs/imaging_lens_5l.yaml
d2nn-eval --config configs/mnist_phase_only_5l.yaml --checkpoint path/to/checkpoint.pt
d2nn-export-heightmap --config configs/mnist_phase_only_5l.yaml --checkpoint path/to/checkpoint.pt
d2nn-make-figures --run-dir runs/...
d2nn-simulate-wave-panels --output-dir runs/wave_panels --num-layers 10 --num-segments 10 --mask-mode optimized --opt-steps 300
d2nn-simulate-wave-panels --input-field-npy /tmp/input_field.npy --output-dir runs/wave_panels_from_input
```

## Test
```bash
pytest
```
