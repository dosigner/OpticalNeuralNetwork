# D2NN Reproducible Simulation PRD

## Goal
Reproduce the main numerical and visualization outputs from the 2018 D2NN paper with deterministic, config-driven code.

## Must-have
- Scalar wave optics with ASM propagation
- MNIST/Fashion detector-based classification
- Imaging lens training and SSIM evaluation
- Reproducible artifacts: checkpoints, metrics, figures

## Optional in this implementation
- Error source modeling (misalignment, absorption, quantization)
- Lumerical export pipeline

## Success criteria
- MNIST test accuracy >= 0.90
- Fashion-MNIST test accuracy >= 0.78
- Imaging SSIM better than free-space baseline
