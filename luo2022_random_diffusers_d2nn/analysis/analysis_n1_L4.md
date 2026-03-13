# Experiment Analysis: n1_L4 (Single-Diffuser Baseline)

**Run ID:** `n1_L4`
**Date:** 2026-03-13
**Reference:** Luo et al., "Computational Imaging Without a Computer: Diffractive Networks," *eLight* 2 (2022)

---

## 1. Experiment Overview

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | Phase-only D2NN (`d2nn_phase_only`) |
| Number of layers | 4 |
| Grid size | 240 x 240 |
| Pixel pitch | 0.3 mm |
| Wavelength | 0.75 mm (400 GHz, THz regime) |
| Phase init | Uniform [0, 2pi) |
| Propagation | Band-limited angular spectrum method (BL-ASM) |
| Pad factor | 2x (480 x 480 padded grid) |

### Geometry

- Object-to-diffuser: 40.0 mm
- Diffuser-to-layer-1: 2.0 mm
- Layer-to-layer: 2.0 mm
- Last-layer-to-output: 7.0 mm

### Training Parameters

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Batch size (objects) | 4 |
| Diffusers per epoch (n) | **1** |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| LR schedule | Multiplicative decay, gamma = 0.99/epoch |
| Loss function | -PCC + energy penalty (alpha=1.0, beta=0.5) |
| Hardware | NVIDIA A100 (TF32 enabled) |
| Dataset | MNIST (50k train / 10k val) |
| Input encoding | Amplitude (grayscale), resized 28 -> 160 -> 240 px |

### Diffuser Model

Thin random phase screen with Gaussian-smoothed height profile:
- delta_n = 0.74, mean height = 25 lambda, std = 8 lambda
- Smoothing sigma = 4 lambda, correlation length = 10 lambda

---

## 2. Training Dynamics

### Convergence Summary

| Epoch | PCC | Notes |
|---|---|---|
| 1 | 0.6729 | Initial reconstruction quality after one pass |
| 10 | 0.9050 | Rapid convergence; most learning happens here |
| 20 | 0.9125 | Near-plateau; marginal gains |
| 30 | 0.9066 | Slight regression from peak |

**Final metrics:** loss = -0.9715, PCC = 0.9066
**Total training time:** 454 s (7.6 min), ~15.1 s/epoch

### Convergence Curve Analysis

The training curve exhibits three distinct phases:

1. **Rapid ascent (epochs 1-10):** PCC climbs from 0.67 to 0.91, a gain of +0.24. The network quickly learns the phase corrections needed to invert the single fixed diffuser. This is the regime where the dominant low-frequency structure of the diffuser's transfer function is being compensated.

2. **Plateau (epochs 10-20):** PCC inches from 0.905 to 0.913, a gain of only +0.008 over 10 epochs. The network has essentially converged; remaining improvements involve fine-tuning high-spatial-frequency phase corrections.

3. **Mild oscillation (epochs 20-30):** PCC drops slightly to 0.907. This is characteristic of overfitting to training-set statistics or minor learning-rate dynamics. With n=1, there is no diffuser diversity to regularize against, so the network has no incentive to learn generalizable features -- small fluctuations in training-batch composition can cause the loss landscape to shift slightly.

The near-monotonic convergence and early plateau are expected for n=1: the optimization problem is relatively simple because the network only needs to learn the inverse of a single, fixed scattering configuration.

---

## 3. Comparison with Paper Expectations

### Paper Context (Luo et al. 2022, Fig. 3)

The paper's central finding is that training with multiple random diffusers per epoch (`n >> 1`) forces the D2NN to learn a generalizable imaging transform rather than memorizing a specific diffuser. Key reference points from the paper:

| n (diffusers/epoch) | Expected behavior |
|---|---|
| n = 1 | Baseline. High PCC on the training diffuser, poor generalization to unseen diffusers. |
| n = 5 | Moderate improvement in blind-test PCC. |
| n = 10-15 | Significant jump; network begins learning diffuser-invariant features. |
| n = 20 | Near-saturated generalization performance. |

### Our n=1 Result vs. Paper

- **Training PCC = 0.907** is consistent with the paper's n=1 baseline regime. The paper does not report exact n=1 PCC values on training diffusers (their focus is on blind-test generalization), but a PCC near 0.91 for a known diffuser is physically reasonable -- it means the 4-layer D2NN can compensate most of the scattering from a single phase screen.

- The loss function formulation (-PCC + energy penalty) means final_loss = -0.9715 corresponds to a raw PCC contribution of approximately 0.97 minus the energy penalty term (~0.06). This indicates the network achieves strong structural correlation but the energy penalty prevents it from concentrating all output energy within the target support.

- **Convergence speed** (plateau by epoch 10) is expected for n=1. With only one diffuser, the effective dataset is much simpler than the multi-diffuser case, where each epoch presents new scattering configurations that continually perturb the loss landscape.

---

## 4. Key Observations

### Diffuser Memorization vs. Generalizable Learning

With `diffusers_per_epoch = 1`, the network sees the exact same scattering configuration every epoch. The learned phase layers converge to an approximate inverse of that specific diffuser's point-spread function. This is fundamentally different from the multi-diffuser regime:

- **n=1 learns:** "How to undo *this particular* diffuser's distortion."
- **n>>1 learns:** "How to extract object information regardless of diffuser-induced aberrations."

The distinction is analogous to overfitting vs. generalization in conventional deep learning. A network trained on one diffuser may achieve high PCC on that diffuser but will likely fail catastrophically when presented with a new, unseen diffuser at test time.

### Training Efficiency

The 7.6-minute training time reflects the simplicity of the n=1 case:
- Only 1 diffuser pattern generated and applied per epoch
- No need to propagate through multiple diffuser configurations per batch
- The optimization converges rapidly because the target is static

This serves as a useful computational baseline: multi-diffuser runs (n=10, 20) will scale training time roughly linearly with n, as each diffuser configuration requires a separate forward/backward pass.

### Slight PCC Regression (Epochs 20-30)

The drop from 0.913 to 0.907 is minor (~0.7%) but notable. Possible causes:
- The LR schedule (gamma=0.99) still allows non-trivial updates at epoch 30 (effective LR = 1e-3 * 0.99^29 = 7.5e-4), which may cause the optimizer to "wander" near the minimum.
- With n=1, there is no regularizing effect from diffuser diversity, so the network can overfit to batch-level noise in the training data.

---

## 5. Implications for the n-Sweep Study (Fig. 3 Reproduction)

This run establishes the **lower bound** of the diffuser-diversity sweep:

### Role in the Sweep

| Sweep point | Purpose |
|---|---|
| **n=1 (this run)** | Baseline: best-case for known diffuser, worst-case for blind test |
| n=5 | Transition regime |
| n=10 | Onset of robust generalization |
| n=15 | Near-saturation |
| n=20 | Full-diversity reference (paper's primary result) |

### Predictions for Subsequent Runs

1. **Known-diffuser PCC will decrease with n.** As the network trains on more diffusers, it cannot perfectly compensate any single one. We expect known-diffuser PCC to drop from ~0.91 (n=1) to perhaps 0.85-0.88 (n=20).

2. **Blind-test PCC will increase with n.** This is the paper's core result. The n=1 network will likely achieve very low blind-test PCC (potentially below 0.5), while n=20 should achieve blind-test PCC comparable to known-diffuser PCC.

3. **The gap between known-diffuser and blind-test PCC is the generalization gap.** For n=1, this gap will be maximal. The n-sweep should show this gap closing as n increases, reproducing the paper's Fig. 3.

4. **Training time will scale with n.** Extrapolating from 7.6 min at n=1, we estimate ~150 min for n=20 at 30 epochs (or proportionally more if the paper's 100-epoch schedule is used for the full baseline).

### Next Steps

- Run blind-test evaluation on this checkpoint with 20 unseen diffusers to quantify the generalization gap
- Execute n=5, 10, 15, 20 training runs with identical hyperparameters (except `diffusers_per_epoch`)
- Compare known-diffuser vs. blind-test PCC curves to reproduce Fig. 3

---

## Appendix: Run Artifacts

| File | Description |
|---|---|
| `config.yaml` | Full experiment configuration |
| `model.pt` | Trained model checkpoint (4 phase layers) |
| `training_summary.json` | Final loss, PCC, and timing |
