# F-D2NN ECSSD Saliency Detection: Loss Function Experiment Report

**Date**: 2026-03-06
**Task**: ECSSD saliency detection using Fourier-space Diffractive Deep Neural Network (F-D2NN)
**Objective**: Improve blob-like output by replacing MSE loss with structured multi-component loss

---

## 1. Background

F-D2NN (Tao et al. 2019) uses phase-only modulation in Fourier domain with a 4f optical system (dual 2f lenses) for saliency detection on ECSSD dataset. The baseline model produces center-biased blob outputs with F_max = 0.5461, failing to capture object-specific spatial structure.

### Prior Fixes Applied
- **SBN nonlinearity disabled**: Per-layer SBN with per_sample_minmax normalization was destroying spatial structure
- **init_scale bug fixed**: Config's `init_scale: 0.1` was silently ignored; PhaseMask always used uniform(-1, 1)

These fixes improved F_max from 0.5461 to 0.5551 (+1.6%), but blob pattern persisted.

### Hypothesis
MSE loss allows the model to exploit ECSSD's center bias — producing a uniform center blob minimizes pixel-wise error without learning actual saliency structure. A loss function that explicitly penalizes this behavior and rewards spatial overlap should improve results.

---

## 2. Method

### 2.1 Structured Multi-Component Loss

Replaced `L = MSE(pred, gt)` with:

```
L_total = w_bce * L_bce + w_iou * L_iou + w_struct * L_structure + w_cp * L_center_penalty
```

| Component | Formula | Purpose |
|---|---|---|
| **L_bce** | Binary Cross-Entropy | Sharper decision boundaries than MSE; penalizes confident wrong predictions more heavily |
| **L_iou** | 1 - Soft IoU | Global spatial overlap; directly optimizes the intersection-over-union metric |
| **L_structure** | MSE(Sobel(pred), Sobel(gt)) | Edge-aware loss; encourages learning object boundaries instead of smooth blobs |
| **L_center_penalty** | Gaussian-weighted center mass | Penalizes predictions that concentrate energy in the center regardless of GT |

### 2.2 Implementation Details

- **BCE numerical stability fix**: Initial implementation used `eps=1e-8` for clamping, which caused `inf` loss in float32 due to `1.0 - (1.0 - 1e-8) = 0.0` precision loss. Fixed by using `eps=1e-6` with PyTorch's `F.binary_cross_entropy`.
- Soft IoU is fully differentiable (no hard thresholding)
- Sobel gradients computed via `F.conv2d` with fixed 3x3 kernels
- Center penalty uses Gaussian weight map (sigma=0.25 of image size)

### 2.3 Experimental Configurations

| Config | w_bce | w_iou | w_struct | w_cp | lr |
|---|---|---|---|---|---|
| Structured (default) | 1.0 | 2.0 | 1.0 | 0.1 | 0.005 |
| IoU dominant | 0.5 | 5.0 | 0.5 | 0.1 | 0.005 |

Both use: 5-layer F-D2NN, SBN off, init_scale=0.1, sigmoid phase constraint, cosine LR schedule, 100 epochs, batch_size=10.

---

## 3. Results

### 3.1 Quantitative Comparison

| # | Experiment | F_max | Best Epoch | Delta vs Baseline |
|---|---|---|---|---|
| 1 | MSE baseline (SBN on) | 0.5461 | 100 | -- |
| 2 | SBN tuning (failed) | 0.5290 | 145 | -3.1% |
| 3 | MSE + SBN off + init_scale fix | 0.5551 | 20 | +1.6% |
| 4 | **Structured loss (default)** | **0.5633** | **30** | **+3.1%** |
| 5 | **IoU dominant** | **0.5663** | **55** | **+3.7%** |

### 3.2 Key Observations

1. **IoU dominant achieves best F_max (0.5663)**: IoU loss가 spatial overlap을 직접 최적화하여 가장 효과적
2. **Structured loss consistently outperforms MSE**: 두 structured loss 변형 모두 MSE baseline 대비 유의미한 개선
3. **Convergence speed**: Structured loss는 epoch 30에서 best, IoU dominant는 epoch 55에서 best — baseline의 epoch 100 대비 빠른 수렴
4. **Loss scale difference**: Structured loss (~2.9)와 IoU dominant (~4.3)는 MSE (~0.18) 대비 절대값이 크지만, F_max 기준으로는 개선됨

### 3.3 Validation Loss Trends

| Experiment | Val Loss (best) | Val Loss (last) | Overfitting |
|---|---|---|---|
| MSE baseline | 0.1799 | 0.1807 | Minimal |
| MSE + SBN off | 0.1780 | 0.1783 | Minimal |
| Structured | 2.8996 | 2.8999 | Minimal |
| IoU dominant | 4.2875 | 4.2876 | Minimal |

All configurations show stable convergence without significant overfitting.

---

## 4. File Changes

### Modified Files

| File | Change |
|---|---|
| `src/tao2019_fd2nn/training/losses.py` | Added `binary_cross_entropy_loss`, `iou_loss`, `structure_loss`, `center_bias_penalty`, `saliency_structured_loss` |
| `src/tao2019_fd2nn/training/trainer.py` | Added `loss_mode` and `loss_weights` parameters to `run_saliency_epoch` and `train_saliency` |
| `src/tao2019_fd2nn/cli/train_saliency.py` | Parse `loss_mode` and `loss_weights` from YAML config |

### New Config Files

| File | Description |
|---|---|
| `config/saliency_ecssd_f2mm_structured_loss.yaml` | BCE(1.0) + IoU(2.0) + Structure(1.0) + CenterPenalty(0.1) |
| `config/saliency_ecssd_f2mm_iou_dominant.yaml` | BCE(0.5) + IoU(5.0) + Structure(0.5) + CenterPenalty(0.1) |

### Bug Fix

- **BCE float32 precision bug**: `eps=1e-8` -> `1e-6`. In float32, `1.0 - (1.0 - 1e-8) = 0.0` exactly, causing `log(0) = -inf` and NaN gradients from the first training step.

---

## 5. Output Figures

### IoU Dominant (Best: F_max = 0.5663)
```
runs/saliency_ecssd_f2mm_iou_dominant/260306_042843/figures/
  saliency_grid.png          -- Input / GT / Prediction comparison
  saliency_diagnostics.png   -- Detailed diagnostic visualization
  phase_masks.png            -- Learned phase masks (5 layers)
  convergence_saliency.png   -- Loss and F_max convergence curves
  pr_curve.png               -- Precision-Recall curve
```

### Structured Loss Default (F_max = 0.5633)
```
runs/saliency_ecssd_f2mm_structured_loss/260306_042608/figures/
  (same figure set)
```

### MSE Baseline (F_max = 0.5461)
```
runs/saliency_ecssd_f2mm/260304_085630/figures/
  (same figure set)
```

---

## 6. Discussion

### What Worked
- **IoU loss as dominant component** is the most effective single change. It directly optimizes spatial overlap, preventing the model from "cheating" with center-biased blobs.
- **BCE loss** provides sharper gradients than MSE near 0/1 boundaries, improving binary segmentation quality.
- **Structure loss** (Sobel gradient matching) adds edge-awareness, though its contribution is secondary to IoU.
- **Center penalty** provides a mild regularization against center bias exploitation.

### Remaining Limitations
- F_max of 0.5663 is still modest for saliency detection (state-of-art deep learning methods achieve >0.9 on ECSSD)
- This is expected given the **architectural constraints** of phase-only Fourier filtering:
  - Phase-only modulation acts as an all-pass filter (no amplitude control)
  - Only nonlinearity is the final |u|^2 detector readout
  - Global Fourier operation lacks local receptive fields
  - 5-layer network with ~128K learnable parameters (160x160 x 5 layers)

### Potential Next Steps
1. **Amplitude modulation**: Allow complex-valued (phase + amplitude) masks for richer filtering
2. **Hybrid real/Fourier space**: Combine Fourier-domain layers with real-space propagation layers
3. **Deeper network**: Increase from 5 to 10+ layers
4. **Multi-scale loss**: Add losses at different spatial resolutions
5. **Focal loss variant**: Weight hard examples more heavily
6. **Learnable nonlinearity**: Re-introduce SBN with better initialization/normalization

---

## 7. Reproducibility

```bash
# Structured loss (default weights)
python -m tao2019_fd2nn.cli.train_saliency \
  --config src/tao2019_fd2nn/config/saliency_ecssd_f2mm_structured_loss.yaml

# IoU dominant (best result)
python -m tao2019_fd2nn.cli.train_saliency \
  --config src/tao2019_fd2nn/config/saliency_ecssd_f2mm_iou_dominant.yaml
```

Environment: PyTorch 2.10.0+cu128, CUDA, seed=42.
