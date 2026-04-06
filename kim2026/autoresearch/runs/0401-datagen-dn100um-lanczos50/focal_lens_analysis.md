# Focal Lens Selection: f = 4.5mm

## Airy Disk vs PIB Bucket

D2NN output beam: D = 2mm (aperture), λ = 1.55μm

```
Airy disk radius = 1.22 × λf/D = 1.22 × 1.55μm × 4.5mm / 2mm = 4.26μm
PIB bucket radius = 10μm = 2.35 × Airy radius
Focal plane pitch = λf/(N·dx) = 1.55μm × 4.5mm / (1024 × 2μm) = 3.4μm/pixel
```

## Focal Length Comparison

| f (mm) | Airy radius | dx_focal | 10μm bucket / Airy | Vacuum PIB@10μm | Verdict |
|-------:|----------:|---------:|--------------------:|----------------:|---------|
| 1.0 | 0.95μm | 0.76μm | 10.5× | ~100% | Bucket too large — no discrimination |
| 2.5 | 2.37μm | 1.89μm | 4.2× | ~99% | Almost always high — insensitive |
| **4.5** | **4.26μm** | **3.4μm** | **2.35×** | **95.5%** | **Selected — good sensitivity** |
| 10.0 | 9.46μm | 7.56μm | 1.06× | ~55% | Only partial main lobe — too sensitive |
| 25.0 | 23.7μm | 18.9μm | 0.42× | ~5% | Spot > bucket — unmeasurable |

## Why f=4.5mm Is Appropriate

### 1. Sensitivity (discrimination)

- Vacuum PIB@10μm = 95.5% (upper bound for perfect beam)
- Turbulence PIB@10μm = 83.8% (degraded by atmosphere, no D2NN)
- D2NN target: PIB@10μm > 90% (corrected)

The 95% → 84% gap provides meaningful gradient for D2NN training loss.

### 2. Focal Plane Sampling

- Airy disk diameter ≈ 8.5μm
- dx_focal = 3.4μm → ~2.5 pixels across Airy disk
- Satisfies Nyquist criterion (≥2 pixels per resolution element)

### 3. Physical Realizability

- 2mm beam + f=4.5mm → NA ≈ 0.22 (standard aspheric lens range)
- Focal spot ~8.5μm → compatible with InGaAs APD pixel sizes (10-25μm)
- Near-match to single-mode fiber MFD ~10μm @1550nm

## Conclusion

f = 4.5mm is well-chosen. The bucket/Airy ratio of 2.35× sits in the
"achievable but non-trivial" sweet spot, creating a good loss landscape
for D2NN training.

## Data Generation Verification (delta_n=100μm + Lanczos 50:1)

| Metric | Value | Pass Criterion |
|--------|-------|----------------|
| Vacuum WFE (higher-order) | 26.0 nm | < 50 nm ✓ |
| Vacuum PIB@10μm (collimated) | 95.5% | > 80% ✓ |
| Time per realization | 3.2 s | - |
| VRAM peak | 6.8 GB | - |
| Dataset | 5000 (4000/500/500) | - |
| Output path | `data/kim2026/1km_cn2_5e-14_tel15cm_dn100um_lanczos50/` | - |
