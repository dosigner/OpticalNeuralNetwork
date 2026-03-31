# FD2NN Spacing Sweep Report (f=10mm)

> Generated: spacing sweep with corrected focal length
> dx_fourier = 7.57 µm (4.9λ) — fabrication-realistic

## 1. Executive Summary

- **Best spacing**: 6 mm (z/z_R = 0.52)
- **Complex Overlap**: 0.3156 (+64.9% vs baseline 0.1913)
- **Phase RMSE**: 1.356 rad (77.7°)
- **Intensity Overlap**: 0.9010 (baseline: 0.9725)
- **Strehl Ratio**: 2.231

## 2. System Parameters

| Parameter | Value |
|-----------|-------|
| Wavelength λ | 1.55 µm |
| Grid size n | 1024 |
| Input pixel pitch dx | 2.0 µm |
| Receiver window | 2.048 mm |
| Focal length f | 10 mm |
| Fourier pixel pitch dx_f | 7.57 µm (4.9λ) |
| Fourier window | 7.750 mm |
| Numerical aperture NA | 0.16 |
| Number of layers | 5 |
| Phase constraint | symmetric_tanh, 2π |
| Training | 30 epochs, lr=5e-4, batch=2 |

## 3. Spacing Configurations

| Config | Spacing | z/z_R | Physical Regime |
|--------|---------|-------|----------------|
| spacing_0mm | 0 mm | 0.00 | Stacked (no propagation) |
| spacing_1mm | 1 mm | 0.09 | Near-field |
| spacing_3mm | 3 mm | 0.26 | Near-field |
| spacing_6mm | 6 mm | 0.52 | Intermediate (diffraction coupling) |
| spacing_12mm | 12 mm | 1.03 | Intermediate (diffraction coupling) |
| spacing_25mm | 25 mm | 2.15 | Far-field approach |
| spacing_50mm | 50 mm | 4.31 | Far-field approach |

z_R = π·(10·dx_f)²/λ = 11.61 mm (Rayleigh range for 10-pixel feature)

## 4. Results

### 4.1 Metrics Summary

| Config | z/z_R | CO ↑ | IO ↑ | Phase RMSE ↓ | Strehl |
|--------|-------|------|------|-------------|--------|
| spacing_0mm | 0.00 | 0.1649 | 0.9226 | 1.749 | 2.552 |
| spacing_1mm | 0.09 | 0.1966 | 0.9323 | 1.691 | 2.042 |
| spacing_3mm | 0.26 | 0.2750 | 0.9139 | 1.520 | 2.161 |
| spacing_6mm | 0.52 | 0.3156 **best** | 0.9010 | 1.356 | 2.231 |
| spacing_12mm | 1.03 | 0.3126 | 0.8898 | 1.315 | 2.617 |
| spacing_25mm | 2.15 | 0.3023 | 0.8894 | 1.294 | 2.657 |
| spacing_50mm | 4.31 | 0.3046 | 0.8843 | 1.285 | 2.759 |
| Baseline (no D2NN) | — | 0.1913 | 0.9725 | — | — |

### 4.2 Training Convergence

![Epoch Curves](../../figures/spacing_sweep_f10mm/fig1_epoch_curves.png)

### 4.3 Test Metrics

![Test Metrics](../../figures/spacing_sweep_f10mm/fig2_test_metrics.png)

### 4.4 Field Comparison

![Field Full](../../figures/spacing_sweep_f10mm/fig3_field_full_comparison.png)

![Field Zoom](../../figures/spacing_sweep_f10mm/fig4_field_zoom_comparison.png)

### 4.5 Field Profiles

![Profiles](../../figures/spacing_sweep_f10mm/fig5_field_profiles.png)

### 4.6 Phase Mask Evolution

![Phase Masks](../../figures/spacing_sweep_f10mm/fig6_phase_masks.png)

### 4.7 Fresnel Number Analysis

![Fresnel Analysis](../../figures/spacing_sweep_f10mm/fig7_fresnel_analysis.png)

### 4.8 Phase Masks [0, 2π] All Spacings

![Phase Masks 0-2pi](../../figures/spacing_sweep_f10mm/fig8_phase_masks_0_2pi.png)

## 5. Physics Analysis

### 5.1 회절 커플링과 layer 간 spacing의 관계

FD2NN에서 layer 간 spacing은 angular spectrum 전파를 통해 회절 커플링을 제공한다.
transfer function H(fx,fy;z) = exp(j·kz·z)에서 z가 결정하는 것은:

- **z/z_R < 0.3 (near-field)**: H ≈ 1 (identity), layer 간 독립적.
  각 phase mask가 같은 spatial frequency를 보므로 중복 학습 우려.
- **z/z_R ≈ 1 (Rayleigh range)**: 최적 information mixing.
  회절이 인접 pixel feature를 혼합하여 layer 간 비선형적 표현력 증가.
- **z/z_R > 3 (far-field)**: 고주파 성분이 확산하여 소실.
  spatial bandwidth 감소, phase mask의 fine structure가 무의미해짐.

### 5.2 Angular Spectrum 관점의 해석

전파 transfer function의 passband는 |f| < 1/λ 로 제한된다.
NA=0.16 cutoff에서 최대 공간주파수는 103226 lp/m이고,
이는 Fourier plane에서 800 pixel에 해당한다.

spacing이 클수록:
1. 저주파 성분은 전파가 잘 되지만 (평면파에 가까움)
2. 고주파 성분은 빠르게 발산하여 window 밖으로 나감
3. 결과적으로 effective spatial bandwidth가 줄어들어
   phase mask의 pixel 수가 의미없게 됨

### 5.3 Metasurface Phase Pattern 해석

학습된 phase mask에서 관찰되는 패턴:

- **동심원 구조 (Fresnel lens)**: 빔의 초점을 조절하는 가장 기본적인 phase profile.
  r² 의존성이 있으며, 각 ring의 주기가 λ·f/r 에 비례.
- **Speckle-like 패턴**: 난류 보상을 위한 random phase correction.
  특정 입력 realization에 대한 conjugate phase를 근사.
- **Spacing이 작을 때**: 모든 layer가 비슷한 패턴 → 중복 (표현력 낭비)
- **Spacing이 적절할 때**: 각 layer가 다른 spatial scale의 correction 담당

### 5.4 CO vs IO Trade-off의 물리적 원인

Phase-only metasurface의 근본적 한계:

1. **에너지 보존**: |exp(jφ)| = 1이므로 amplitude reshaping 불가
2. CO 개선 = phase 정합 → 난류 wavefront 보상 성공
3. IO 저하 = amplitude profile 왜곡 → Gaussian beam shape 깨짐
4. Phase-only mask는 intensity를 redistribution 할 수 있지만,
   원하는 Gaussian profile로 복원하는 것은 불가능
5. 이 trade-off는 amplitude mask 추가 또는 multi-pass 구조로만 해결 가능

## 6. Conclusions

1. f=10mm 설계에서 dx_fourier=7.6µm으로 현실적 제작 가능
2. 최적 spacing은 6mm (z/z_R=0.52)에서 CO=0.3156
3. Near-field (z/z_R < 0.3)과 far-field (z/z_R > 3) 모두 suboptimal
4. Phase-only mask의 CO vs IO trade-off는 물리적 한계

## 7. Future Work

- [ ] 최적 spacing 주변 fine sweep (±30%)
- [ ] Layer 수 증가 (7, 10 layers) + 최적 spacing 조합
- [ ] Amplitude+phase mask (complex-valued mask) 구현
- [ ] Multiple turbulence strength (Cn²) 조건에서의 일반화 성능
- [ ] 물리적 제작 제약 조건 반영 (phase level quantization)
