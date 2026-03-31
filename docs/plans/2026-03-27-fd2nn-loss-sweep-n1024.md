# FD2NN Loss Sweep Plan — f=25mm, N=1024, No Crop

**Date**: 2026-03-26 (refined 2026-03-27)
**Feature**: fso / fd2nn-loss-sweep
**Status**: Plan (refined)

---

## Executive Summary

| Item | Value |
|------|-------|
| **Objective** | Loss function별 FD2NN beam cleanup 성능 비교 |
| **Grid** | 1024x1024, dx=2.0um, no crop |
| **Lens** | Thorlabs AC254-025-C (f=25mm, Ø25.4mm, NA=0.508) |
| **Configs** | 22개 → **15개** (512 결과 기반 pruning) |
| **Primary metric** | complex_overlap (higher=better) |
| **Expected time** | ~10h (15 configs x ~40min) |
| **Prerequisite** | Throughput diagnostic (zero-phase, 5min) |

---

## 0. Prerequisites: 512 Sweep Analysis + Throughput Diagnostic

### 0.1 512 Sweep Results (Axis 1 완료)

| Config | CO | vs baseline | 판정 |
|--------|------|-------------|------|
| roi80 | 0.2002 | +4.4% | **best** |
| roi70 | 0.1966 | +2.6% | good |
| roi90 | 0.1944 | +1.4% | marginal |
| baseline | 0.1917 | — | reference |
| roi60 | 0.1793 | -6.5% | weak |
| roi50 | 0.1685 | -12.1% | **drop** |

**Key findings**:

- Throughput 0.7598 동일 (전 config) → loss가 아닌 optical system property
- baseline_co_amp = baseline_co_only (identical) → amplitude_mse 무효
- roi80 > roi70 → 넓은 ROI가 유리
- roi50 < baseline → 너무 타이트한 ROI는 역효과

### 0.2 Config Pruning (22 → 15)

**Drop (7개)**: roi50, roi90, roi50_ph1, roi60_ph1, roi90_ph1, roi70_sph1_g3, roi70_ph1_lk025

| Category | Keep | Count |
|----------|------|-------|
| Baseline | baseline_co_amp, baseline_co_only | 2 |
| Axis 1: ROI | roi60, roi70, roi80 | 3 |
| Axis 2: Phasor | roi70_ph05/ph1/ph2, roi80_ph1 | 4 |
| Axis 3: Soft phasor | roi70_sph1_g2, roi70_sph1_g4 | 2 |
| Axis 4: Leakage | roi70_ph1_lk05, roi70_ph1_lk2 | 2 |
| Combo | roi60_ph1_sph05_g3, roi70_ph2_sph1_g3 | 2 |
| **Total** | | **15** |

Time savings: 15 × 40min ≈ 10h (vs 22 × 40min = 15h)

### 0.3 Throughput Diagnostic (FIRST STEP)

n=512에서도 TP=0.7598인데, fourier_window(9.69mm) < AC127 Ø(12.7mm)이라 **NA 클리핑이 아님**.
→ throughput 저하의 진짜 원인을 찾기 위해 zero-phase 진단 필수.

```python
# loss_sweep.py에 추가됨 (main() 시작부분)
model = make_model()  # NA=0.508 (AC254)
for layer in model.layers:
    layer.raw.data.zero_()  # 모든 mask phase = 0
tp_zero = throughput_check(model, test_loader, device)
```

**Decision gate**:

| Zero-phase TP | 의미 | 다음 행동 |
|---------------|------|----------|
| ≈ 1.0 | NA=0.508에서 에너지 보존 ✓ | Pilot → Full sweep |
| 0.9~0.95 | 약간의 diffraction loss | Sweep 진행 (알려진 한계) |
| < 0.9 | 구조적 문제 | `apply_scaling`, propagation 조사 |

**가능한 TP 원인들** (NA 클리핑 외):

1. `apply_scaling=False` — ortho FFT에서 에너지 비보존 가능
2. Inter-layer angular spectrum propagation (5mm) 시 edge diffraction
3. Phase mask 학습이 에너지 분산 (zero-phase로 배제 가능)

---

## 1. Optical System Parameters

### End-to-End Path

```
[1] TX: 1550nm, 0.3mrad full div
     ↓ 1km, Cn²=10⁻¹⁴, 6 phase screens
[2] RX telescope: D=15cm (빔 30cm의 25% 포착)
     ↓ Beam Reducer 75:1
[3] Receiver plane: 1024x1024, dx=2um, window=2.048mm
     ↓ Aperture D=2mm (NO crop)
[4] FD2NN: Lens₁→Masks×5→Lens₂
     ↓
[5] Focus lens → APD/MMF
```

### FD2NN Hardware

| Parameter | Value | Product |
|-----------|-------|---------|
| Lens f | 25 mm | Thorlabs AC254-025-C |
| Lens Ø | 25.4 mm | (1인치 achromatic doublet) |
| NA | 0.508 | |
| Phase mask grid | 1024×1024 | |
| Mask pixel pitch | 18.92 μm | SLM compatible |
| Mask physical size | 19.38 mm × 19.38 mm | |
| Layer count | 5 | |
| Layer spacing | **5 mm** | (변경: 1mm→5mm, 물리적 여유) |
| Total length | 25+20+25 = **70 mm** | |
| Parameters | 5,242,880 | |

### Fourier Plane Analysis

| Item | Value | 비고 |
|------|-------|------|
| dx_fourier | 18.92 μm | λf/(n×dx_in) |
| Fourier window | 19.38 mm | < Lens Ø 25.4mm ✓ |
| Vacuum beam spot | 12.3 μm | < 1 pixel (18.9μm) ⚠ |
| Turbulence speckle | 37 μm | ~2 pixels |
| D/r₀ | 1.9 | ~4 speckle modes |
| Fresnel N_F (5mm) | 0.046 | safe (max 237mm) |

### NA Upgrade Impact (n=512 → n=1024)

| Grid | dx_in | fourier_window | AC127 (Ø12.7mm) | AC254 (Ø25.4mm) |
|------|-------|----------------|------------------|------------------|
| n=512 | 4.0 μm | 9.69 mm | ✓ fits | ✓ fits |
| n=1024 | 2.0 μm | 19.38 mm | **✗ clips 35%** | ✓ fits |

NA=0.254 (AC127) cutoff freq: 163,871 m⁻¹ vs grid max freq: 250,000 m⁻¹
NA=0.508 (AC254) cutoff freq: 327,742 m⁻¹ > grid max freq → **zero clipping**

**코드 변경 완료**: `loss_sweep.py` ARCH dict에서 NA=0.254 → NA=0.508

### Known Issue: Fourier Plane Under-Resolution

Vacuum beam Fourier spot (12μm) < pixel pitch (18.9μm).
빔 에너지가 중앙 ~2x2 pixels에 집중. 대부분의 mask pixel이 미사용.
→ 이번 sweep에서 이 한계가 loss 선택에 어떤 영향을 주는지 관찰.
→ 추후: D_out=3mm sweep으로 비교 (Section 5b 참조).

---

## 2. Loss Configurations (15개, pruned)

### Strategy

```
L = ROI_complex(코어 모양) + phasor(φ_turb→0) + leak(에너지 가두기)
    + phasor_smoothness(mask 부드러움, weight=0.01)
```

### Config Table

| # | Name | Mode | ROI th | Leak wt | Phasor wt | Soft ph wt | γ | 512 결과 |
|---|------|------|--------|---------|-----------|------------|---|----------|
| **Baseline** | | | | | | | | |
| 1 | baseline_co_amp | composite | - | - | - | - | - | CO=0.191 |
| 2 | baseline_co_only | composite | - | - | - | - | - | CO=0.191 |
| **Axis 1: ROI 크기** | | | | | | | | |
| 3 | roi60 | ROI | 0.60 | 1.0 | - | - | - | CO=0.179 |
| 4 | roi70 | ROI | 0.70 | 1.0 | - | - | - | CO=0.197 |
| 5 | roi80 | ROI | 0.80 | 1.0 | - | - | - | **CO=0.200** |
| **Axis 2: ROI + phasor** | | | | | | | | |
| 6 | roi70_ph05 | ROI | 0.70 | 1.0 | 0.5 | - | - | pending |
| 7 | roi70_ph1 | ROI | 0.70 | 1.0 | 1.0 | - | - | pending |
| 8 | roi70_ph2 | ROI | 0.70 | 1.0 | 2.0 | - | - | pending |
| 9 | roi80_ph1 | ROI | 0.80 | 1.0 | 1.0 | - | - | pending |
| **Axis 3: soft_phasor gamma** | | | | | | | | |
| 10 | roi70_sph1_g2 | ROI | 0.70 | 1.0 | - | 1.0 | 2.0 | pending |
| 11 | roi70_sph1_g4 | ROI | 0.70 | 1.0 | - | 1.0 | 4.0 | pending |
| **Axis 4: leakage weight** | | | | | | | | |
| 12 | roi70_ph1_lk05 | ROI | 0.70 | 0.50 | 1.0 | - | - | pending |
| 13 | roi70_ph1_lk2 | ROI | 0.70 | 2.0 | 1.0 | - | - | pending |
| **Combo** | | | | | | | | |
| 14 | roi60_ph1_sph05_g3 | ROI | 0.60 | 1.0 | 1.0 | 0.5 | 3.0 | pending |
| 15 | roi70_ph2_sph1_g3 | ROI | 0.70 | 1.0 | 2.0 | 1.0 | 3.0 | pending |

**Dropped configs** (512 결과 기반):

- roi50: CO=0.169, baseline 이하
- roi90: CO=0.194, marginal improvement
- roi50_ph1, roi60_ph1, roi90_ph1: base ROI가 약하므로 phasor 추가해도 무의미
- roi70_sph1_g3: g2/g4 endpoints로 충분히 보간 가능
- roi70_ph1_lk025: 약한 leakage는 정보가 적음

---

## 3. Visualization Plan

### 3.1 Stage-by-Stage Beam Comparison (per config)

각 config 학습 완료 후, test set sample 1개에 대해 아래 figure 생성:

```
Row 1: Irradiance |u|²
  [Turbulent input] [FD2NN corrected] [Vacuum target] [Difference]

Row 2: Phase angle(u)
  [Turbulent phase]  [Corrected phase]  [Vacuum phase]  [Residual: corrected-vacuum]

Row 3: Fourier plane
  [Input spectrum]   [After masks]      [Mask phase φ₀]  [Mask phase φ₄]

Row 4: Focused image (simulated APD plane)
  [Turbulent focus]  [Corrected focus]  [Vacuum focus]   [Airy reference]
```

### 3.2 Cross-Config Comparison Dashboard

15개 config 결과를 한 figure에:

```
Panel A: Bar chart — complex_overlap ranked (with baseline line)
Panel B: Bar chart — throughput (with 0.90 threshold line)
Panel C: Scatter — co vs throughput (Pareto front)
Panel D: ROI threshold vs co curve (Axis 1)
Panel E: Phasor weight vs co curve (Axis 2)
Panel F: Table — all metrics
```

### 3.3 Focused Beam (APD Plane) Visualization

FD2NN 출력 → 집속 렌즈(f=4.5mm) → detector plane 시뮬레이션:

```
propagate(corrected_field, f=4.5mm) → focused_field

Metrics:
  - Airy disk diameter: 8.5 μm
  - Power in bucket (PIB): energy within 50μm circle / total
  - Coupling efficiency: |<focused, fiber_mode>|² / (|focused|²·|mode|²)
  - Received power: throughput × PIB × P_input
```

### 3.4 New Diagnostic Panels (추가)

- **Fourier-plane diagnostic**: config별 power spectrum before/after correction
- **Phase mask visualization**: top 3 configs의 5-layer phase mask 시각화
- **512 vs 1024 comparison**: 동일 config (roi70, roi80)의 grid size 효과

| Metric | Definition | Target |
|--------|-----------|--------|
| **Throughput** | Σ\|output\|² / Σ\|input\|² | > 0.90 |
| **PIB** | Energy in 50μm bucket / total output energy | > 0.80 |
| **Coupling efficiency** | Mode overlap with MMF fundamental mode | > 0.50 |
| **Received power** | throughput × PIB × input_power | maximize |
| **Complex overlap** | \|⟨pred, target⟩\| / (\|\|pred\|\|·\|\|target\|\|) | maximize |
| **Improvement ratio** | (CO_trained - CO_baseline) / CO_baseline | > 5% |
| **TP-corrected CO** | CO × √(throughput) | maximize |
| **ROI phase flatness** | RMS residual phase within ROI mask | minimize |

---

## 4. Execution Steps

### Step 0: Throughput Diagnostic (5분) — FIRST

```bash
# loss_sweep.py main()에 자동 포함됨
# NA=0.508로 zero-phase 모델의 throughput 측정
cd /root/dj/D2NN/kim2026 && python -m autoresearch.loss_sweep 2>&1 | head -20
```

→ "Zero-phase throughput: X.XXXX" 확인 후 decision gate

### Step 1: Pilot Run (3 configs, ~2h)

TP 진단 통과 후, loss_sweep.py의 LOSS_CONFIGS에서 pilot 3개만 활성화:

- `baseline_co_amp`, `roi70`, `roi80`

목적: NA=0.508에서 CO improvement 확인, training time 추정

**Decision gate**:

- CO improvement > 5% over baseline → proceed to full sweep
- CO improvement < 2% → architecture 조사 (D_out, layers, spacing)

### Step 2: Full Sweep (나머지 12개, ~8h)

Resume 로직 추가됨 — pilot에서 완료된 3개는 자동 skip.

```bash
cd /root/dj/D2NN/kim2026 && python -m autoresearch.loss_sweep 2>&1 | tee autoresearch/loss_sweep_f25mm_n1024.log
```

각 config에서 자동 저장:

- `results.json` — all metrics
- `checkpoint.pt` — model weights
- `phases_wrapped.npy` — phase masks [0,2π)
- `sample_fields.npz` — turbulent/corrected/vacuum fields

### Step 3: Visualization script

`autoresearch/visualize_n1024_sweep.py`:

- Section 3.1: per-config beam comparison (15 figures)
- Section 3.2: cross-config dashboard (1 figure)
- Section 3.3: focused beam + PIB/coupling (15 figures)
- Section 3.4: Fourier diagnostic + phase mask vis

### Step 4: Analysis and report

- Best loss config identification
- Throughput validation (all > 0.90?)
- 512 vs 1024 comparison (roi70, roi80)
- Fourier under-resolution impact analysis
- Recommendation for next sweep

---

## 5. Follow-up Sweeps

### 5a. Data Regeneration + Retrain Best Configs

**Strategy**: Sweep은 기존 데이터(half_angle=3e-4)로 먼저 실행.
Loss 비교는 유효 (동일 입력). 절대 metric은 재생성 데이터 필요.

Best 3 configs 확정 후:

1. `sweep_base_fd2nn.yaml` half_angle_rad: 1.5e-4 (이미 업데이트됨)
2. 새 데이터셋 생성: `1km_cn2e-14_w2m_n1024_dx2mm_ha15` (100 realizations, ~2-3h)
3. Best 3 configs만 재학습 (~2h)
4. Old vs new data 비교: half_angle가 absolute metric에 미치는 영향 정량화

### 5b. D_out=3mm Comparison Sweep

Fourier under-resolution 해소를 위한 별도 sweep.

| Parameter | D_out=2mm (현재) | D_out=3mm (비교) |
|-----------|-----------------|-----------------|
| dx_in | 2.0 μm | 2.93 μm |
| dx_fourier | 18.92 μm | 12.91 μm |
| Vacuum spot / pixel | 0.65 (under) | 0.95 (matched!) |
| fourier_window | 19.38 mm | 13.22 mm |
| Beam reducer | 75:1 | 50:1 |

변경 사항:

- `RECEIVER_WINDOW_M = 0.003072`
- `APERTURE_DIAMETER_M = 0.003`
- 새 데이터 생성 필요 (beam reducer 비율 변경)

Top 3-5 configs만 실행 (~5h).

---

## 6. Autoresearch (별도 세션)

이 sweep 결과에서 best loss config 확정 후, 별도 세션에서:

1. `program.md`에 best loss config 고정
2. LR schedule, epochs, init_scale 등 다른 hyperparameter 자율 탐색
3. autoresearch loop 실행 (overnight)

---

## 7. Open Questions

### Resolved

1. ~~**Fourier under-resolution**~~: D_out=3mm comparison sweep으로 대응 (Section 5b)
2. ~~**Layer spacing**~~: 5mm 확정 (N_F=0.046, safe, max 237mm)
3. ~~**Data regeneration**~~: Sweep 먼저 → best configs만 재생성 (Section 5a)

### Active

4. **Unconstrained phase + smoothness**: weight=0.01이 적절한지. 너무 약하면 고주파 산란, 너무 강하면 correction 억제. Sweep 결과에서 관찰.

5. **Throughput 원인**: n=512에서도 TP=0.7598 (NA 클리핑 없음). Zero-phase 진단으로 원인 규명 필요.
   - `apply_scaling=False` → FFT 에너지 비보존?
   - Inter-layer AS propagation (5mm) → edge diffraction?
   - Phase mask 학습 → 에너지 분산? (zero-phase로 배제 가능)

6. **roi80 > roi70**: 512에서 roi80이 best. 1024에서도 유지되는지? NA 변경 후 ranking 변화?

7. **NA=0.508 baseline effect**: NA 변경으로 baseline CO 자체가 개선될 가능성. 입력 빔이 더 많은 고주파를 포함 → 더 정확한 correction 가능?

---

## 8. Code Changes Log

| File | Change | Status |
|------|--------|--------|
| `autoresearch/loss_sweep.py:68-73` | NA: 0.254 → 0.508 (AC127 → AC254) | ✅ Done |
| `autoresearch/loss_sweep.py:344-349` | Resume-skip logic (results.json 존재 시 skip) | ✅ Done |
| `autoresearch/loss_sweep.py:main()` | Zero-phase throughput diagnostic 추가 | ✅ Done |
| `autoresearch/loss_sweep.py` | Config pruning (22 → 15) | Pending |
| `training/metrics.py` | PIB, TP-corrected CO, ROI phase flatness 추가 | Pending |
| `autoresearch/visualize_n1024_sweep.py` | Cross-config dashboard + Fourier diagnostic | Pending |
