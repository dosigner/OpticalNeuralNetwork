# Hyperparameters: Complex Field D2NN Beam Cleanup

> Reference config: `configs/fso_1024_complex.yaml`
> Date: 2026-03-23

---

## 1. D2NN Model Hyperparameters

### 1.1 Architecture

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Layers | `model.num_layers` | 5 | Phase-only layer 수 |
| Layer spacing | `model.layer_spacing_m` | 50.0 m | Layer 간 자유공간 전파 거리 |
| Detector distance | `model.detector_distance_m` | 50.0 m | 마지막 layer → detector |
| Phase init | (hardcoded) | `zeros(N,N)` | 학습 시작 위상 (all zero = 투과) |
| Phase wrap | `model.phase_wrap` | True | `remainder(phi, 2pi)` 적용 |
| Total D2NN path | (derived) | 250 m | (5-1) x 50 + 50 |

**Fresnel number per layer**:
```
N_F = dx^2 / (lambda * z) = (2mm)^2 / (1.55um * 50m) = 0.052
```
Deep Fresnel regime → phase-only layer에서 amplitude redistribution 발생.
이것이 50m spacing을 쓰는 이유 (mm급이면 N_F >> 1로 phase만 바뀜).

### 1.2 Training

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Epochs | `training.epochs` | 40 | 총 학습 epoch |
| Batch size | `training.batch_size` | 2 | 학습 배치 크기 |
| Learning rate | `training.learning_rate` | 5e-4 | Adam initial LR |
| Optimizer | (hardcoded) | Adam | `trainer.py:157` |
| LR schedule | (없음) | constant | decay 미구현 |
| Eval batch size | `training.eval_batch_size` | 2 | 평가시 배치 |

### 1.3 Loss Function

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Loss mode | `training.loss.mode` | `"complex"` | Complex field 학습 |
| complex_overlap weight | `training.loss.complex_weights.complex_overlap` | 1.0 | Phase-sensitive overlap |
| amplitude_mse weight | `training.loss.complex_weights.amplitude_mse` | 0.5 | Amplitude MSE |

Intensity mode fallback (backward compat):

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| overlap weight | `training.loss.weights.overlap` | 1.0 | Normalized overlap |
| radius weight | `training.loss.weights.radius` | 0.25 | Beam radius MSE |
| encircled weight | `training.loss.weights.encircled` | 0.25 | Encircled energy MSE |

---

## 2. Optics / Channel Hyperparameters

### 2.1 Source (Transmitter)

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Wavelength | `optics.lambda_m` | 1.55e-6 m | 1550nm near-IR |
| Half-angle divergence | `optics.half_angle_rad` | 3e-4 rad | 0.3 mrad |
| M^2 | `optics.m2` | 1.0 | Ideal Gaussian |

Derived:

| Parameter | Value | Formula |
|-----------|-------|---------|
| Beam waist w0 | 1.645 mm | M^2 * lambda / (pi * theta_half) |
| Source window | 6.578 mm | ~4 * w0 (99.97% energy) |
| Rayleigh range z_R | 5.48 m | pi * w0^2 / (M^2 * lambda) |

### 2.2 Turbulence Channel

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Path length | `channel.path_length_m` | 1000 m | Tx → Rx 거리 |
| Cn^2 | `channel.cn2` | 1e-15 m^(-2/3) | Weak turbulence |
| Outer scale L0 | `channel.outer_scale_m` | 30.0 m | von Karman outer scale |
| Inner scale l0 | `channel.inner_scale_m` | 5e-3 m | von Karman inner scale (5mm) |
| Num screens | `channel.num_screens` | 6 | Path cell 수 (interior screens = 5) |
| Channel mode | `channel.mode` | `"static"` | Static snapshot, no frozen flow |
| Num realizations | `channel.num_realizations` | 200 | 독립 난류 snapshot 수 |

Derived turbulence parameters (Cn^2 = 1e-15):

| Parameter | Value | Formula |
|-----------|-------|---------|
| r0 (plane-wave) | ~0.72 m | (0.423 k^2 Cn2 Dz)^(-3/5) |
| r0 (spherical-wave) | ~1.13 m | (0.423 k^2 Cn2 3/8 Dz)^(-3/5) |
| Rytov variance sigma^2_chi | ~0.003 | << 0.25 (weak fluctuation) |
| D/r0 | ~0.21 | 0.15m / 0.72m (few speckles) |

### 2.3 Receiver

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Aperture diameter | `receiver.aperture_diameter_m` | 0.15 m | 15cm circular |
| Receiver window | `grid.receiver_window_m` | 2.048 m | 계산 영역 |
| Grid size N | `grid.n` | 1024 | 1024 x 1024 pixels |

Derived:

| Parameter | Value | Formula |
|-----------|-------|---------|
| dx (receiver) | 2.0 mm | 2.048 / 1024 |
| Aperture pixels | ~75 px diameter | 150mm / 2mm |
| Aperture fill ratio | ~7.3% | (75/1024)^2 * pi/4 |

### 2.4 Data / Runtime

| Parameter | Config Key | Value | Description |
|-----------|-----------|-------|-------------|
| Train realizations | `data.split_counts.train` | 160 | 학습 데이터 수 |
| Val realizations | `data.split_counts.val` | 20 | 검증 데이터 수 |
| Test realizations | `data.split_counts.test` | 20 | 테스트 데이터 수 |
| Seed | `runtime.seed` | 20260323 | Global RNG seed |
| Device | `runtime.device` | cuda | GPU |
| FFT warmup | `runtime.fft_warmup_iters` | 3 | cuFFT plan 초기화 |
| Deterministic | `runtime.deterministic_algorithms` | True | 재현성 보장 |
| TF32 | `runtime.allow_tf32` | False | 정밀도 우선 |
| Num workers | `runtime.num_workers` | 0 | 단일 프로세스 로딩 |

---

## 3. Tuning Priority

영향도 순서:

| Rank | Parameter | Rationale |
|:----:|-----------|-----------|
| 1 | `cn2` | 난류 강도 = 학습 난이도. 1e-15(weak)에서 검증 후 1e-14로 증가 |
| 2 | `num_layers` | Layer 많을수록 correction capacity up, GPU memory up |
| 3 | `layer_spacing_m` | Phase-amplitude coupling 결정. 짧으면 phase만 변조됨 |
| 4 | `complex_overlap / amplitude_mse` ratio | Phase vs amplitude 학습 균형 |
| 5 | `learning_rate` | 너무 크면 phase wrapping 불안정 |
| 6 | `aperture_diameter_m` | 15cm vs 60cm → correction mode 수 변화 |

### Cn^2 scaling roadmap

| Stage | Cn^2 | Regime | D/r0 (15cm) | Expected difficulty |
|:-----:|------|--------|:-----------:|---------------------|
| 1 | 1e-15 | Very weak | 0.21 | Easy (validation) |
| 2 | 5e-15 | Weak | 0.39 | Moderate |
| 3 | 1e-14 | Moderate | 0.55 | Challenging |
| 4 | 5e-14 | Strong | 1.03 | Hard (scintillation onset) |
