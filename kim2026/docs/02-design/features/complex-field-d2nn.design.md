# Design: Complex Field D2NN Beam Cleanup

## Executive Summary

| Aspect | Description |
|--------|-------------|
| **Feature** | complex-field-d2nn |
| **Created** | 2026-03-23 |
| **Phase** | Design |
| **Plan Reference** | `docs/01-plan/features/complex-field-d2nn.plan.md` |

### Value Delivered

| Perspective | Description |
|-------------|-------------|
| **Problem** | `BeamCleanupD2NN`은 complex field를 출력하지만, trainer가 intensity(`abs().square()`)로 변환하여 위상 정보를 버림 |
| **Solution** | Loss, target, trainer, metrics, config에 `mode: "complex"` 분기를 추가하여 complex field 직접 학습 |
| **Function UX Effect** | `training.loss.mode: "complex"` 한 줄로 활성화, 기존 intensity 모드 100% 호환 |
| **Core Value** | Coherent FSO 수신기에서 wavefront correction → BER 개선, adaptive optics 피드백 가능 |

---

## 1. Architecture Overview

### 1.1 현재 데이터 흐름 (intensity mode)

```
u_turb ──→ [Circular Aperture] ──→ [BeamCleanupD2NN] ──→ pred_field (complex)
                                                              │
                                                         .abs().square()
                                                              │
                                                         pred_intensity ──→ beam_cleanup_loss()

u_vacuum ──→ [Circular Aperture] ──→ propagate ──→ .abs().square() ──→ target_intensity
```

### 1.2 변경된 데이터 흐름 (complex mode, static turbulence)

```
u_turb ──→ [Circular Aperture] ──→ [BeamCleanupD2NN] ──→ pred_field (complex)
                                                              │
                                                         complex_field_loss()
                                                              │
u_vacuum ──→ [Circular Aperture] ──→ propagate ──→ target_field (complex)
```

### 1.3 Turbulence Model: Static Phase Screens (No Frozen Flow)

기존 kim2026은 frozen-flow temporal model (Taylor hypothesis)을 사용하지만,
complex field D2NN 학습에서는 **static snapshot** 방식만 사용한다.

**제거 항목**:
- `frozen_flow` config 섹션 전체 (wind_speed, dt_s, frames_per_episode, screen_canvas_scale)
- `extract_frozen_flow_window()` 호출 (canvas scrolling)
- Episode/frame 2중 루프

**대체 방식**:
- 각 realization마다 독립적인 static phase screen 세트 생성
- `channel.num_realizations`로 총 생성 수 지정
- Phase screen은 grid size N에서 직접 생성 (canvas 확장 불필요)

**근거**:
- Complex field 복원 학습에서 시간 상관은 불필요 (단일 snapshot 복원이 목표)
- Frozen flow는 temporal AO 제어 시뮬레이션용이지 D2NN 학습에는 과도
- Static screen이 구현/디버깅 단순화, 데이터 생성 속도 향상

```
# Static turbulence pair generation (simplified)
for realization_id in range(num_realizations):
    screens = [generate_phase_screen(seed=...) for cell in cells]
    u_turb = propagate_split_step(source, phase_screens=screens)
    save_pair(u_vacuum, u_turb)
```

### 1.4 변경 대상 파일 목록

| File | Change Type | Description |
|------|:-----------:|-------------|
| `src/kim2026/training/losses.py` | **ADD** | Complex field loss functions 3개 추가 |
| `src/kim2026/training/targets.py` | **MODIFY** | `make_detector_plane_target()`에 `complex_mode` param |
| `src/kim2026/training/trainer.py` | **MODIFY** | `_epoch_pass()`에 loss mode 분기 |
| `src/kim2026/training/metrics.py` | **ADD** | Complex field metrics 3개 추가 |
| `src/kim2026/config/schema.py` | **MODIFY** | `training.loss.mode` validation + frozen_flow optional |
| `src/kim2026/turbulence/channel.py` | **MODIFY** | Static realization 모드 추가 (frozen_flow 불필요) |
| `src/kim2026/cli/generate_pairs.py` | **MODIFY** | Static pair generation 지원 |
| `src/kim2026/cli/evaluate_beam_cleanup.py` | **MODIFY** | Complex metrics 출력 지원 |
| `configs/fso_1024_complex.yaml` | **NEW** | Complex mode training config (static turbulence) |
| `tests/test_complex_losses.py` | **NEW** | Complex loss unit tests |

---

## 2. Detailed Specifications

### 2.1 Complex Field Loss Functions

**File**: `src/kim2026/training/losses.py` (기존 파일에 함수 추가)

#### 2.1.1 `align_global_phase(pred, target) -> Tensor`

Global phase ambiguity 제거. D2NN 출력이 target과 상수 위상 `exp(jα)` 차이가 나는 것을 보정.

```python
def align_global_phase(
    pred: torch.Tensor,       # (B, N, N) complex
    target: torch.Tensor,     # (B, N, N) complex
) -> torch.Tensor:
    """Remove global phase offset between pred and target.

    Returns pred multiplied by exp(-j*alpha) where alpha minimizes
    ||pred*exp(-j*alpha) - target||^2.

    Optimal alpha = angle(sum(pred * conj(target))).
    """
    # (B,) complex inner product
    inner = (pred.reshape(pred.shape[0], -1) * target.reshape(target.shape[0], -1).conj()).sum(dim=1)
    alpha = torch.angle(inner)  # (B,)
    correction = torch.exp(-1j * alpha).to(pred.dtype)  # (B,)
    return pred * correction.reshape(-1, 1, 1)
```

**근거**: `alpha = arg(⟨u_pred, u_target⟩)`는 `||u_pred·e^{-jα} - u_target||²`를 최소화하는 closed-form solution.

#### 2.1.2 `complex_overlap_loss(pred, target) -> Tensor`

Phase-sensitive normalized overlap. Global phase에 불변.

```python
def complex_overlap_loss(
    pred: torch.Tensor,       # (B, N, N) complex
    target: torch.Tensor,     # (B, N, N) complex
) -> torch.Tensor:
    """1 - |<pred, target>| / (||pred|| * ||target||).

    Measures how well pred matches target in both amplitude and phase,
    invariant to global phase offset. Range: [0, 1], lower is better.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inner = (pred_flat * target_flat.conj()).sum(dim=1)  # complex (B,)
    norm_pred = torch.linalg.vector_norm(pred_flat, dim=1)   # real (B,)
    norm_target = torch.linalg.vector_norm(target_flat, dim=1)
    overlap = inner.abs() / (norm_pred * norm_target).clamp_min(1e-12)
    return (1.0 - overlap).mean()
```

**Note**: `torch.linalg.vector_norm`은 complex tensor에서 자동으로 `sqrt(sum(|x_i|²))`를 계산.

#### 2.1.3 `amplitude_mse_loss(pred, target) -> Tensor`

Amplitude-only MSE (phase 무시).

```python
def amplitude_mse_loss(
    pred: torch.Tensor,       # (B, N, N) complex
    target: torch.Tensor,     # (B, N, N) complex
) -> torch.Tensor:
    """MSE between amplitudes: mean(||pred| - |target||^2)."""
    return torch.mean((pred.abs() - target.abs()).square())
```

#### 2.1.4 `complex_field_loss(pred, target, weights) -> Tensor`

Composite loss for complex field training.

```python
def complex_field_loss(
    pred: torch.Tensor,       # (B, N, N) complex
    target: torch.Tensor,     # (B, N, N) complex
    *,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Composite complex field loss.

    Default weights: complex_overlap=1.0, amplitude_mse=0.5
    """
    if weights is None:
        weights = {"complex_overlap": 1.0, "amplitude_mse": 0.5}
    loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
    w_overlap = float(weights.get("complex_overlap", 1.0))
    w_amp = float(weights.get("amplitude_mse", 0.5))
    if w_overlap > 0:
        loss = loss + w_overlap * complex_overlap_loss(pred, target)
    if w_amp > 0:
        loss = loss + w_amp * amplitude_mse_loss(pred, target)
    return loss
```

**설계 결정 - Phase MSE를 별도 항으로 안 넣는 이유**:
- `angle()` 함수는 `-π`와 `+π` 근처에서 discontinuity → gradient 불안정
- `complex_overlap_loss`가 이미 phase fidelity를 포함 (complex inner product)
- Amplitude MSE를 추가하여 energy 분포까지 학습

---

### 2.2 Target Pipeline 수정

**File**: `src/kim2026/training/targets.py`

```python
def make_detector_plane_target(
    vacuum_field: torch.Tensor,
    *,
    wavelength_m: float,
    receiver_window_m: float,
    aperture_diameter_m: float,
    total_distance_m: float,
    complex_mode: bool = False,        # NEW parameter
) -> torch.Tensor:
    """Propagate the aperture-limited vacuum field to the detector plane.

    If complex_mode=True, returns the complex field directly.
    If complex_mode=False (default), returns intensity |field|^2.
    """
    apertured = apply_receiver_aperture(
        vacuum_field,
        receiver_window_m=receiver_window_m,
        aperture_diameter_m=aperture_diameter_m,
    )
    propagated = propagate_same_window(
        apertured,
        wavelength_m=wavelength_m,
        window_m=receiver_window_m,
        z_m=total_distance_m,
    )
    if complex_mode:
        return propagated
    return propagated.abs().square()
```

**Backward compatibility**: `complex_mode=False` default → 기존 코드 변경 없음.

---

### 2.3 Trainer 수정

**File**: `src/kim2026/training/trainer.py`

`_epoch_pass()` 함수의 loss 계산 부분만 수정:

```python
# 현재 (lines 81-98):
target = make_detector_plane_target(u_vacuum, ...)
input_field = apply_receiver_aperture(u_turb, ...)
pred_field = model(input_field)
pred_intensity = pred_field.abs().square()
loss = beam_cleanup_loss(pred_intensity, target, ...)

# 변경:
loss_mode = str(cfg["training"]["loss"].get("mode", "intensity"))
complex_mode = loss_mode == "complex"

target = make_detector_plane_target(u_vacuum, ..., complex_mode=complex_mode)
input_field = apply_receiver_aperture(u_turb, ...)
pred_field = model(input_field)

if complex_mode:
    complex_weights = cfg["training"]["loss"].get("complex_weights", None)
    loss = complex_field_loss(pred_field, target, weights=complex_weights)
else:
    pred_intensity = pred_field.abs().square()
    loss = beam_cleanup_loss(pred_intensity, target, window_m=receiver_window_m, weights=weights)
```

**변경 범위**: `_epoch_pass()` 함수 내부만. 외부 인터페이스 변경 없음.

---

### 2.4 Complex Field Metrics

**File**: `src/kim2026/training/metrics.py` (기존 파일에 함수 추가)

#### 2.4.1 `complex_overlap(pred, target) -> Tensor`

```python
def complex_overlap(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Phase-sensitive overlap |<pred, target>| / (||pred|| ||target||). Higher is better."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inner = (pred_flat * target_flat.conj()).sum(dim=1)
    norm_pred = torch.linalg.vector_norm(pred_flat, dim=1)
    norm_target = torch.linalg.vector_norm(target_flat, dim=1)
    return inner.abs() / (norm_pred * norm_target).clamp_min(1e-12)
```

#### 2.4.2 `phase_rmse(pred, target) -> Tensor`

```python
def phase_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Phase RMSE in radians after global phase alignment. Lower is better."""
    from kim2026.training.losses import align_global_phase
    aligned = align_global_phase(pred, target)
    # Only compute where both fields have significant amplitude
    amp_threshold = 0.1 * target.abs().amax(dim=(-2, -1), keepdim=True)
    mask = (target.abs() > amp_threshold) & (aligned.abs() > amp_threshold)
    phase_diff = torch.angle(aligned) - torch.angle(target)
    # Wrap to [-pi, pi]
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    # Masked RMS per batch
    masked_sq = (phase_diff.square() * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1)
    return masked_sq.sqrt()
```

**설계 결정**: amplitude가 작은 영역의 phase는 noise dominated → `amp_threshold`로 마스킹.

#### 2.4.3 `amplitude_rmse(pred, target) -> Tensor`

```python
def amplitude_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE between amplitudes, normalized by target peak. Lower is better."""
    diff = pred.abs() - target.abs()
    peak = target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return (diff.square().mean(dim=(-2, -1)).sqrt() / peak.squeeze(-1).squeeze(-1))
```

#### 2.4.4 `summarize_metrics()` 확장

```python
def summarize_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_m: float | None = None,
    complex_mode: bool = False,         # NEW
) -> dict[str, float]:
    if complex_mode:
        return {
            "complex_overlap": float(complex_overlap(pred, target).mean().item()),
            "phase_rmse_rad": float(phase_rmse(pred, target).mean().item()),
            "amplitude_rmse": float(amplitude_rmse(pred, target).mean().item()),
            # Also report intensity metrics for comparison
            "intensity_overlap": float(gaussian_overlap(
                pred.abs().square(), target.abs().square()
            ).mean().item()),
            "strehl": float(strehl_ratio(
                pred.abs().square(), target.abs().square()
            ).mean().item()),
        }
    # existing intensity-only path unchanged
    ...
```

---

### 2.5 Static Pair Generation (`turbulence/channel.py`)

**File**: `src/kim2026/turbulence/channel.py`

기존 `generate_pair_cache()`에 static mode 분기 추가. Frozen flow 로직을 건너뛰고 직접 phase screen 생성.

```python
def generate_static_pair_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate vacuum/turbulence pairs with static (independent) phase screens.

    Each realization gets an independent set of phase screens — no temporal
    correlation, no frozen-flow canvas, no wind scrolling.
    """
    # ... setup same as generate_pair_cache ...

    num_realizations = int(cfg["channel"]["num_realizations"])
    split_counts = dict(cfg["data"]["split_counts"])

    # Single vacuum propagation (shared across all realizations)
    vacuum_field = propagate_to_receiver(source_field, schedule, phase_screens=None)

    for real_id in range(num_realizations):
        real_seed = global_seed + 1009 * real_id
        # Generate independent phase screens directly at grid size N
        screens = []
        for cell in schedule.screen_cells:
            screens.append(generate_phase_screen(
                n=n,
                window_m=source_window_m,
                wavelength_m=wavelength_m,
                cn2=cn2,
                path_segment_m=cell.length_m,
                outer_scale_m=outer_scale_m,
                inner_scale_m=inner_scale_m,
                seed=real_seed * 100 + cell.screen_index,
            ))
        # Propagate with turbulence
        turb_field = propagate_to_receiver(source_field, schedule, phase_screens=screens)
        save_pair(vacuum_field, turb_field, ...)
```

**핵심 차이**:
- Canvas 생성 없음 (`screen_canvas_scale` 불필요)
- `extract_frozen_flow_window()` 호출 없음
- Episode/frame 구조 없음 → flat realization list
- Phase screen을 grid size N에서 직접 생성

### 2.6 Config Schema 수정

**File**: `src/kim2026/config/schema.py`

#### 2.6.1 Channel mode validation

```python
# channel 섹션에 mode 추가:
channel_mode = str(channel.get("mode", "frozen_flow"))
if channel_mode not in ("frozen_flow", "static"):
    raise ValueError(f"channel.mode must be 'frozen_flow' or 'static', got '{channel_mode}'")
channel["mode"] = channel_mode

if channel_mode == "static":
    # frozen_flow 섹션 불필요, num_realizations 필수
    num_realizations = int(channel.get("num_realizations", 200))
    if num_realizations <= 0:
        raise ValueError("channel.num_realizations must be > 0")
    channel["num_realizations"] = num_realizations
else:
    # 기존 frozen_flow validation (lines 78-85 유지)
    ...
```

#### 2.6.2 Loss mode validation

```python
# training.loss 섹션에 mode 추가:
loss_mode = str(loss.get("mode", "intensity"))
if loss_mode not in ("intensity", "complex"):
    raise ValueError(f"training.loss.mode must be 'intensity' or 'complex', got '{loss_mode}'")
loss["mode"] = loss_mode

if loss_mode == "complex":
    complex_weights = loss.get("complex_weights", {"complex_overlap": 1.0, "amplitude_mse": 0.5})
    _non_negative(complex_weights.get("complex_overlap", 1.0), "training.loss.complex_weights.complex_overlap")
    _non_negative(complex_weights.get("amplitude_mse", 0.5), "training.loss.complex_weights.amplitude_mse")
    loss["complex_weights"] = complex_weights

if loss_mode == "complex":
    evaluation.setdefault("metrics", [
        "complex_overlap", "phase_rmse_rad", "amplitude_rmse",
        "intensity_overlap", "strehl",
    ])
```

---

### 2.7 Evaluate CLI 수정

**File**: `src/kim2026/cli/evaluate_beam_cleanup.py`

변경 사항:
- `make_detector_plane_target()`에 `complex_mode` 전달
- `summarize_metrics()`에 `complex_mode` 전달
- Baseline 계산시 complex mode면 complex field 유지

```python
# Line 113-131 변경:
loss_mode = str(cfg["training"]["loss"].get("mode", "intensity"))
complex_mode = loss_mode == "complex"

target = make_detector_plane_target(
    u_vacuum, ..., complex_mode=complex_mode,
)
...
if complex_mode:
    baseline_field = propagate_same_window(apertured, ...)
    pred_field = model(apertured)
    baseline_summaries.append(summarize_metrics(baseline_field, target, complex_mode=True))
    model_summaries.append(summarize_metrics(pred_field, target, complex_mode=True))
else:
    baseline = propagate_same_window(apertured, ...).abs().square()
    pred = model(apertured).abs().square()
    baseline_summaries.append(summarize_metrics(baseline, target, window_m=receiver_window_m))
    model_summaries.append(summarize_metrics(pred, target, window_m=receiver_window_m))
```

---

### 2.8 Training Config

**File**: `configs/fso_1024_complex.yaml` (NEW)

```yaml
experiment:
  id: fso_1024_complex_field
  save_dir: runs/fso_1024_complex

optics:
  lambda_m: 1.55e-6
  half_angle_rad: 3.0e-4
  m2: 1.0

grid:
  n: 1024
  source_window_m: 0.006578
  receiver_window_m: 2.048

channel:
  path_length_m: 1000.0
  cn2: 1.0e-15              # Weak turbulence for initial validation
  outer_scale_m: 30.0
  inner_scale_m: 5.0e-3
  num_screens: 6
  mode: "static"            # Static phase screens, no frozen flow
  num_realizations: 200     # Total independent turbulence snapshots
  # frozen_flow 섹션 없음 — static mode에서는 불필요

receiver:
  aperture_diameter_m: 0.15    # 15cm circular receiver

model:
  num_layers: 5
  layer_spacing_m: 50.0
  detector_distance_m: 50.0

training:
  epochs: 40
  batch_size: 2
  pair_generation_batch_size: 2
  eval_batch_size: 2
  learning_rate: 5.0e-4
  loss:
    mode: "complex"              # Complex field training
    complex_weights:
      complex_overlap: 1.0
      amplitude_mse: 0.5
    # Intensity weights still required by schema for backward compat
    weights:
      overlap: 1.0
      radius: 0.25
      encircled: 0.25

data:
  cache_dir: data/kim2026/fso_1024_complex/cache
  split_manifest_path: data/kim2026/fso_1024_complex/split_manifest.json
  split_counts:
    train: 160
    val: 20
    test: 20

evaluation:
  split: test
  metrics: [complex_overlap, phase_rmse_rad, amplitude_rmse, intensity_overlap, strehl]
  save_json: true

visualization:
  save_raw: true
  save_plots: true
  output_dir: runs/fso_1024_complex/figures

runtime:
  seed: 20260323
  strict_reproducibility: true
  allow_tf32: false
  deterministic_algorithms: true
  cublas_workspace_config: ":4096:8"
  device: cuda
  num_workers: 0
  pin_memory: true
  persistent_workers: false
  prefetch_factor: 2
  fft_warmup_iters: 3
```

**Note**: `cn2: 1.0e-15` (weak turbulence)로 시작. Complex field 복원은 phase distortion이 작을 때 먼저 검증 후 점진적으로 증가.

**Static vs Frozen Flow 차이**:

| | Static (이 config) | Frozen Flow (기존) |
|---|---|---|
| Phase screen | 각 realization 독립 생성 | 큰 canvas에서 wind scroll |
| 시간 상관 | 없음 | Taylor hypothesis |
| 데이터 구조 | realization 단일 루프 | episode → frame 2중 루프 |
| 용도 | D2NN snapshot 학습 | Temporal AO 시뮬레이션 |
| Config 복잡도 | 낮음 | 높음 (wind, dt, canvas) |

---

## 3. Implementation Order

| Step | File | Description | Estimated Lines |
|:----:|------|-------------|:---------------:|
| 1 | `training/losses.py` | `align_global_phase`, `complex_overlap_loss`, `amplitude_mse_loss`, `complex_field_loss` | +55 |
| 2 | `training/targets.py` | `complex_mode` param to `make_detector_plane_target` | +3 (modify) |
| 3 | `training/trainer.py` | Loss mode branching in `_epoch_pass` | +12 (modify) |
| 4 | `training/metrics.py` | `complex_overlap`, `phase_rmse`, `amplitude_rmse`, extended `summarize_metrics` | +50 |
| 5 | `config/schema.py` | `channel.mode` + `training.loss.mode` validation | +25 (modify) |
| 6 | `turbulence/channel.py` | `generate_static_pair_cache()` static mode pair generation | +60 (add) |
| 7 | `cli/generate_pairs.py` | Static mode dispatch | +5 (modify) |
| 8 | `cli/evaluate_beam_cleanup.py` | Complex metrics 출력 지원 | +15 (modify) |
| 9 | `configs/fso_1024_complex.yaml` | Static turbulence + complex loss config | +55 (new) |
| 10 | `tests/test_complex_losses.py` | Unit tests for all new functions | +120 (new) |

**Total**: ~400 lines added/modified across 10 files.

---

## 4. Test Strategy

### 4.1 Unit Tests (`tests/test_complex_losses.py`)

| Test | Validates |
|------|-----------|
| `test_align_global_phase_identity` | `align(u, u)` returns `u` |
| `test_align_global_phase_rotation` | `align(u*exp(jα), u)` removes α |
| `test_complex_overlap_identical` | Identical fields → loss = 0 |
| `test_complex_overlap_orthogonal` | Orthogonal fields → loss = 1 |
| `test_complex_overlap_phase_invariant` | `loss(u*exp(jα), v) == loss(u, v)` |
| `test_amplitude_mse_zero` | Identical amplitudes → loss = 0 |
| `test_complex_field_loss_composite` | Weighted sum correctness |
| `test_phase_rmse_zero_for_identical` | Same field → RMSE = 0 |
| `test_phase_rmse_known_offset` | Known phase → expected RMSE |
| `test_make_target_complex_mode` | Returns complex tensor |
| `test_make_target_intensity_mode` | Returns real tensor (backward compat) |
| `test_config_complex_mode_valid` | Schema accepts `mode: "complex"` |
| `test_config_intensity_mode_default` | Schema defaults to `mode: "intensity"` |

### 4.2 Integration: Smoke Test

기존 `configs/smoke_test.yaml`에 `training.loss.mode: "complex"` 추가한 별도 smoke config로 end-to-end 검증.

### 4.3 Regression

기존 intensity mode 테스트가 모두 통과해야 함 (default `mode: "intensity"`).

---

## 5. Gradient Flow Analysis

Complex field loss의 gradient가 phase parameter까지 올바르게 전파되는지 확인:

```
L = complex_overlap_loss(pred_field, target_field)
     ↓ grad
pred_field = model.forward(input_field)  # complex
     ↓ grad (through propagate_same_window → FFT → complex multiply)
phase_layer.phase  # real parameter
     ↓ update via exp(1j * phase)
```

PyTorch의 `torch.fft.fft2`, `torch.exp(1j * ...)`, complex multiplication 모두 autograd 지원.
`complex_overlap_loss`의 `inner.abs()`도 differentiable (except at 0, which doesn't occur in practice).

---

## 6. Memory Impact

| Component | Intensity Mode | Complex Mode | Delta |
|-----------|:-:|:-:|:-:|
| Target tensor | (B, N, N) float32 | (B, N, N) complex64 | **+2×** |
| pred_field | 이미 complex | 이미 complex | 0 |
| Loss computation | real ops | complex ops | ~+30% |
| **Total GPU memory** | baseline | **+~15%** | within NF-01 (≤20%) |

Complex64 = 2× float32, 하지만 target만 영향받고 model/pred는 이미 complex.
