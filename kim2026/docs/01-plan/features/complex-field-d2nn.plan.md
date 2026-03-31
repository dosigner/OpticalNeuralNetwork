# Plan: Complex Field D2NN Beam Cleanup

## Executive Summary

| Aspect | Description |
|--------|-------------|
| **Feature** | complex-field-d2nn |
| **Created** | 2026-03-23 |
| **Status** | Plan |

### Value Delivered

| Perspective | Description |
|-------------|-------------|
| **Problem** | 현재 BeamCleanupD2NN은 intensity(|u|²)만 복원하며 위상(phase) 정보를 버림. FSO 통신에서 coherent detection이나 adaptive optics 피드백에는 complex field (amplitude + phase) 복원이 필수. |
| **Solution** | 기존 D2NN 모델은 이미 complex field를 출력하지만 loss 계산시 intensity로 변환. Complex field loss와 target pipeline을 추가하여 위상까지 학습하도록 확장. |
| **Function UX Effect** | `mode: "complex"` config 옵션 하나로 complex field 학습 활성화. 기존 intensity 모드와 완전 호환. |
| **Core Value** | Phase-only D2NN이 난류 왜곡된 위상을 복원할 수 있으면 coherent FSO 수신기 성능이 비약적으로 향상. 15cm 원형 수신부에서 실질적 wavefront correction 가능. |

---

## 1. Background & Motivation

### 1.1 현재 아키텍처 분석

kim2026 프로젝트의 `BeamCleanupD2NN`은 다음과 같은 구조:

```
Turbulent field (complex) → [Circular Aperture 15cm] → [Phase Layer 1] → propagate →
[Phase Layer 2] → ... → [Phase Layer N] → propagate to detector → complex output
```

**핵심 발견**: 모델 자체(`d2nn.py:65`)는 이미 complex field를 반환:
```python
return output.to(torch.complex64)
```

하지만 training pipeline(`trainer.py:97`)에서 intensity로 변환:
```python
pred_intensity = pred_field.abs().square()
```

그리고 target 생성(`targets.py:44`)도 intensity만:
```python
return propagated.abs().square()
```

### 1.2 왜 Complex Field 복원이 필요한가

| Use Case | Intensity 충분? | Complex Field 필요? |
|----------|:-:|:-:|
| Incoherent imaging (카메라) | Yes | No |
| **Coherent detection (FSO comm)** | No | **Yes** |
| **Adaptive optics feedback** | No | **Yes** |
| Wavefront sensing | No | **Yes** |
| Interferometry | No | **Yes** |

FSO 통신에서 coherent receiver는 local oscillator와 수신 field를 간섭시키므로, amplitude + phase 모두 복원되어야 BER이 개선됨.

### 1.3 물리적 타당성

Phase-only D2NN이 complex field를 복원할 수 있는 이유:
- Phase layer는 `exp(jφ)` 곱셈 → amplitude를 보존하면서 phase만 조절
- Multi-layer + 전파(diffraction)를 통해 amplitude redistribution도 가능 (Gerchberg-Saxton 원리)
- 난류 왜곡은 주로 phase distortion → phase-only correction이 자연스러움

**한계**: Phase-only layer만으로는 amplitude 복원에 한계. Scintillation이 강한 경우 (σ²χ > 0.25) complex field 완벽 복원 어려움 → weak-to-moderate turbulence에서 효과적.

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|------------|----------|
| FR-01 | Complex field loss function 구현 (amplitude MSE + phase MSE 가중합) | P0 |
| FR-02 | Complex field target pipeline (intensity 변환 없이 complex field 유지) | P0 |
| FR-03 | Config에 `training.loss.mode: "complex"` 옵션 추가 | P0 |
| FR-04 | Complex field evaluation metrics (complex overlap, phase RMSE, Strehl) | P0 |
| FR-05 | 기존 intensity 모드와 backward compatible | P1 |
| FR-06 | Global phase ambiguity 처리 (D2NN은 global phase offset을 자유도로 가짐) | P0 |
| FR-07 | 15cm circular receiver aperture에서 동작 검증 | P1 |

### 2.2 Non-Functional Requirements

| ID | Requirement |
|----|------------|
| NF-01 | GPU 메모리 증가 최소화 (complex target은 intensity target 대비 2× 메모리) |
| NF-02 | 기존 테스트 모두 통과 |
| NF-03 | Training 속도 intensity 모드 대비 ≤1.2× 증가 |

---

## 3. Technical Approach

### 3.1 Complex Field Loss Function

**Global phase ambiguity 문제**: D2NN 출력 `u_pred`와 target `u_target` 사이에 global phase `exp(jα)` 차이가 존재할 수 있음. 이는 물리적으로 의미 없으므로 제거 필요.

```python
# Phase-aligned complex MSE
alpha = angle(sum(u_pred * conj(u_target)))  # optimal global phase
u_aligned = u_pred * exp(-j*alpha)
loss = |u_aligned - u_target|^2 / |u_target|^2
```

**Loss 구성**:
```
L_complex = w_amp * L_amplitude + w_phase * L_phase + w_overlap * L_complex_overlap

L_amplitude = MSE(|u_pred|, |u_target|)            # amplitude fidelity
L_phase = 1 - |<u_pred, u_target>| / (||u_pred|| * ||u_target||)  # phase-sensitive overlap
L_complex_overlap = 위와 동일 (complex inner product 기반)
```

### 3.2 Target Pipeline 변경

```python
# 현재 (intensity):
def make_detector_plane_target(...) -> torch.Tensor:
    ...
    return propagated.abs().square()  # float tensor

# 변경 (complex):
def make_detector_plane_target(..., complex_mode=False) -> torch.Tensor:
    ...
    if complex_mode:
        return propagated  # complex tensor (amplitude + phase 유지)
    return propagated.abs().square()
```

### 3.3 Config 확장

```yaml
training:
  loss:
    mode: "complex"           # "intensity" (default, backward compat) | "complex"
    weights:
      # intensity mode
      overlap: 1.0
      radius: 0.25
      encircled: 0.25
      # complex mode
      complex_overlap: 1.0
      amplitude_mse: 0.5
      phase_mse: 0.3
```

### 3.4 Evaluation Metrics 추가

| Metric | Definition | 의미 |
|--------|-----------|------|
| `complex_overlap` | `|⟨u_pred, u_target⟩| / (‖u_pred‖ · ‖u_target‖)` | Phase-sensitive fidelity (0~1) |
| `phase_rmse_rad` | `RMSE(angle(u_pred) - angle(u_target))` (global phase 보정 후) | Wavefront error [rad] |
| `amplitude_rmse` | `RMSE(|u_pred| - |u_target|)` | Amplitude error |
| `strehl_complex` | Complex overlap의 제곱 (Maréchal approx.) | Coherent Strehl ratio |

---

## 4. Implementation Order

### Step 1: Complex field loss functions (`training/losses.py`)
- `complex_overlap_loss()`: Phase-sensitive overlap loss with global phase removal
- `amplitude_mse_loss()`: Amplitude-only MSE
- `complex_field_loss()`: Composite weighted loss
- **파일**: `src/kim2026/training/losses.py` (기존 파일에 추가)

### Step 2: Target pipeline 수정 (`training/targets.py`)
- `make_detector_plane_target()`에 `complex_mode` 파라미터 추가
- Complex mode일 때 `propagated` 그대로 반환
- **파일**: `src/kim2026/training/targets.py`

### Step 3: Trainer 수정 (`training/trainer.py`)
- `_epoch_pass()`에서 loss mode 분기
- Complex mode: `pred_field` 직접 사용 (intensity 변환 안 함)
- Intensity mode: 기존과 동일
- **파일**: `src/kim2026/training/trainer.py`

### Step 4: Complex field metrics (`training/metrics.py`)
- `complex_overlap()`, `phase_rmse()`, `amplitude_rmse()`
- `summarize_metrics()`에 complex metrics 통합
- **파일**: `src/kim2026/training/metrics.py`

### Step 5: Config schema 확장 (`config/schema.py`)
- `training.loss.mode` 필드 추가 (default: "intensity")
- Complex mode weights validation
- **파일**: `src/kim2026/config/schema.py`

### Step 6: Training config 작성
- `configs/fso_1024_complex.yaml` (complex mode 학습용)
- 15cm 원형 수신부 설정 유지
- **파일**: `configs/fso_1024_complex.yaml` (신규)

### Step 7: Tests
- Complex loss function unit tests
- Complex target pipeline tests
- End-to-end smoke test (complex mode)
- **파일**: `tests/test_complex_losses.py` (신규)

---

## 5. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Global phase ambiguity로 학습 불안정 | High | Phase alignment 후 loss 계산, gradient clipping |
| Phase wrapping (-π ~ π) discontinuity | Medium | Complex domain에서 loss 계산 (angle 직접 비교 안 함) |
| Strong scintillation에서 amplitude 복원 한계 | Medium | Weak turbulence (σ²χ < 0.25)에서 먼저 검증 |
| 기존 intensity pipeline 깨짐 | Low | `mode: "intensity"` default로 backward compat 보장 |

---

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| Complex overlap (weak turbulence) | ≥ 0.85 |
| Phase RMSE | ≤ π/4 rad (~0.79 rad) |
| 기존 intensity test 통과 | 100% |
| Training 속도 증가 | ≤ 1.2× |

---

## 7. Scope Exclusions

- Amplitude modulation layer 추가 (phase-only 유지)
- Strong turbulence (σ²χ > 0.5) 지원
- Multi-wavelength 지원
- Temporal/frozen-flow 학습 (단일 snapshot 기반)
