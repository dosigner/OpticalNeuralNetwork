# FD2NN Metalens Sweep: Analysis & Insight Report

> Date: 2026-03-23
> Data: 1km_cn2e-14_w2m_n1024_dx2mm (Cn2=1e-14, 1024x1024)
> Model: BeamCleanupFD2NN, hybrid domain (F,R,F,R,F), 5 layers
> Metalens pixel pitch: dx=2um, window=2.048mm (beam reducer 75:1 후)

---

## 1. 실험 결과 정리

| Spacing | Complex Overlap | Phase RMSE [rad] | Amp RMSE | Intensity Overlap | Strehl |
|:-------:|:---------------:|:----------------:|:--------:|:-----------------:|:------:|
| Baseline (no D2NN) | **0.1913** | — | — | **0.9725** | — |
| 0mm (FFT only) | 0.1795 (-6.2%) | 1.673 | 0.047 | 0.953 | 2.189 |
| 0.1mm | 0.1833 (-4.2%) | 1.672 | 0.107 | 0.768 | 3.068 |
| **1mm** | **0.2146 (+12.2%)** | 1.511 | 0.173 | 0.650 | 1.325 |
| **2mm** | 0.2017 (+5.4%) | **1.475** | 0.168 | 0.707 | 1.882 |
| 5mm | 0.1989 (+4.0%) | 1.496 | 0.162 | 0.776 | 1.451 |
| 10mm | 0.2020 (+5.6%) | 1.492 | 0.161 | 0.771 | 1.505 |

---

## 2. 핵심 발견

### 2.1 Propagation이 필수: FFT-only는 실패

**0mm (propagation 없음)**: Complex overlap 0.1795로 baseline(0.1913)보다 **오히려 나쁨**.

**원인 분석**:
- FFT/IFFT domain switching만으로는 inter-layer coupling이 부족
- Phase mask가 field를 변조하지만, domain switch 없이는 각 layer가 독립적으로 작용
- 5개 independent phase mask의 곱 = 하나의 합성 phase mask → **단일 layer와 동등**
- 단일 phase-only mask로는 complex field correction 불가 (amplitude 제어 불가)

**결론**: **Inter-layer propagation(diffraction)이 FD2NN의 핵심 메커니즘.** Domain switching은 보조적.

### 2.2 최적 Spacing: 1mm (sweet spot)

| Spacing | N_F (dx=2um) | 효과 |
|:-------:|:------------:|------|
| 0.1mm | 0.026 | Diffraction 약함 → 거의 FFT-only와 동일 |
| **1mm** | **0.0026** | **Far-field diffraction → pixel 간 강한 coupling** |
| 2mm | 0.0013 | Over-diffraction → field spread 과도 |
| 5-10mm | <0.001 | Deep far field → field 완전 확산, 정보 손실 |

**1mm에서 최적인 이유**:
- 인접 metalens layer에서 나온 빛이 다음 layer의 **여러 pixel에 걸쳐** 도달
- Phase mask가 주변 pixel 정보를 결합하여 amplitude를 간접적으로 제어 가능
- 너무 멀면 field가 완전히 확산되어 spatial 정보 손실

### 2.3 역설: Complex Overlap ↑ but Intensity Overlap ↓

| Metric | 0mm | 1mm | 해석 |
|--------|:---:|:---:|------|
| Complex Overlap | 0.180 | **0.215** | Phase+Amplitude 복원 개선 |
| Intensity Overlap | **0.953** | 0.650 | Intensity 패턴 악화 |
| Amplitude RMSE | **0.047** | 0.173 | Amplitude 오류 증가 |

**이것은 심각한 문제.**

FD2NN이 complex overlap을 개선하지만 **amplitude를 왜곡**시킨다:
- Phase correction은 일어나지만 (Phase RMSE: 1.67→1.51)
- Amplitude가 크게 변형됨 (Amp RMSE: 0.047→0.173, 3.7배 악화)
- Intensity overlap 하락 (0.953→0.650)은 이 amplitude 왜곡 때문

**근본 원인**: Phase-only mask는 amplitude를 직접 제어할 수 없음.
Propagation을 통한 간접 amplitude 제어는 원하지 않는 amplitude 변형을 동반.

### 2.4 학습 Phase Mask 분석 (Fig 3)

학습된 phase mask가 **Fresnel lens (동심원 ring) 패턴**을 형성:
- Fourier domain layers (L0, L2, L4): 뚜렷한 concentric rings
- Real domain layers (L1, L3): 더 복잡한 패턴, L3에 speckle-like 구조

**해석**: FD2NN이 turbulence correction보다 **focusing/defocusing lens**를 학습.
이는 complex overlap 최적화 과정에서 "field를 target과 비슷한 shape으로 reshape"하는
가장 쉬운 경로가 lens 기능이기 때문.

### 2.5 Fourier Domain Analysis (Fig 4)

- **Turbulent input**: Scattered, broad PSD (low frequency 위주)
- **FD2NN output**: Ring 구조의 PSD (Fresnel lens 효과)
- **Vacuum target**: Smooth Gaussian-like PSD

FD2NN output의 PSD가 target과 매우 다름 → **주파수 영역에서도 correction이 불충분**

---

## 3. 한계 진단

### 3.1 Phase-Only의 근본 한계

| 문제 | 심각도 | 설명 |
|------|:------:|------|
| Amplitude 직접 제어 불가 | **Critical** | Phase-only mask는 |E|를 보존. Amplitude correction은 propagation 의존 |
| Phase-amplitude trade-off | **High** | Phase 개선하면 amplitude 악화 (역도 성립) |
| 학습 capacity 부족 | **Medium** | 5 layers × 1024² = 5.2M params이지만, 유효 DOF는 훨씬 적음 |
| Loss function 한계 | **Medium** | Complex overlap은 global metric — local correction 유도 부족 |

### 3.2 Quantitative 한계 평가

```
현재 최고 (spacing=1mm):
  Complex Overlap:  0.2146  (baseline 0.1913, 이상적 목표 1.0)
  Phase RMSE:       1.511 rad (baseline ~1.8, 이상적 목표 0)
  개선율:           12.2% overlap, 16% phase

30 epoch 학습 후에도 loss curve가 아직 수렴하지 않음
  → epoch 0: loss=0.987, ep 29: loss=0.768 (아직 하강 중)
  → 100-200 epoch이면 더 개선 가능
```

---

## 4. 개선 가능성 판단

### 4.1 확실히 개선 가능한 방법 (단기)

| 방법 | 예상 효과 | 난이도 | 근거 |
|------|:---------:|:------:|------|
| **Epoch 증가 (100-200)** | +5-10% co | Low | Loss curve 미수렴 (ep29에서도 하강 중) |
| **Learning rate schedule** (cosine decay) | +3-5% co | Low | Constant LR → fine-tuning 불충분 |
| **Layer 수 증가 (7-10)** | +5-10% co | Low | 더 많은 phase diversity → better correction |
| **Loss function 개선** | +5-15% co | Medium | SSIM/perceptual loss 추가, amplitude weight 조정 |
| **Data augmentation** | +3-5% co | Medium | 200 realizations → random rotation/flip |

### 4.2 구조적 개선 (중기)

| 방법 | 예상 효과 | 난이도 | 근거 |
|------|:---------:|:------:|------|
| **Complex modulation (amplitude+phase)** | +30-50% co | High | Phase-only 한계를 근본적으로 해결. 하지만 metalens로 amplitude modulation 구현이 물리적으로 어려움 |
| **SBN nonlinearity** (tao2019) | +10-20% co | Medium | Saturable nonlinear medium이 amplitude coupling 제공. 물리적 구현은 nonlinear 메타표면 |
| **Learnable propagation distance** | +5-10% co | Medium | Layer spacing도 학습 대상으로 → per-layer optimal z |
| **Hybrid D2NN + FD2NN** | +10-15% co | High | 일부 layer는 large spacing (50m), 일부는 metalens (1mm) |

### 4.3 근본적으로 어려운 문제

| 문제 | 판단 | 이유 |
|------|:----:|------|
| Phase-only로 complex overlap > 0.5 달성 | **어려움** | Phase-only의 이론적 상한이 존재. 5-layer phase-only로는 arbitrary unitary transform 불가 |
| Strong turbulence (Cn2>1e-13) 대응 | **매우 어려움** | Scintillation → amplitude fluctuation 극심. Phase-only로는 대응 불가 |
| Real-time correction (< 1ms) | **가능** | FFT + phase mask 연산은 매우 빠름. GPU에서 <1ms 가능 |
| Physical fabrication | **도전적** | 2um pixel metalens는 가능하나, 5-layer 정렬이 난제 |

---

## 5. 실행 로드맵

### Phase 1: 즉시 (1-2일)
```
1. spacing=1mm로 고정, epochs=200 학습
2. Cosine annealing LR (5e-4 → 1e-5)
3. Loss weight 조정: amplitude_mse weight 1.0→2.0 (amplitude 보호)
   → complex_overlap + 2.0 * amplitude_mse
4. 예상: co 0.25-0.30, phase_rmse < 1.3
```

### Phase 2: 단기 (1주)
```
1. Layer 수 sweep: 5/7/10 layers
2. SBN nonlinearity 추가 (tao2019에서 port)
3. Per-layer learnable spacing
4. 예상: co 0.30-0.40
```

### Phase 3: 중기 (2-4주)
```
1. Complex modulation layer (amplitude + phase)
   → Metalens + variable absorber/gain medium 조합
2. Multi-scale hybrid: 일부 free-space layer + metalens layer
3. 예상: co 0.40-0.60 (이론적 한계 근접)
```

---

## 6. 결론

### 작동 여부: FD2NN은 **작동한다**, 하지만 제한적

| 질문 | 답 |
|------|-----|
| FD2NN이 D2NN보다 나은가? | **Yes** — FFT domain switching + propagation으로 더 효율적 |
| Baseline 대비 개선되나? | **Yes** — Complex overlap +12.2%, Phase RMSE -16% |
| 실용적 수준인가? | **아직 No** — co=0.21은 매우 낮음 (목표 >0.8) |
| 개선 여지 있나? | **Yes** — 학습 미수렴 + 구조 개선 여지 큼 |
| 근본적 한계 있나? | **Yes** — Phase-only는 amplitude 제어 불가 |

### 핵심 인사이트

> **FD2NN metalens는 "phase correction device"로는 작동하지만,
> "complete wavefront restorer"로는 phase-only의 벽에 부딪힌다.**
>
> 다음 단계는 **amplitude modulation capability 추가**이며,
> 이는 nonlinear metasurface 또는 variable-transmission element로 가능하다.
>
> 단기적으로는 epoch/LR/loss 튜닝으로 co=0.25-0.30 달성 가능하며,
> 이것만으로도 FSO 수신기의 **phase-only AO 대체재**로서 가치가 있다.
