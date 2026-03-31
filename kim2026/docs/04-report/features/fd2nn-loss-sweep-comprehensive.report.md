---
title: FD2NN Metalens Loss Strategy Sweep
aliases:
  - fd2nn-loss-sweep-comprehensive
tags:
  - kim2026
  - fd2nn
  - optics
  - report
status: reviewed
date: 2026-03-24
---

# FD2NN Metalens Loss Strategy Sweep: 종합 분석 보고서

> Date: 2026-03-24
> Author: Research notebook (자기 기록용)
> Model: BeamCleanupFD2NN (dual-2f, 5 layers, spacing=1mm, dx=2um)
> Data: 1km_cn2e-14_w2m_n1024_dx2mm (Cn2=1e-14, 1024×1024)
> Epochs: 30 (전 실험 동일)
>
> Scope: 본 노트는 1차 `f=1mm, 30 epochs` sweep과 2차 `f=10mm` rerun(일부 `spacing=50mm, 100 epochs`)을 함께 정리한다. 서로 다른 설정 사이 비교는 동일 조건 통제 비교가 아니라는 점을 전제로 읽어야 한다.

> [!info]
> 세부 5관점 검토 메모: [[fd2nn-loss-sweep-comprehensive.review]]

> [!warning]
> 이 노트의 `Peak Proxy`는 고전적 Strehl ratio가 아니라 unit-energy 정규화 후 peak ratio 기반의 집중도 proxy이다. `z/z_diff,10px`는 실제 beam waist의 Rayleigh range가 아니라 Fourier-plane 10-pixel feature에 대한 heuristic diffraction-length 정규화다. 제작성 관련 문장은 모두 공정 검토 전의 가설 수준으로 읽어야 한다.

---

## 0. 이 실험을 왜 했는가

**문제**: FD2NN metalens (phase-only)로 대기 난류 wavefront correction을 하고 싶다.
하지만 어떤 loss function이 최적인지 모른다.

**Phase-only의 근본적 제약**: Metalens는 빛의 위상만 바꿀 수 있고, 진폭(amplitude)은 직접 제어 불가.
그런데 대기 난류는 위상도, 진폭도 동시에 왜곡시킨다.
→ Loss function 선택이 "이 제약 안에서 무엇을 최적화할 것인가"를 결정한다.

**4가지 전략을 비교 실험했다:**

| # | 전략 | 손실함수 | 물리적 의미 |
|:-:|------|---------|------------|
| 02 | Complex | complex_overlap + amplitude_mse | "complex field 전체를 맞춰라" |
| 03 | Phasor | unit phasor MSE | "위상만 맞춰라, 진폭은 무시" |
| 04 | Irradiance | intensity_overlap + beam_radius + encircled_energy | "빔 형태(세기)만 맞춰라" |
| 05 | Hybrid | 위 조합 4종 | "둘 다 어느 정도씩 맞춰라" |

---

## 1. FD2NN metalens가 실용적인가?

### 1.1 결과 요약

![Fig 1: Loss Strategy Comparison](../../../runs/figures_sweep_report/fig1_loss_strategy_comparison.png)

> [!note] Fig 1 해석
> **결론**: 1차 sweep에서는 loss별로 명확한 역할 분담이 생겼다. Complex는 co를, Irradiance는 io를, Hybrid는 절충을, Phasor는 둘 다 놓쳤다.
> **인사이트**: phase-only 시스템에서는 "좋은 loss 하나"보다 "어떤 물리량을 우선할 것인가"가 먼저 결정돼야 한다.

![Fig 6: Training Curves](../../../runs/figures_sweep_report/fig6_training_curves.png)

> [!note] Fig 6 해석
> **결론**: 각 loss는 초반 수렴 속도보다도 최종 수렴 지점이 다르다. 특히 phasor 계열은 metric collapse 성향이 뚜렷하다.
> **인사이트**: 이 문제에서는 optimizer instability보다 objective misalignment가 더 큰 병목이다.

| 전략 | Complex Overlap | Phase RMSE [rad] | Intensity Overlap | Baseline 대비 |
|------|:-:|:-:|:-:|:-:|
| **Baseline (no D2NN)** | **0.191** | — | **0.973** | — |
| Complex | **0.270** (+41%) | **0.359** | 0.378 (-61%) | co 최고 |
| Phasor | 0.098 (-49%) | 0.874 | 0.387 (-60%) | 전부 나쁨 |
| Irradiance | 0.099 (-48%) | 1.679 | **0.933** (-4%) | io 최고 |
| Hybrid (combo3) | 0.126 (-34%) | 1.540 | 0.910 (-6%) | 중간 |

### 1.2 무엇이 달성 가능하고 무엇이 불가능한가

![Fig 3: CO vs IO Trade-off](../../../runs/figures_sweep_report/fig3_co_vs_io_tradeoff.png)

> [!note] Fig 3 해석
> **결론**: 우상단 ideal region에 도달한 점이 없으므로, 현재 sweep 범위에서는 co와 io를 동시에 높이는 해를 찾지 못했다.
> **인사이트**: trade-off는 우연한 run failure가 아니라 목적함수와 phase-only 자유도의 구조적 긴장으로 보는 편이 맞다.

**달성 가능:**
- **Intensity proxy 유지**: Irradiance loss로 io=0.933 달성 (baseline 0.973의 96%).
  → 현재 intensity-overlap 지표 기준으로는 빔 형태를 비교적 잘 유지한다. 다만 direct-detection receiver 유용성은 coupling efficiency, BER, misalignment tolerance 검증 없이는 단정할 수 없다.
- **Phase-sensitive metric 개선**: Complex loss로 masked/global-phase-aligned phase RMSE = 0.36 rad.
  → 이 지표 기준으로는 일부 개선이 보이지만, full-pupil wavefront RMS와 동일하게 해석하면 안 된다.

**현재 불가능:**
- **동시 달성**: co와 io를 동시에 높이는 것. Fig 3 trade-off scatter에서 확인 — ideal region(우상단)에 도달한 점이 없다.
- **완전한 wavefront 복원**: co=0.27은 목표(>0.8)의 34% 수준.
- **Strong turbulence 대응**: Cn2=1e-14에서 이미 한계. 더 강하면 더 어려움.

### 1.3 현재 지표 기준 해석

| 용도 | 판단 | 이유 |
|------|:----:|------|
| Direct detection FSO 빔 정형 | **잠재력 관찰** | io=0.933. 다만 BER, coupling efficiency, misalignment tolerance 등 시스템 지표는 아직 미검증 |
| Coherent 수신 wavefront 보조 | **일부 개선** | support-masked phase metric은 개선되지만, AO 대체를 주장하기엔 증거가 부족 |
| 단독 AO 대체 | **근거 부족** | co=0.27만으로 시스템 레벨 대체를 결론내리기 어렵다 |
| AO 보조 (pre-correction) | **가설 수준** | 잔여 보정 부담 완화 가능성은 있으나 별도 시스템 검증이 필요 |

---

## 2. 어떤 loss가 최적이고 왜?

### 2.1 Complex Loss — 왜 co가 가장 높은가

$$
\mathcal{L}_{\mathrm{complex}}
= \mathcal{L}_{\mathrm{CO}} + 0.5\,\mathcal{L}_{\mathrm{amp}}
= \left(1 - \frac{\left|\langle \mathrm{pred}, \mathrm{target}\rangle\right|}{\lVert \mathrm{pred}\rVert \,\lVert \mathrm{target}\rVert}\right)
+ 0.5\,\mathrm{mean}\!\left(\left(|\mathrm{pred}|-|\mathrm{target}|\right)^2\right)
$$

**왜 이것이 작동하는가:**
- `complex_overlap`은 field의 amplitude와 phase를 동시에 고려하는 유일한 metric.
- Phase correction을 하면 overlap이 올라가지만, amplitude를 심하게 왜곡하면 overlap이 떨어진다.
- `amplitude_mse`가 amplitude 왜곡에 대한 **regularizer** 역할.
- 결과적으로 "amplitude를 너무 깨지 않으면서 phase를 고치는" 방향으로 학습.

**왜 io가 낮은가 (0.378):**
- Phase correction 과정에서 동심원형의 wrapped spectral phase가 나타난다 (Fig 7, row 1).
- 이것은 low-order quadratic phase 성분 또는 defocus-like 성분과 양립하지만, wrapped phase만으로 lensing/refocusing을 단정할 수는 없다.
- 다만 current coherent-field overlap metric 관점에서는 이러한 경로가 유리하게 작동했을 가능성이 있다.

**결론**: 본 sweep의 coherent-field overlap 지표에는 가장 유리했다. 단, intensity proxy 저하를 함께 감수했다.

### 2.2 Phasor Loss — 왜 완전히 실패했는가

$$
\mathcal{L}_{\mathrm{phasor}}
= \mathrm{mean}\!\left(
\left|
\frac{\mathrm{pred}}{|\mathrm{pred}|}
- \frac{\mathrm{target}}{|\mathrm{target}|}
\right|^2
\right)
= \mathrm{mean}\!\left(2(1-\cos \Delta\phi)\right)
$$

단, 구현상 plain phasor MSE는 amplitude threshold mask 위에서 계산된다.

**실패 원인:**
1. **Amplitude 제약 없음**: loss가 amplitude를 전혀 보지 않으므로, 모델이 amplitude를 마음대로 재분배.
2. **Peak collapse**: 총 에너지가 증가한 것이 아니라, 에너지가 몇 pixel에 과도하게 집중되거나 넓게 퍼진다.
3. **Phase도 오히려 악화**: amplitude가 붕괴되면 phase도 의미를 잃음 (amplitude=0인 곳에서 phase는 정의 불가).

**교훈**: Phase-only loss가 phase correction에 좋을 것 같지만, **amplitude에 대한 최소한의 제약이 없으면 전체가 망한다.** Complex overlap이 이 제약을 자연스럽게 포함하고 있었음.

### 2.3 Irradiance Loss — 왜 io는 좋지만 co가 바닥인가

$$
\mathcal{L}_{\mathrm{irr}}
= \mathcal{L}_{\mathrm{IO}}\!\left(|\mathrm{pred}|^2, |\mathrm{target}|^2\right)
+ \mathcal{L}_{\mathrm{BR}}
+ \mathcal{L}_{\mathrm{EE}}
$$

여기서 `intensity_overlap`은 Section 2.1의 `complex_overlap`과 같은 normalized-overlap 구조를 intensity map에 적용한 것이다.

**왜 io가 높은가 (0.933):**
- Loss가 직접 intensity 패턴을 비교하므로, 모델이 빔 형태를 정확히 학습.
- Phase를 적절히 조절하여 diffraction을 통해 amplitude 재분배 → 빔 집중.

**왜 co가 바닥인가 (0.099):**
- Loss에 phase 정보가 전혀 없음 → 모델이 phase를 아무렇게나 설정.
- Intensity가 맞더라도 phase가 완전히 다르면 co는 거의 0.
- 이것은 센서 물리의 차이라기보다 **objective의 phase 민감도 차이**다:
  - `intensity_overlap`은 direct-detection-like, phase-insensitive objective
  - `complex_overlap`은 coherent-field, phase-sensitive objective

**결론**: Direct-detection-like irradiance objective와는 잘 정렬된다. 반면 현재 coherent-field overlap metric에는 불리했다.

### 2.4 Hybrid Loss — 왜 기대만큼 안 되는가

| Combo | 조합 | co | io | 관찰 |
|:-----:|------|:--:|:--:|------|
| 1 | io:1 + co:0.5 | 0.081 | 0.940 | io 지배. co는 무시됨. |
| 2 | io:1 + br:0.5 + ee:0.5 | 0.100 | 0.933 | 순수 irradiance와 거의 동일 |
| 3 | **co:1 + io:0.5 + br:0.5** | **0.126** | **0.910** | **가장 균형. 하지만 둘 다 중간** |
| 4 | phasor + leak + io | 0.220 | 0.407 | Collapsed spot (peak ratio≈30) |

**왜 기대만큼 안 되는가:**
1. **Gradient 경쟁**: 본 sweep들에서는 co loss와 io loss가 서로 다른 방향의 업데이트를 요구하는 경우가 많았다.
2. **관찰된 co↔io 충돌**: 이 setting에서는 co를 올리는 trajectory가 io 하락과 함께 나타났다. 이것을 phase-only optics의 보편 법칙으로 일반화하긴 이르다.
3. **Loss scale 차이**: io loss가 gradient magnitude가 크면 co를 압도하거나, 반대로 co가 io를 압도.

**Combo 4가 왜 Peak Proxy=30인가:**
- `soft_weighted_phasor_loss`는 amplitude로 weight → 큰 amplitude 영역의 phase만 강하게 학습.
- `leakage_loss`는 에너지가 target 밖으로 나가는 것을 방지하지만, target 안에서의 분포는 제약 없음.
- 결과: 모든 에너지를 1-2 pixel에 집중 → Peak Proxy 폭등, io 폭락.

---

## 3. Fig 7 Field Comparison 해석

![Fig 7: Field Comparison](../../../runs/figures_sweep_report/fig7_field_comparison.png)

> [!note] Fig 7 해석
> **결론**: Complex는 ring-like phase를 학습해 coherent metric을 끌어올리지만 intensity residual이 커지고, Irradiance는 target-like beam shape를 복원하지만 phase는 방치된다.
> **인사이트**: 같은 "복원"이라도 coherent-field 복원과 direct-detection beam shaping은 서로 다른 문제다.

### Row 1: Phase 비교 (Complex loss model)

| Panel | 설명 |
|-------|------|
| Input Phase | 난류에 의한 random phase distortion. 불규칙한 패턴. |
| Complex Loss Output Phase | **동심원형 spectral-phase 패턴**. quadratic-like phase shaping 또는 focusing/defocus 성분이 강하게 나타난다. |
| Target Phase | Aperture-limited vacuum reference field의 위상. 이상적 자유공간 Gaussian phase로 단정하긴 어렵다. |
| Phase Error | Output과 target의 위상 차이. Ring 잔류 → 아직 correction 불완전. |

**왜 동심원형 wrapped phase가 나타나는가:**
- Complex overlap loss는 "pred와 target의 inner product를 최대화"해라.
- Inner product가 최대가 되려면 pred의 field 패턴이 target과 닮아야 한다.
- Phase-only로 field를 reshape하는 과정에서 저차의 radially symmetric spectral phase가 나타날 수 있다.
- 선형 phase ramp라면 steering으로, quadratic-like phase라면 defocus/focusing 성분과 양립한다. 다만 wrapped phase만으로 실제 lens 기능을 단정할 수는 없다.

### Row 2: Complex Loss Irradiance

| Panel | 설명 |
|-------|------|
| Input |E|^2 | 난류로 speckle pattern이 된 빔. 에너지 분산. |
| Complex Output |E|^2 | 중심에 밝은 점 + 주변 ring. Refocusing-like redistribution과 양립하지만, 그것만으로 lensing을 단정하진 않는다. |
| Target |E|^2 | Aperture-limited vacuum reference beam. |
| Residual | 큰 차이 — 빔 형태가 다르기 때문. |

### Row 3: Irradiance Loss Irradiance

| Panel | 설명 |
|-------|------|
| Input |E|^2 | 동일한 speckle. |
| Irradiance Output |E|^2 | **Target과 매우 유사한 smooth beam**. Intensity matching 성공. |
| Target |E|^2 | Aperture-limited vacuum reference beam. |
| Residual | 작지만 존재 — 완벽하지는 않음. 현재 objective가 phase를 직접 보지 않고, NA/window/optimization 영향도 함께 반영된다. |

### Row 4: Hybrid Loss Irradiance

| Panel | 설명 |
|-------|------|
| Hybrid Output |E|^2 | Irradiance와 유사하나 약간 더 noisy. co와 io 사이에서 타협한 결과. |

---

## 4. Phase Range 분석

![Fig 2: Phase Range Effect](../../../runs/figures_sweep_report/fig2_phase_range_effect.png)

> [!note] Fig 2 해석
> **결론**: 최소 2π range가 있어야 co가 의미 있게 올라가고, 4π 확장은 물리 자유도 증가라기보다 optimization smoothing에 가깝다.
> **인사이트**: fabrication representation과 optimizer-friendly parameterization은 구분해서 설계해야 한다.

### 왜 phase range가 중요한가

본 모델에서 각 pixel parameter가 표현하는 위상 범위.
- [0, π]: 반파장 지연만 가능. 제한적.
- `[-π, π]`: `mod 2π` wrapping 후 `[0, 2π)` fabrication-view와 물리적으로 동치.
- `[-2π, 2π]`: optimizer-friendly 확장 학습 범위. `mod 2π` wrapping 후 fabrication-view로 사상할 수 있지만, 실제 소자 대응 시에는 wrap/quantization 재검증이 필요하다.

### 결과 (Complex loss 기준)

| Phase Range | Total Range | co | pr |
|:-----------:|:----------:|:--:|:--:|
| [0, π] (sig_pi) | π | 0.219 | 0.625 |
| [-π/2, π/2] (tanh_pi2) | π | 0.251 | 0.411 |
| [0, 2π] (sig_2pi) | 2π | 0.254 | 0.400 |
| [-π, π] (tanh_pi) | 2π | 0.264 | 0.351 |
| [0, 4π] (sig_4pi) | 4π | 0.264 | 0.351 |
| **[-2π, 2π] (tanh_2pi)** | **4π** | **0.270** | **0.359** |

### 해석

1. **π range → 2π range**: 큰 개선 (co: 0.219→0.264). 이 sweep에서는 2π-range parameterization이 π-range보다 유리했다.
2. **2π → 4π**: 미세 개선 (co: 0.264→0.270). 물리적으로 동일 (mod 2π).
3. **4π가 약간 더 좋은 이유**: Optimizer가 2π boundary를 넘을 수 있어서 gradient landscape가 smoother. 학습 trick이지 물리적 차이 아님.
4. **tanh > sigmoid (같은 range)**: Symmetric parameterization이 zero-centered → 초기 학습 안정.
5. **Irradiance loss는 tested phase range에 둔감**: io가 0.899~0.933으로 range에 관계없이 비슷했다. Intensity proxy는 이 sweep에서 phase range보다 모델 구조에 더 민감했다.

### 실용적 결론
- **학습**: tanh_2pi (4π)로 학습 → smooth optimization.
- **제작 검토**: 학습된 phase를 `mod 2π` fabrication representation으로 변환해 검토할 수 있다. 다만 실제 메타표면에서는 효율, 양자화, 편광, 분산, 공정 오차 영향 재검증이 필요하다.

---

## 5. 관찰된 Trade-off의 물리적 해석

### Unit-modulus phase mask의 점wise 제약

Phase-only mask `exp(j·φ(x,y))`를 적용하면:
$$
\left|E_{\mathrm{out}}(x,y)\right|^2
= \left|E_{\mathrm{in}}(x,y)\,e^{j\phi(x,y)}\right|^2
= \left|E_{\mathrm{in}}(x,y)\right|^2
$$
→ **mask 직후에는 점wise intensity가 변하지 않는다.** 이것은 Parseval이 아니라 `|exp(iφ)|=1`인 unit-modulus 성질 때문이다.

Amplitude 재분배는 주로 **inter-layer propagation, finite NA, 유한 window**를 거친 뒤 나타난다.
- Phase mask가 위상 구조를 바꾸고
- propagation/필터링이 출력면 intensity를 재분배한다.

### co vs io Trade-off 메커니즘

```
Observed co-improving trajectories in this sweep:
  → wavefront 변경 → propagation/interference에 의한 field-magnitude 재분배
  → intensity proxy 변화 → io↓

Observed io-preserving trajectories in this sweep:
  → output intensity proxy 유지
  → current coherent-field overlap metric 개선은 제한
```

이것은 본 실험 조건에서 관찰된 강한 trade-off이며, 현재의 loss weight 조정만으로는 해소되지 않았다. 다만 이를 phase-only optics의 보편 법칙으로 일반화하긴 이르다.

---

## 6. 다음에 무엇을 할 것인가

### 단기 (같은 구조로 최적화)

| 방법 | 목적 | 난이도 |
|------|:----:|:------:|
| **Matched-control rerun** (same epochs/spacing/ROI로 f만 sweep) | focal length 효과를 다른 변수와 분리 | Low |
| **Loss sweep @ f=10mm, spacing=6mm** | 2차 설계에서 objective 비교를 동일 조건으로 재실행 | Low |
| **Epoch 증가 + LR schedule** | 남은 headroom 확인 | Low |
| **Layer 수 7-10 / spacing tolerance sweep** | 표현력과 제작 허용오차를 함께 평가 | Medium |

### 중기 (구조 변경)

| 방법 | 기대 효과 | 근거 |
|------|:---------:|------|
| **Complex modulation / amplitude+phase 제어** | trade-off 완화 가능성 | 직접 amplitude 자유도 추가 |
| **Learnable / mixed layer spacing** | 더 넓은 선형 전달함수 탐색 | geometry freedom 증가 |
| **AO-assisted initialization** | coherent metric 개선 가능성 | 시스템 수준 prior 활용 |
| **SBN / nonlinear element** | amplitude-phase coupling 도입 | 위상만으로 부족한 자유도 보완 |

### 장기 (물리적 제작)

1. 동일 조건 통제 sweep으로 제작 후보를 먼저 확정
2. 선택한 phase map을 fabrication representation으로 변환하고 quantization/efficiency를 평가
3. Layer alignment, spacer tolerance, polarization/bandwidth sensitivity를 분석
4. 그 후에야 GDS/process planning과 실험 검증으로 넘어간다

---

## 7. 실험 디렉토리 참조

| # | 디렉토리 | 설명 |
|:-:|---------|------|
| 01 | `중요_01_fd2nn_spacing_sweep_f10mm_claude` | Spacing sweep (0~50mm) |
| 02 | `02_fd2nn_complexloss_roi1024_phase_range_sweep_claude` | Complex loss × phase range |
| 03 | `03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude` | Phasor loss × phase range |
| 04 | `04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude` | Irradiance loss × phase range |
| 05 | `05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude` | Hybrid combo 1-4 |

Figure 경로: `runs/figures_sweep_report/*.png`

---

## 8. f=10mm 재설계 실험 (2차 실험)

### 8.1 왜 f를 바꿨는가

1차 실험의 `f=1mm`에서는 `dx_fourier = λ·f/(n·dx_in) = 0.757µm ≈ λ/2`가 된다. 이것이 곧바로 “물리적으로 불가능한 metasurface pitch”를 뜻하는 것은 아니지만, **1024×1024 independent phase cells를 5-layer로 정렬 구현한다는 관점에서는 매우 공격적인 Fourier-plane sampling proxy**다. `f=10mm`으로 키우면 `dx_fourier = 7.57µm ≈ 4.9λ`가 되어 보다 보수적인 sampling/정렬 proxy를 검토할 수 있다.

| Parameter | 1차 (f=1mm) | 2차 (f=10mm) |
|-----------|:----------:|:-----------:|
| dx_fourier | 0.76 µm (매우 공격적) | **7.57 µm (보수적 후보)** |
| Fourier window | 0.78 mm | **7.75 mm** |
| z_diff,10px | 0.12 mm | **11.6 mm** |
| phase_max | π | **2π** (1차 결과 반영) |

### 8.2 Spacing Sweep 결과 (f=10mm, 30 epochs)

아래 Figure 8A와 Figure 8B는 각각 spacing sweep의 field dashboard와 combined metrics 요약이다.

![Figure 8A: f=10mm spacing sweep field dashboard](../../../figures/f10mm_dashboard/fig_spacing_field_dashboard.png)

> [!note] Figure 8A 해석
> **결론**: spacing이 0mm에서 6mm로 갈수록 보정 구조가 분명해지고, 6mm 부근에서 가장 해석 가능한 correction 패턴이 나온다.
> **인사이트**: inter-layer propagation이 충분히 생겨야 각 layer가 다른 field를 보며 의미 있는 역할 분담을 만든다.

![Figure 8B: f=10mm combined metrics dashboard for spacing and loss sweeps](../../../figures/f10mm_dashboard/fig_combined_metrics.png)

> [!note] Figure 8B 해석
> **결론**: spacing sweep과 loss sweep을 한 화면에서 보면, "spacing plateau 상단"과 "loss-sweep CO 최대점"이 같지 않다. spacing에서는 6mm가, loss에서는 Hybrid@50mm가 co 최댓값을 만든다.
> **인사이트**: geometry와 objective는 독립 축이 아니며, metric 우선순위에 따라 최적 조합이 달라진다.

| Spacing | z/z_diff,10px | CO ↑ | IO ↑ | φ RMSE ↓ | Peak Proxy |
|:-------:|:-----:|:----:|:----:|:--------:|:------:|
| 0 mm | 0 | 0.165 | 0.923 | 1.749 | 2.55 |
| 1 mm | 0.09 | 0.197 | 0.932 | 1.691 | 2.04 |
| 3 mm | 0.26 | 0.275 | 0.914 | 1.520 | 2.16 |
| **6 mm** | **0.52** | **0.316** | **0.901** | **1.356** | **2.23** |
| 12 mm | 1.03 | 0.313 | 0.890 | 1.315 | 2.62 |
| 25 mm | 2.15 | 0.302 | 0.889 | 1.294 | 2.66 |
| 50 mm | 4.31 | 0.305 | 0.884 | 1.285 | 2.76 |

- **z/z_diff,10px < 0.3**: 전파는 있으나 identity에 가까워 layer들이 거의 같은 field를 봄. coupling이 약하고 중복적 동작에 가깝다.
- **z/z_diff,10px ≈ 0.5 (6mm)**: 관찰된 plateau의 상단부다. 동심원형 wrapped phase가 가장 구조적으로 보이며 CO=0.316 (+65% vs baseline).
- **z/z_diff,10px > 1**: CO plateau. 이 추세는 diffraction-length heuristic뿐 아니라 NA/window/band-limit/optimization conditioning 영향도 함께 반영한다.

### 8.3 Loss 함수 비교 (f=10mm, spacing=50mm, 100 epochs)

아래 Figure 8C와 Figure 8B의 loss-sweep 패널이 `spacing=50mm, 100 epochs` loss 비교를 요약한다.

![Figure 8C: f=10mm loss sweep field dashboard at spacing=50mm](../../../figures/f10mm_dashboard/fig_loss_field_dashboard.png)

> [!note] Figure 8C 해석
> **결론**: 같은 spacing=50mm에서도 loss에 따라 출력 field morphology가 크게 달라진다. Hybrid는 coherent 정렬을, Phasor는 spike-like collapse를, Complex/Irradiance는 유사한 해를 만든다.
> **인사이트**: 이 setting에서 objective 차이가 실제 공간 패턴 차이로 번역된다는 점이 수치표를 뒷받침한다.

#### 손실함수 수식

> [!warning]
> 이 2차 loss sweep은 실험 스크립트 라벨을 그대로 따른다. 여기서 `Complex*`는 Section 2.1의 `complex_overlap + 0.5 amplitude_mse`와 같은 objective가 아니라, `CO + IO + BR + EE`를 함께 쓰는 별도 조합이다.

| Loss | 수식 | weights |
|------|------|---------|
| Complex* | `L_CO + L_IO + L_BR + L_EE` | all 1.0 |
| Phasor | `soft_weighted_phasor_loss + 0.5·leakage` | soft_phasor:1.0, leak:0.5 |
| Irradiance | `L_IO + L_BR + L_EE` | all 1.0 |
| Hybrid | `L_CO + 0.5·L_IO + 0.5·L_BR` | CO주, IO/BR보조 |

여기서
$$
\begin{aligned}
L_{\mathrm{CO}} &= 1 - \frac{\left|\langle \mathrm{pred}, \mathrm{target}\rangle\right|}{\lVert \mathrm{pred}\rVert \,\lVert \mathrm{target}\rVert}, \\
L_{\mathrm{IO}} &= 1 - \operatorname{normalizedOverlap}\!\left(|\mathrm{pred}|^2, |\mathrm{target}|^2\right), \\
L_{\mathrm{BR}} &= \left(w_{\mathrm{pred}} - w_{\mathrm{target}}\right)^2, \\
L_{\mathrm{EE}} &= \left(\mathrm{EE}_{\mathrm{pred}} - \mathrm{EE}_{\mathrm{target}}\right)^2.
\end{aligned}
$$

Phasor run은 plain phasor MSE가 아니라 global phase alignment와 target-amplitude weighting이 들어간 `soft_weighted_phasor_loss`를 쓴다. **amp_mse 제거** (1차 대비 변경).

| Loss | CO ↑ | IO ↑ | φ RMSE ↓ | Peak Proxy |
|------|:----:|:----:|:--------:|:------:|
| Baseline | 0.191 | 0.973 | 1.707 | — |
| Complex* | 0.317 | 0.889 | 1.281 | 2.73 |
| Phasor | 0.288 | 0.334 | **0.805** | 27.8 |
| Irradiance | 0.317 | 0.889 | 1.281 | 2.73 |
| **Hybrid** | **0.350** | 0.845 | 1.218 | 2.59 |

#### 핵심 발견

1. **이 setting에서는 보고된 `Complex*`와 `Irradiance` 요약 metric이 일치했다**: 다만 이것만으로 `L_CO`가 추가 신호를 거의 못 줬다고 단정할 수는 없다. raw objective는 서로 다르며, 같은 구현인지 같은 basin인지 여부는 별도 artifact 검증이 필요하다.

2. **Hybrid CO 최고 (0.350)**: CO weight=1.0이 IO/BR(0.5)를 압도하면서 beam collapse 방지. 1차 Hybrid(0.126) 대비 **+178%**.

3. **Phasor: 위상 최적(0.805 rad)이나 빔 파괴(IO=0.334)**: Beam profile에서 극단적 spike 확인.

### 8.4 설정 변경을 포함한 1차 vs 2차 비교

| 항목 | 1차 (f=1mm, 30ep) | 2차 (f=10mm, 100ep) |
|------|:------------------:|:-------------------:|
| Best CO | 0.270 | **0.350** (+30%) |
| dx_fourier | 0.76 µm (공정 부담 큼) | **7.57 µm (보수적 후보)** |
| Plateau 상단 spacing | 1 mm | **6 mm** (`z/z_diff,10px≈0.5`) |

단, 이 비교는 epoch, spacing, ROI, phase range까지 함께 바뀐 **설정 묶음 비교**다. focal length 단독 효과로 해석하면 과하다.

---

## 9. 실험 디렉토리 참조 (업데이트)

| # | 디렉토리 | f | Epochs |
|:-:|---------|:-:|:------:|
| 01 | `중요_01_fd2nn_spacing_sweep_f10mm_claude` | **10mm** | 30 |
| 02 | `02_fd2nn_complexloss_roi1024_phase_range_sweep_claude` | 1mm | 30 |
| 03 | `03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude` | 1mm | 30 |
| 04 | `04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude` | 1mm | 30 |
| 05 | `05_fd2nn_hybridloss_roi1024_loss_combo_sweep_claude` | 1mm | 30 |
| 06 | `06_fd2nn_loss_sweep_f10mm_sp50mm_claude` | **10mm** | **100** |

---

## 10. 한 줄 결론

> **이 문서의 sweep 범위에서는, 1차(`f=1mm`) 설정에서 co와 io를 동시에 크게 끌어올리는 loss를 찾지 못했다.**
> **2차(`f=10mm`) rerun에서는 목표 metric에 따라 후보가 갈린다: CO 최대치는 `Hybrid@50mm`, plateau 상단의 균형 후보는 `Complex-loss spacing sweep@6mm`였다. 다만 이 둘은 동일 통제 조건 비교가 아니므로 가설 후보로만 유지해야 한다.**
> **Amplitude modulation, geometry 변경, tolerance analysis 없이 fabrication/receiver 성능을 단정할 단계는 아직 아니다.**
>
> **2차 실험 추가: 이 setting에서는 보고된 `Complex*`와 `Irradiance`의 요약 metric이 같았고, `Hybrid(CO:1+IO:0.5+BR:0.5)`가 CO=0.350으로 최고였다.**

---

## Appendix A. 왜 Irradiance Loss가 가장 잘 되는가 — 물리적 심층 분석

> btw-001 분석 노트 (2026-03-24)

### A.1. 핵심 질문

Irradiance loss (io=0.933)는 baseline (io=0.973)의 96%에 도달했지만,
Complex loss (io=0.378)는 39%에 불과하다. **2.5배 차이의 물리적 원인은 무엇인가?**

### A.2. Phase-Only Mask의 구조적 성질

FD2NN의 각 layer는 `FourierPhaseMask`로 구현되어 있다:

$$
\mathrm{output} = \mathrm{field}\,e^{i\phi},
\qquad |e^{i\phi}| \equiv 1
$$

이 연산의 물리적 의미:

1. **진폭 불변**: |output| = |field| × |exp(iφ)| = |field|. 단일 mask는 amplitude를 절대 바꿀 수 없다.
2. **mask 직후의 총 power는 이상적 unitary limit에서 유지**되지만, 본 구현 전체는 NA mask와 band-limited propagation을 포함하므로 최종 출력 power가 항상 엄밀히 같다고 볼 수는 없다.
3. **위상 자유도만 존재**: N×N mask에 N² 개의 자유 파라미터가 있지만, 전부 위상 변조에만 쓰인다.

### A.3. Inter-layer Propagation이 만드는 Phase→Amplitude 변환

Phase-only mask 하나만으로는 amplitude를 바꿀 수 없다. 하지만 **layer 사이의 자유공간 전파 (angular spectrum propagation)**가 핵심적인 역할을 한다:

```
Layer k: field → exp(iφ_k) × field     (phase modulation)
   ↓
Propagation: Angular Spectrum, z=1mm    (phase→amplitude 변환)
   ↓
Layer k+1: 새로운 amplitude 분포에 대해 phase modulation
```

자유공간 전파에서 일어나는 일:

- Phase modulation은 각 pixel에 서로 다른 위상 지연을 부여
- 전파 과정에서 인접 pixel들의 복소 field가 **간섭** (constructive/destructive)
- 이 간섭이 amplitude를 재분배 → **phase modulation이 intensity redistribution으로 변환**

이것이 spacing=0mm에서 성능이 나빴던 이유이다 (sweep 01 결과):
spacing=0이면 모든 phase mask가 단일 Fourier-plane phase filter에 가깝게 collapse되어 중간 plane 다양성이 사라진다. 출력면 intensity reshaping 자체는 여전히 가능하지만, inter-layer propagation이 만드는 추가 자유도는 없다.

### A.4. Loss Function별 Gradient가 Phase Mask에 미치는 영향

#### Irradiance Loss (io + br + ee)

$$
\mathcal{L}_{\mathrm{irr}}
= w_1\!\left(1-\frac{\langle I_{\mathrm{pred}}, I_{\mathrm{target}}\rangle}{\lVert I_{\mathrm{pred}}\rVert \,\lVert I_{\mathrm{target}}\rVert}\right)
+ w_2\,\mathrm{MSE}(r_{\mathrm{pred}}, r_{\mathrm{target}})
+ w_3\,\mathrm{MSE}(\mathrm{EE}_{\mathrm{pred}}, \mathrm{EE}_{\mathrm{target}})
$$

- **최적화 대상**: intensity 분포의 형태 (shape matching)
- **gradient 방향**: "에너지를 target과 같은 공간 분포로 재배치하라"
- **phase mask에 대한 요구**: 간섭 패턴을 조절하여 output-plane 에너지를 재배치하라

→ Phase mask가 간접적으로 할 수 있는 일(전파를 통한 에너지 재분배)과 loss가 요구하는 일(intensity matching)이 **상대적으로 더 잘 정렬**된다.

#### Complex Loss (co + amp_mse)

$$
\mathcal{L}_{\mathrm{cmplx}}
= w_1\!\left(1-\frac{\left|\langle E_{\mathrm{pred}}, E_{\mathrm{target}}\rangle\right|}{\lVert E_{\mathrm{pred}}\rVert \,\lVert E_{\mathrm{target}}\rVert}\right)
+ w_2\,\mathrm{MSE}\!\left(|E_{\mathrm{pred}}|, |E_{\mathrm{target}}|\right)
$$

- **최적화 대상**: complex field 전체 (field magnitude + phase)
- **gradient 방향**: "각 pixel에서 amplitude도 맞추고 phase도 맞춰라"
- **phase mask에 대한 요구**: output-plane amplitude/magnitude mismatch까지 줄여라

→ 단일 phase mask는 자기 평면에서 amplitude를 직접 바꿀 수 없고, multi-plane 구조에서도 output-plane amplitude는 propagation/interference를 통해서만 간접적으로 제어된다. 따라서 Loss가 요구하는 complex-field similarity와 mask의 제어 자유도 사이에 **구조적 불일치**가 존재한다.

Appendix C의 ablation 결과를 보면, `amp_mse` 자체가 1차 setting의 결과를 크게 바꾸지는 않았다. 따라서 이 구간의 핵심은 “amplitude regularizer가 beam을 망쳤다”기보다, **phase-sensitive overlap을 끌어올리는 경로가 intensity 분포를 희생하는 방향으로 형성되었다**는 점에 가깝다.

결과: optimizer가 phase-sensitive metric을 올리면서 intensity 분포를 크게 왜곡 → io=0.378.

#### Phasor Loss (unit phasor MSE)

$$
\mathcal{L}_{\mathrm{phasor}}
= \sum \left|
\frac{E_{\mathrm{pred}}}{|E_{\mathrm{pred}}|}
- \frac{E_{\mathrm{target}}}{|E_{\mathrm{target}}|}
\right|^2
$$

실제 plain phasor 구현은 amplitude threshold mask 위에서 계산된다.

- **최적화 대상**: 위상만 (amplitude 완전 무시)
- **gradient 방향**: "위상을 맞춰라, amplitude는 알아서 되겠지"
- **phase mask에 대한 요구**: 위상 correction

→ 문제: amplitude에 대한 제약이 전혀 없어서 optimizer가 에너지를 몇 pixel에 과도하게 집중시키거나 넓게 퍼뜨린다.

amplitude threshold mask (|E| > 0.1·max|E|)가 있지만, mask가 학습 과정에서 변하면서
"high amplitude pixel"의 위치가 shift → phase target도 drift → **불안정한 학습**.

결과: phase도 오히려 악화 (pr=0.874, baseline보다 나쁨), intensity 파괴 (io=0.387).

### A.5. Unit-modulus 제약, 유효 손실, 그리고 에너지 bookkeeping

Dual-2f 시스템에서:

$$
E_{\mathrm{in}}
\xrightarrow{\mathrm{FFT}}
\text{Fourier plane}
\xrightarrow{\text{phase masks + propagation}}
\xrightarrow{\mathrm{IFFT}}
E_{\mathrm{out}}
$$

이상적 unitary계에서는 Parseval에 따라 **Σ|E_in|² = Σ|E_out|²**가 성립한다. 다만 본 구현은 NA filtering과 band-limited propagation을 포함하므로, 실제 수치 실험에서는 총 power가 이 이상화와 정확히 일치하지 않을 수 있다.

이 보존 법칙의 함의:

| 속성 | 가능 여부 | 이유 |
|------|:--------:|------|
| Total power 보존 | ⚠️ 이상적 unitary sub-operator에서는 성립 | NA/window/filter 포함 전체 구현에서는 직접 측정 필요 |
| Intensity 공간 재분배 | ✅ 가능 | 다층 phase mask + propagation에 의한 간섭 |
| 특정 pixel의 amplitude 증가 | ⚠️ 제한적 | 다른 pixel의 amplitude 감소 수반 |
| Pixel-wise amplitude control | ⚠️ 직접 제어 없음 | phase만 조절하고 amplitude는 propagation/interference를 통해 간접 재분배 |
| Global beam shape matching | ✅ 가능 | beam radius, encircled energy 같은 output-plane objective는 달성 가능 |

→ **Irradiance loss가 요구하는 output-plane shape matching은 현재 시스템과 잘 정렬되고,
Complex loss는 직접 제어하기 어려운 complex-field 오차까지 함께 줄이려 한다.**

### A.6. Fourier Plane에서의 물리적 직관

FD2NN은 Fourier plane에서 동작한다. 입력 field의 FFT가 phase mask에 도달한다.

- **Fourier plane의 phase modulation**은 일반적으로 **spectral-phase filtering / PSF shaping**에 가깝다.
- 선형 phase ramp 성분은 beam steering으로, quadratic-like phase 성분은 defocus/focusing으로 해석할 수 있다.

대기 난류가 만드는 wavefront 왜곡의 효과:
- 위상 왜곡 → beam이 여러 방향으로 흩어짐 (speckle)
- Amplitude 왜곡 (scintillation) → intensity 불균일

Phase-only correction으로:
- **Beam steering 복원 (에너지 재집중)**: 가능 ✅ → irradiance-like objective가 활용
- **Scintillation 보정 (pixel-wise amplitude control)**: 직접 제어는 어려움 ⚠️ → complex objective와의 정렬이 약함

### A.7. 수치적 증거 정리 — 난류 입력 대비 전체 비교

#### A.7.1. 절대값 비교 (test set 평균, tanh_2pi config)

| Strategy | co | io | pr [rad] | amp RMSE |
|----------|:--:|:--:|:--------:|:--------:|
| **Turbulent (보정 없음)** | 0.191 | 0.973 | — | ~0.02 |
| **Complex (co+amp)** | **0.270** | 0.378 | **0.359** | 0.176 |
| **Irradiance (io+br+ee)** | 0.099 | **0.933** | 1.679 | 0.159 |
| **Hybrid (co+io+br)** | 0.126 | 0.910 | 1.540 | 0.164 |
| **Phasor (phase only)** | 0.098 | 0.387 | 0.874 | 0.179 |

#### A.7.2. 난류 대비 변화량

| Strategy | Δco | Δio | amp RMSE 변화 | 판정 |
|----------|:---:|:---:|:------------:|:----:|
| **Complex** | +0.079 (+41%) ✅ | **−0.595 (−61%)** ❌ | 0.02→0.176 (8.8×악화) | phase↑ beam↓↓ |
| **Irradiance** | −0.092 (−48%) ❌ | −0.040 (−4%) ✅ | 0.02→0.159 (8.0×악화) | beam≈유지 |
| **Hybrid** | −0.065 (−34%) | −0.063 (−6%) | 0.02→0.164 (8.2×악화) | 절충 |
| **Phasor** | −0.093 (−49%) ❌ | **−0.586 (−60%)** ❌ | 0.02→0.179 (9.0×악화) | 전부 악화 |

#### A.7.3. 이 표에서 드러나는 5가지 사실

**① 이 데이터셋에서는 난류 입력의 io가 이미 0.973으로 매우 높다.**
현재 aperture와 overlap metric 아래에서는 receiver-plane intensity proxy가 vacuum target과 비교적 가깝게 유지된다.
난류 입력의 amp RMSE ≈ 0.02 (현재 scalar amplitude metric 기준으로는 작은 오차).

**② Complex loss는 io를 난류보다 더 나쁘게 만든다 (0.973→0.378).**
Phase correction을 위해 간섭 패턴을 재구성하는 과정에서 beam shape을 파괴한다.
D2NN을 **안 쓰는 것보다 못한** 결과.

**③ Irradiance loss는 이미 좋은 io를 유지한다 (0.973→0.933, −4%).**
난류가 거의 보존한 intensity 분포를 D2NN 통과 후에도 유지.
이 차이는 NA/window/passband truncation과 learned redistribution의 합성 효과로 보는 편이 안전하다.

**④ amp RMSE가 ~0.02→~0.17로 모든 전략에서 8~9배 악화된다.**
이것은 다음 해석을 지지한다:
- 난류 입력은 amplitude가 거의 완벽 (RMSE 0.02)
- D2NN 통과 후 모든 전략에서 amplitude가 비슷하게 악화 (0.159~0.179)
- **어떤 loss를 쓰든 이 설정에서 amplitude는 직접 제어되지 않고 간접 재분배만 가능했다**
- 전략 차이는 scalar amp RMSE 하나보다 **field magnitude / irradiance의 공간 분포와 phase residual**에서 더 크게 드러난다.

**⑤ 현재 sweep들에서는 co와 io를 동시에 크게 개선한 전략을 찾지 못했다.**
이 결과는 본 설정에서 co를 올리는 trajectory가 io 하락과 함께 나타났음을 뜻한다.
이는 phase-only system의 보편 법칙이라기보다, 현재 architecture/objective/metric 조합에서 관찰된 **empirical co↔io trade-off**로 읽는 편이 안전하다.

#### A.7.4. 시각화

![Fig 8: Amplitude/Intensity/Phase — Complex vs Irradiance](../../../runs/figures_sweep_report/fig8_loss_physics_evidence.png)

> [!note] Fig 8 해석
> **결론**: Complex와 Irradiance의 차이는 amplitude 자체보다 에너지의 공간 분포와 phase residual에 있다.
> **인사이트**: amp RMSE 하나로는 beam quality를 설명하기 어렵고, spatial redistribution과 phase sensitivity를 함께 봐야 한다.

### A.8. 결론: Loss-Physics Alignment 원리

> **최적의 loss function은 물리적 시스템의 제어 가능한 자유도와 정렬되어야 한다.**

| 시스템 특성 | 제어 가능 | Loss 정렬 |
|------------|:---------:|:---------:|
| Phase-only mask | Phase modulation | ✅ 모든 loss |
| Multi-layer + propagation | Intensity redistribution | ✅ Irradiance loss |
| Energy conservation (Parseval) | 이상적 unitary sub-operator에서 total power 보존 | ✅ bookkeeping 해석에는 도움 |
| Pixel-wise amplitude control | ⚠️ 직접 제어 없음 | ⚠️ Complex objective와 정렬이 약함 |

**Irradiance loss가 io=0.933을 달성한 이유**:
1. Phase-only mask + inter-layer propagation의 물리적 능력 = **에너지 공간 재분배**
2. Irradiance loss의 최적화 목표 = **에너지 공간 분포 matching**
3. 두 가지가 더 잘 정렬 → 현재 sweep에서는 효율적 수렴이 관찰됨

**Complex loss가 io=0.378에 그친 이유**:
1. Phase-sensitive overlap을 높이는 경로가 interference pattern을 크게 재구성했다.
2. Appendix C ablation 기준으로는 `amp_mse` 자체보다 coherent-field objective가 더 지배적으로 보였다.
3. 결과적으로 phase metric은 개선됐지만 beam shape는 희생됐다.

이것은 **"불가능한 목표를 포함한 loss는 가능한 목표마저 해친다"**는
최적화의 일반 원리를 보여주는 사례이다.

---

## Appendix B. Focal Length 수정 및 Spacing Sweep 재실험 (f=10mm)

> 2026-03-24 추가. 이전 실험의 f=1mm 설계 오류 발견 → f=10mm으로 수정하여 재실험.

### B.1. 왜 f를 바꿔야 했는가

이전 모든 실험(Run 01~05)은 dual-2f focal length **f=1mm**으로 수행되었다.
이때 Fourier plane pixel pitch는:

$$
dx_{\mathrm{fourier}}
= \frac{\lambda f}{n\,dx_{\mathrm{in}}}
= \frac{1.55\,\mu\mathrm{m}\times 1\,\mathrm{mm}}{1024\times 2\,\mu\mathrm{m}}
= 0.757\,\mu\mathrm{m}
\approx \frac{\lambda}{2}
$$

이 `dx_fourier`는 ideal 2f Fourier-plane sampling pitch다. 이것을 곧바로 meta-atom lattice constant와 동일시할 수는 없지만, **이상적인 1024×1024 independent phase cells를 5-layer 정렬 구현한다는 관점에서는 매우 공격적**이다.

| f (mm) | dx_fourier (µm) | dx/λ | sampling / fabrication proxy |
|--------|:-:|:-:|:-:|
| 1 | 0.76 | 0.49 | 매우 공격적, 다층 정렬 부담 큼 |
| 5 | 3.78 | 2.44 | 공정/효율 검토 필요 |
| **10** | **7.57** | **4.9** | **보수적 후보** |
| 20 | 15.1 | 9.8 | 보수적 후보 |

**f=10mm을 선택한 이유**: `dx_fourier=7.57µm (4.9λ)`는 이 워크플로우에서 더 여유 있는 fabrication representation을 주고, Fourier window=7.75mm로 시스템 크기도 여전히 관리 가능하다.

### B.2. 시스템 파라미터 비교

| Parameter | Old (Run 01~05) | New (B실험) |
|-----------|:-:|:-:|
| f₁ = f₂ | 1 mm | **10 mm** |
| dx_fourier | 0.76 µm (λ/2) | **7.57 µm (4.9λ)** |
| fourier window | 0.78 mm | **7.75 mm** |
| phase_max | π (Run01), 2π (Run02~05) | **2π** |
| ROI | 512 (center crop) | **1024 (full field)** |
| 기타 | 동일 | 동일 |

고정 파라미터: λ=1.55µm, n=1024, dx_in=2µm, NA=0.16, 5 layers, 30 epochs, lr=5e-4.

### B.3. Spacing 설계 — heuristic diffraction-length scale

여기서의 `z_diff,10px`는 실제 beam waist의 Rayleigh range가 아니라, Fourier-plane의 `10-pixel feature`를 Gaussian-like transverse scale로 근사해 만든 비교용 diffraction-length다.

f=10mm에서 Fourier plane의 10-pixel feature에 대한 heuristic diffraction length:

$$
z_{\mathrm{diff},10\mathrm{px}}
= \frac{\pi (10\,dx_{\mathrm{fourier}})^2}{\lambda}
= \frac{\pi (75.7\,\mu\mathrm{m})^2}{1.55\,\mu\mathrm{m}}
= 11.6\,\mathrm{mm}
$$

| Config | Spacing | z/z_diff,10px | heuristic mixing regime |
|--------|:-:|:-:|:-:|
| spacing_0mm | 0 mm | 0 | Stacked (전파 없음) |
| spacing_1mm | 1 mm | 0.09 | very small relative to heuristic scale |
| spacing_3mm | 3 mm | 0.26 | weak feature mixing |
| spacing_6mm | 6 mm | 0.52 | intermediate feature mixing |
| spacing_12mm | 12 mm | 1.0 | comparable to heuristic scale |
| spacing_25mm | 25 mm | 2.2 | strong feature mixing |
| spacing_50mm | 50 mm | 4.3 | very large relative to heuristic scale |

### B.4. 최종 결과 (7/7 config 완료)

![Fig 9: Spacing Sweep f=10mm — CO Plateau & IO Trade-off](../../../runs/figures_sweep_report/fig9_spacing_sweep_f10mm.png)

> [!note] Fig 9 해석
> **결론**: co는 `z/z_diff,10px≈0.5` 부근에서 최고점에 도달한 뒤 plateau를 형성하고, io는 완만하게만 감소한다.
> **인사이트**: f=10mm setting에서는 1차 sweep보다 trade-off 곡선이 훨씬 완만해져 geometry 변경 효과가 실제로 있었음을 보여준다.

![Fig 10: Phase RMSE / Peak Proxy / Amp RMSE vs Spacing](../../../runs/figures_sweep_report/fig10_phase_strehl_spacing.png)

> [!note] Fig 10 해석
> **결론**: spacing이 커질수록 phase RMSE는 계속 감소하고 Peak Proxy는 증가하지만, co는 6mm 이후 더 좋아지지 않는다.
> **인사이트**: 낮은 phase RMSE나 높은 peak concentration만으로는 global coherent overlap을 보장하지 않는다.

> **Loss**: Complex loss (complex_overlap + 0.5×amplitude_mse, Section 2.1과 동일). Phase range: tanh_2pi.

| Config | z/z_diff,10px | CO ↑ | IO ↑ | φ RMSE ↓ (rad) | Peak Proxy | Baseline 대비 CO |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Baseline (no D2NN)** | — | **0.191** | **0.973** | 1.707 | — | — |
| spacing_0mm | 0 | 0.165 | 0.923 | 1.749 | 2.55 | −14% ❌ |
| spacing_1mm | 0.09 | 0.197 | 0.932 | 1.691 | 2.04 | +3% |
| spacing_3mm | 0.26 | 0.275 | 0.914 | 1.520 | 2.16 | +44% ✅ |
| **spacing_6mm** | **0.52** | **0.316** | **0.901** | **1.356** | 2.23 | **+65%** ✅ |
| spacing_12mm | 1.03 | 0.313 | 0.890 | 1.315 | 2.62 | +63% ✅ |
| spacing_25mm | 2.15 | 0.302 | 0.889 | 1.294 | 2.66 | +58% ✅ |
| spacing_50mm | 4.31 | 0.305 | 0.884 | 1.285 | 2.76 | +59% ✅ |

### B.5. 물리적 해석 — f=10mm에서 관찰되는 현상

#### ① CO는 `z/z_diff,10px ≈ 0.5` 근방에서 peak, 이후 plateau

```
CO: 0.165 → 0.197 → 0.275 → 0.316 → 0.313 → 0.302 → 0.305
     0mm     1mm     3mm     6mm★    12mm    25mm    50mm
```

이것은 Appendix A에서 분석한 **inter-layer propagation이 후속 마스크가 보는 field를 바꾸며 자유도를 늘린다**는 점과 정합적이다.

- **spacing_0mm (CO=0.165 < baseline 0.191)**: inter-layer propagation이 사라져 5개 phase mask가 사실상 하나의 Fourier-plane phase filter처럼 작동한다. 출력 reshaping은 여전히 가능하지만, 중간 plane 다양성은 없다.
- **spacing_1mm (`z/z_diff,10px=0.09`)**: Transfer function이 identity에 가까워 각 layer가 거의 같은 field를 본다. “독립”이라기보다 중복적 동작에 가깝다.
- **spacing_3mm (`z/z_diff,10px=0.26`)**: 약한 회절 커플링 시작. 인접 feature가 더 많이 섞이며 실현 가능한 선형 전달함수의 폭이 넓어진다. CO +44%.
- **spacing_6mm (`z/z_diff,10px=0.52`)**: 중간 회절. Feature mixing이 충분하여 각 layer가 다른 spatial scale의 correction을 담당. **CO peak = 0.316 (+65%).**
- **spacing_12mm (`z/z_diff,10px=1.03`)**: CO=0.313으로 6mm와 거의 동일. 추가 mixing이 CO 개선에 기여하지 못함. 이미 표현력 포화.
- **spacing_25mm (`z/z_diff,10px=2.15`)**: CO=0.302로 약간 하락. 이 추세는 z 자체뿐 아니라 유한 NA/window, band-limit, 샘플링, optimization conditioning 영향을 함께 반영한다.
- **spacing_50mm (`z/z_diff,10px=4.31`)**: CO=0.305로 25mm과 비슷. 큰 하락이 없다는 점은 이 heuristic 축에서 **broad plateau (0.30~0.32)**가 형성됨을 보여준다.

**결론**: 현재 heuristic 축에서는 `z/z_diff,10px=1.0` 이후 추가 개선이 거의 없었다. **경험적으로는 `z/z_diff,10px ≈ 0.5` (6mm) 근방이 plateau 상단부였고 이후 plateau가 형성된다.** 다만 이것을 보편적 회절 법칙으로 일반화하긴 이르다.

#### ② IO 감소 폭이 이전보다 작다

f=1mm 실험에서는 Complex loss로 io=0.378 (−61%)까지 떨어졌으나,
f=10mm에서는 spacing_50mm(최대)에서도 io=0.884 (−9%).

```
IO: 0.923 → 0.932 → 0.914 → 0.901 → 0.890 → 0.889 → 0.884
     0mm     1mm     3mm     6mm     12mm    25mm    50mm
```

IO는 spacing 증가에 따라 **완만하게 감소**하지만, 전 구간에서 0.88 이상을 유지한다.

**가능한 이유**: `dx_fourier`가 커지고, NA/window/spacing 조건이 바뀌면서 output redistribution이 더 완만한 regime에 들어갔을 수 있다. 다만 이는 focal length 단독 효과라기보다 설정 묶음의 결과다.

#### ③ 학습된 Phase Mask 패턴

아래 B.7에 삽입한 phase dashboard를 보면:

- **spacing_0mm**: 입력 turbulent phase와 거의 동일 (보정 실패)
- **spacing_1mm**: 중심부에 약한 동심원 힌트
- **spacing_3mm**: 복잡한 phase 구조 — 난류 보상 패턴 형성 중
- **spacing_6mm**: **선명한 동심원형 wrapped phase** — target vacuum beam 구조와 양립하는 저차 phase 성분이 강해 보임
- **spacing_12mm~50mm**: 동심원형 wrapped phase 경향은 유지되되, 고주파 성분이 점차 smoothing된다. 50mm에서 가장 부드러운 패턴이다.

#### ④ Beam Profile 비교

아래 B.7에 삽입한 beam-profile dashboard를 보면:

- spacing_6mm의 radial profile이 target Gaussian에 가장 근접
- 모든 prediction이 r > 200µm에서 target보다 높은 wing (sidelobe 에너지 누출)
- spacing_6mm의 centerline phase error가 중심부 ±50µm에서 ≈ 0 → **위상 보정 성공 영역**

### B.6. 이전 실험(f=1mm)과의 비교

| 비교 항목 | f=1mm (Run 01, spacing=1mm, complex loss) | f=10mm (spacing=6mm, complex loss) |
|----------|:-:|:-:|
| dx_fourier | 0.76 µm (공정 부담 큼) | 7.57 µm (보수적 후보) |
| CO (best) | 0.215 | **0.316** |
| IO (best CO에서) | 0.650 | **0.901** |
| φ RMSE | 1.511 rad | **1.356 rad** |
| CO 개선율 (vs baseline) | +12% | **+65%** |
| IO 감소율 | −33% | **−7%** |

**현재 비교 조건에서는 `f=10mm` 설정군이 더 좋아 보인다.** 특히 IO 감소가 −33% → −7%로 완화되었다. 다만 이는 focal length만 바뀐 실험이 아니다.

이유:
1. `dx_fourier`가 커져 보다 보수적인 sampling/fabrication representation을 사용했다.
2. spacing, phase range, epoch, ROI가 함께 바뀌어 optimizer가 다른 regime에 들어갔다.
3. 따라서 개선 원인은 focal length 하나가 아니라 **설정 묶음 전체**에 가깝다.

### B.7. Figure 삽입

이전 버전에서 경로로만 적어둔 spacing-sweep 관련 figure를 아래에 직접 삽입한다.
Fig 9와 Fig 10은 이미 B.4에 삽입되어 있으므로, 여기서는 나머지 dashboard와 부가 figure를 정리한다.

#### B.7.1 Dashboard

![Spacing-sweep dashboard: irradiance across all spacing configurations](../../../figures/spacing_sweep_f10mm/dashboard_irradiance_spacing.png)

> [!note] Irradiance Dashboard 해석
> **결론**: 모든 spacing에서 intensity는 비교적 target-like하게 유지되지만, 6mm 근방이 중심 lobes와 주변 leakage의 균형이 가장 좋다.
> **인사이트**: f=10mm setting에서는 "보정하면 beam이 망가진다"는 1차 실험의 극단적 양상이 완화됐다.

![Spacing-sweep dashboard: phase across all spacing configurations](../../../figures/spacing_sweep_f10mm/dashboard_phase_spacing.png)

> [!note] Phase Dashboard 해석
> **결론**: spacing 증가와 함께 learned phase가 난류 추종 패턴에서 더 매끈한 동심원형 correction 패턴으로 이동한다.
> **인사이트**: plateau 상단 spacing은 단순히 metric 최대점이 아니라, 해석 가능한 optical function이 가장 선명해지는 지점이기도 하다.

![Spacing-sweep dashboard: radial and centerline beam profiles across spacing configurations](../../../figures/spacing_sweep_f10mm/dashboard_profiles_spacing.png)

> [!note] Beam Profiles Dashboard 해석
> **결론**: radial/centerline profile은 6mm가 target에 가장 가깝고, 큰 spacing으로 갈수록 wing leakage가 남는다.
> **인사이트**: 2D overlap 수치만이 아니라 1D profile이 실제 receiver-friendly beam shape를 판단하는 데 중요하다.

#### B.7.2 Supplementary spacing-sweep figures

![Spacing-sweep epoch curves for seven spacing configurations](../../../figures/spacing_sweep_f10mm/fig1_epoch_curves.png)

> [!note] Epoch Curves 해석
> **결론**: 6mm 부근 config가 비교적 빠르고 안정적으로 높은 co에 도달한다.
> **인사이트**: 좋은 geometry는 최종 성능뿐 아니라 학습 conditioning도 개선한다.

![Spacing-sweep test metrics bar chart for CO, IO, and phase RMSE](../../../figures/spacing_sweep_f10mm/fig2_test_metrics.png)

> [!note] Test Metrics Bar Chart 해석
> **결론**: metric bar chart는 co, io, phase RMSE의 최적점이 서로 다름을 한눈에 보여준다.
> **인사이트**: 단일 scalar score로 모델을 고르면 어떤 응용 목표는 반드시 잃게 된다.

![Spacing-sweep full-field comparison across seven configurations](../../../figures/spacing_sweep_f10mm/fig3_field_full_comparison.png)

> [!note] Full-Field Comparison 해석
> **결론**: full-field 비교에서는 spacing이 작을수록 turbulent input 잔재가 크고, 6mm 이후엔 correction 구조가 분명해진다.
> **인사이트**: output intensity만이 아니라 residual field texture가 "중간 plane 다양성"의 효과를 드러낸다.

![Spacing-sweep center-crop zoom comparison](../../../figures/spacing_sweep_f10mm/fig4_field_zoom_comparison.png)

> [!note] Zoom Comparison 해석
> **결론**: center crop에서는 6mm가 central lobe shape와 주변 ring suppression의 균형이 가장 좋다.
> **인사이트**: 전체 field에서 비슷해 보이는 결과도 receiver core 근처를 확대하면 차이가 분명하다.

![Spacing-sweep field profiles: radial and centerline summaries](../../../figures/spacing_sweep_f10mm/fig5_field_profiles.png)

> [!note] Field Profiles 해석
> **결론**: field profile은 6mm가 target radius와 가장 잘 맞고, 큰 spacing은 peak는 유지해도 skirt가 길어진다.
> **인사이트**: "집중"과 "형상 보존"은 같은 말이 아니며, profile-based 해석이 이를 분리해준다.

![Spacing-sweep learned phase masks across spacing configurations](../../../figures/spacing_sweep_f10mm/fig6_phase_masks.png)

> [!note] Phase Masks 해석
> **결론**: spacing이 충분히 커지면 각 layer phase mask가 더 구조화되고 역할 분담이 보인다.
> **인사이트**: zero-spacing이 단순히 성능이 낮은 이유는 mask 개수가 적어서가 아니라, 서로 다른 plane을 볼 기회를 잃기 때문이다.

![Spacing-sweep Fresnel-number analysis versus spacing](../../../figures/spacing_sweep_f10mm/fig7_fresnel_analysis.png)

> [!note] Fresnel Analysis 해석
> **결론**: Fresnel-number 관점에서도 너무 작은 spacing은 mixing이 부족하고, 너무 큰 spacing은 추가 이득이 제한적이다.
> **인사이트**: empirical optimum이 diffraction-length heuristic과 대체로 정렬되지만, 그 자체를 법칙으로 일반화하면 안 된다.

![Spacing-sweep phase masks shown in the [0, 2pi] wrapped view](../../../figures/spacing_sweep_f10mm/fig8_phase_masks_0_2pi.png)

> [!note] Phase Masks [0, 2π] 해석
> **결론**: wrap된 view에서도 6mm 이후 correction 패턴이 안정적으로 유지되며, fabrication-view 해석이 가능해진다.
> **인사이트**: optimizer range와 fabrication range를 분리해서 보면 "학습은 4π, 구현은 mod 2π" 전략이 왜 유효한지 시각적으로 이해된다.

### B.8. 최종 결론

1. **`f=10mm` setting은 이 워크플로우에서 더 보수적인 sampling / fabrication proxy를 제공하며, 현재 비교 조건에선 더 좋은 성능을 보였다.**
2. **6mm (`z/z_diff,10px=0.52`)는 관찰된 plateau의 상단부였다.** CO=0.316 (+65% vs baseline)으로 전 구간 최고지만, 12mm와의 차이는 작다.
3. **`z/z_diff,10px ≈ 0.5~4.3` 구간에서 CO plateau (0.30~0.32)가 경험적으로 관찰됐다.** 다만 실제 beam의 일반적 허용오차 법칙으로 읽으면 과하다.
4. **IO는 전 구간에서 0.88 이상 유지됐다.** 현재 비교 조건에선 f=1mm보다 CO↔IO trade-off가 완화된 것으로 보인다.
5. **Phase RMSE는 spacing 증가에 따라 계속 감소** (1.749→1.285 rad). 하지만 CO는 6mm 이후 더 오르지 않음 → phase RMSE 감소가 반드시 CO 개선을 의미하지 않는다. Phase correction의 spatial 분포가 중요하다.
6. **`Peak Proxy`는 spacing 증가에 따라 증가** (2.55→2.76). 이는 target-relative peak concentration 증가를 뜻하지만, 고전적 Strehl ratio와 동일한 의미는 아니다.

> **현재 균형 후보**: `f=10mm`, `spacing=6mm`, `complex-loss spacing sweep`, `tanh_2pi`. 다만 CO 최우선 가설 후보로는 `Hybrid@50mm`도 함께 유지해야 한다.

---

## Appendix C. Loss Ablation & Curriculum 실험 — Trade-off 우회 시도 (Sweep 06~07)

> 2026-03-24 추가. Appendix A에서 관찰된 empirical co↔io trade-off를 우회할 수 있는지 실험.

### C.1. 동기

Appendix A에서 밝힌 핵심 딜레마:

| 응용 | 필요 metric | 현재 최선 | 난류 입력 | 실질 효과 |
|------|:---------:|:--------:|:--------:|:--------:|
| Direct detection | io | 0.933 | 0.973 (이미 충분) | D2NN 불필요 |
| Coherent detection | co | 0.270 | 0.191 (+41%) | 개선은 되나 beam 파괴 |

**질문: amplitude_mse 제거 또는 loss scheduling으로 trade-off를 완화할 수 있는가?**

### C.2. 전략 1 — Loss Weight Ablation (Sweep 06)

amplitude_mse가 beam 파괴의 주범이라는 가설 검증. 고정 조건: tanh_2pi, spacing=1mm, f=1mm, 30 epochs.

| Config | co weight | amp_mse | io weight | br weight | 의도 |
|--------|:-:|:-:|:-:|:-:|------|
| co_only | 1.0 | 0 | 0 | 0 | amp 완전 제거 |
| co_amp01 | 1.0 | 0.1 | 0 | 0 | amp 약하게 |
| co_io | 1.0 | 0 | 0.5 | 0 | amp→io 대체 |
| co_io_br | 1.0 | 0 | 0.25 | 0.25 | amp→irradiance terms |

#### 결과

| Config | co | pr [rad] | io | amp RMSE | 판정 |
|--------|:--:|:--------:|:--:|:--------:|:----:|
| **기존 co+amp0.5** (sweep 02) | 0.270 | 0.359 | 0.378 | 0.176 | 기준 |
| co_only (amp 제거) | 0.270 | 0.359 | 0.378 | 0.176 | **동일** |
| co_amp01 (amp 약하게) | 0.270 | 0.359 | 0.378 | 0.176 | **동일** |
| **co+io:0.5** | 0.126 | 1.540 | **0.910** | 0.164 | io 회복, co 하락 |
| co+io:0.25+br:0.25 | 0.231 | 1.191 | 0.601 | 0.172 | 중간 절충 |

#### 해석

**① 이 setting에서는 amplitude_mse 유무가 최종 결과를 바꾸지 않았다.**
co_only와 co+amp0.5가 완전히 동일했다. 이는 이 실험 조건에서 `amplitude_mse`가 상대적으로 약하게 작동했음을 뜻하지만, gradient가 일반적으로 vanishing이라는 결론까지는 주지 않는다.

**② co+io가 sweep 05의 combo3와 동일한 패턴.**
io term이 beam shape을 보존하지만, co term과 직접 경쟁하여 co가 0.126으로 하락. Appendix A의 trade-off가 loss weight 조정으로는 우회 불가능함을 재확인.

### C.3. 전략 2 — Curriculum Learning (Sweep 07)

Irradiance loss로 beam shape을 먼저 학습 → complex loss로 phase correction 추가. "좋은 초기점에서 출발하면 trade-off를 줄일 수 있는가?"

| Config | Phase 1 (irradiance) | Phase 2 (complex) | 전환 방식 |
|--------|:--:|:--:|------|
| cur_10_20 | epoch 0-10 | epoch 10-30 | 즉시 전환 |
| cur_15_15 | epoch 0-15 | epoch 15-30 | 즉시 전환 |
| cur_20_10 | epoch 0-20 | epoch 20-30 | 즉시 전환 |
| cur_blend | epoch 0-30 | epoch 0-30 | α 선형: 1→0 |

#### 결과

| Config | co | pr [rad] | io | amp RMSE | 판정 |
|--------|:--:|:--------:|:--:|:--------:|:----:|
| **irr→co (10/20)** | 0.262 | 0.390 | 0.370 | 0.175 | io 소실 |
| **irr→co (15/15)** | 0.254 | 0.440 | 0.369 | 0.175 | io 소실 |
| **irr→co (20/10)** | 0.248 | 0.511 | 0.369 | 0.175 | io 소실 |
| **linear blend** | 0.257 | 0.405 | 0.384 | 0.175 | io 소실 |

#### 해석

**③ Curriculum은 이 sweep들에서는 실패 — complex loss 전환 이후 io가 빠르게 추락.**

cur_15_15의 학습 궤적이 이를 선명하게 보여준다:

```
ep 15 [IRR] → io=0.925, co=0.005    (beam shape 확보)
ep 20 [CO]  → io=0.862, co=0.034    (beam 파괴 시작)
ep 25 [CO]  → io=0.373, co=0.222    (io 완전 추락)
ep 29 [CO]  → io=0.379, co=0.259    (수렴)
```

**irradiance로 좋은 beam shape를 먼저 잡아도, 이 설정에서는 complex objective가 수 epoch 안에 beam shape를 희생하는 방향으로 끌고 갔다.** 이것은 “좋은 초기점”만의 문제가 아니라, 현재 objective 조합에서 co를 올리는 trajectory가 io를 함께 떨어뜨렸음을 시사한다.

**④ 전환 시점이 늦을수록 co가 더 낮다 (0.262→0.248).**
Complex loss 학습 시간이 짧으면 co 수렴이 불충분. 하지만 io 추락 정도는 동일 (~0.37).

**⑤ Linear blend도 동일한 결과.**
α를 서서히 줄여도 complex overlap gradient가 dominant해지는 순간 io가 급락.

### C.4. 종합 — 8가지 전략 비교 (f=1mm)

| # | Strategy | co | io | pr [rad] | co↔io trade-off 우회? |
|:-:|----------|:--:|:--:|:--------:|:----:|
| — | **난류 입력 (no D2NN)** | 0.191 | 0.973 | — | — |
| 02 | Complex (co+amp) | **0.270** | 0.378 | **0.359** | ❌ |
| 04 | Irradiance (io+br+ee) | 0.099 | **0.933** | 1.679 | ❌ |
| 05 | Hybrid (co+io+br) | 0.126 | 0.910 | 1.540 | ❌ |
| 06a | co only (no amp) | 0.270 | 0.378 | 0.359 | ❌ amp 무관 |
| 06c | co + io:0.5 | 0.126 | 0.910 | 1.540 | ❌ hybrid와 동일 |
| 06d | co + io+br | 0.231 | 0.601 | 1.191 | ❌ 중간 |
| 07a~c | Curriculum (3종) | 0.248~0.262 | 0.369~0.370 | 0.390~0.511 | ❌ io 즉시 추락 |
| 07d | Linear blend | 0.257 | 0.384 | 0.405 | ❌ |

### C.5. Figure 삽입

아래는 Appendix C에서 경로로만 적어둔 Figure 11-14이다.

![Fig 11: CO-IO trade-off across all evaluated strategies](../../../runs/figures_sweep_report/fig11_co_io_tradeoff_all_strategies.png)

> [!note] Fig 11 해석
> **결론**: loss ablation, curriculum, baseline을 모두 올려도 Pareto front가 ideal region을 뚫지 못한다.
> **인사이트**: trade-off는 특정 loss 하나의 실수라기보다 현재 시스템 자유도의 상한을 보여준다.

![Fig 12: Curriculum trajectory and resulting trade-off](../../../runs/figures_sweep_report/fig12_curriculum_trajectory_and_tradeoff.png)

> [!note] Fig 12 해석
> **결론**: irradiance-pretraining으로 잠시 좋은 io를 확보해도 complex phase로 넘어가는 순간 io가 급락한다.
> **인사이트**: 좋은 초기화만으로는 objective conflict를 해결할 수 없고, curriculum 자체가 근본 해법이 아니다.

![Fig 13: Amplitude RMSE evidence across strategies](../../../runs/figures_sweep_report/fig13_amplitude_rmse_evidence.png)

> [!note] Fig 13 해석
> **결론**: amp RMSE는 전략 간 큰 분리력이 없는데 io는 크게 달라진다.
> **인사이트**: 현재 시스템이 잘하는 것은 pixel-wise amplitude recovery가 아니라 에너지 재배치라는 해석이 강화된다.

![Fig 14: Phase and intensity comparison across turbulent, complex, irradiance, and curriculum settings](../../../runs/figures_sweep_report/fig14_phase_intensity_comparison.png)

> [!note] Fig 14 해석
> **결론**: turbulent, complex, irradiance, curriculum을 나란히 보면 각 전략이 무엇을 보존하고 무엇을 희생하는지가 직관적으로 드러난다.
> **인사이트**: curriculum도 결국 complex trajectory로 빨려 들어가므로, 최종 morphology는 complex 계열과 더 닮는다.

### C.6. 결론

> **현재 sweep들에서는 관찰된 empirical co↔io trade-off를 loss function engineering만으로 우회하지 못했다.**

이 trade-off는 loss 설계와 시스템 자유도가 함께 만든 결과로 보인다:
- Phase correction (co↑)은 간섭 패턴 재구성을 요구
- 본 sweep들에서는 간섭 패턴 재구성이 io 하락과 함께 나타났다
- Phase-only mask는 이 두 목표를 robust하게 독립 제어하기 어렵다

**해결 방향**: loss 변경이 아닌 **시스템 아키텍처 변경**이 필요하다.

1. **f=10mm 설계** (Appendix B): io 감소 −7%로 trade-off 대폭 완화. CO=0.316 달성. → **가장 유망**
2. **Complex-valued mask** (amplitude + phase 제어): 자유도 2배 증가
3. **더 많은 layer**: phase→amplitude 변환 경로 다양화
4. **Hybrid system**: D2NN + deformable mirror

> Appendix B의 f=10mm 결과가 이미 trade-off 완화를 보여주고 있다 (CO=0.316, IO=0.901).
> **f=10mm + plateau 상단 spacing(6mm 부근)에서의 matched-control loss sweep이 다음 실험 우선순위이다.**

