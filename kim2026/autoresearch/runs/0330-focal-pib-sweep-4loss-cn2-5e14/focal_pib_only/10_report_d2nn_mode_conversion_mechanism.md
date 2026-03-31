# D2NN의 Mode Conversion 메커니즘 — 물리적 해석

## 실험 요약

| 항목 | 값 |
|------|-----|
| 시스템 | TX → 1km 대기 → 15cm 망원경 → D2NN(5층) → f=4.5mm 렌즈 → 검출기 |
| 난류 | Cn² = 5×10⁻¹⁴, D/r₀ = 5.02 |
| D2NN | 5층 위상판, 1024×1024, dx=2μm, 층간 10mm |
| 학습 | focal PIB@10μm 최적화 (초점면에서 직접 최적화) |
| 데이터 | train 4000 / val 500 / test 500 |

### 테스트 결과

|  | PIB@5μm | PIB@10μm | PIB@25μm | PIB@50μm |
|---|---------|----------|----------|----------|
| Vacuum | 8.9% | 16.5% | 90.4% | 98.5% |
| Turbulent | 6.4% | 17.2% | 83.0% | 97.8% |
| **D2NN** | **55.6%** | **81.4%** | **96.5%** | 97.9% |
| 개선 배율 | **8.65×** | **4.74×** | 1.16× | 1.00× |

---

## 1. 왜 D2NN은 파면을 "교정"할 수 없는가?

### 1.1 유니터리 정리 (Unitary Theorem)

D2NN은 위상판 + 자유공간 전파의 반복이다. 각 층은 다음 연산을 수행한다:

$$U_{k+1}(x,y) = \mathcal{P}\left[ e^{j\phi_k(x,y)} \cdot U_k(x,y) \right]$$

여기서:
- $\phi_k(x,y)$: k번째 층의 학습된 위상 마스크
- $\mathcal{P}$: 자유공간 전파 연산자 (Angular Spectrum Method)
- $U_k$: k번째 층 입사 필드

**핵심**: $e^{j\phi_k}$는 unitary (진폭 불변, 위상만 변조), $\mathcal{P}$도 unitary (에너지 보존 전파). 따라서 전체 D2NN 연산 $H$는 **unitary**이다.

#### 정리 1: Complex Overlap 보존

$$\text{CO}(HU_t, HU_v) = \frac{|\langle HU_t, HU_v \rangle|}{||HU_t|| \cdot ||HU_v||} = \frac{|\langle U_t, U_v \rangle|}{||U_t|| \cdot ||U_v||} = \text{CO}(U_t, U_v)$$

**증명**: $H$가 unitary이면 $\langle Hx, Hy \rangle = \langle x, y \rangle$ (내적 보존). 노름도 보존되므로 CO는 변하지 않는다.

**의미**: CO는 "파면이 vacuum과 얼마나 비슷한가"의 척도다. D2NN을 아무리 학습해도 이 값은 바뀌지 않는다. 즉, **D2NN은 파면 품질을 개선할 수 없다.**

#### 정리 2: L2 거리 보존

$$||HU_t - HU_v||_2 = ||U_t - U_v||_2$$

**의미**: 난류 필드와 vacuum 필드의 "차이"는 D2NN 전후로 동일하다.

### 1.2 직관적 이해

> **비유**: 루빅스 큐브를 생각하자.
>
> - 난류 = 큐브를 랜덤하게 섞은 것
> - 파면 교정 (AO) = 섞은 순서를 알고 역순으로 돌림 → 원래 상태 복원
> - D2NN (고정 위상판) = 항상 같은 순서로만 돌릴 수 있음 → 랜덤 섞임을 복원 불가
>
> D2NN은 매번 같은 동작밖에 못 하므로, 매번 다른 난류를 교정할 수 없다.

---

## 2. 그러면 D2NN은 뭘 한 건가? — Mode Conversion

### 2.1 공간 모드 분해

임의의 광학 필드는 직교 모드의 합으로 쓸 수 있다:

$$U(x,y) = \sum_{m,n} a_{mn} \cdot \psi_{mn}(x,y)$$

여기서 $\psi_{mn}$은 직교 기저 (예: Hermite-Gaussian 모드), $a_{mn}$은 각 모드의 복소 진폭.

**에너지 보존**: $\sum_{m,n} |a_{mn}|^2 = \text{총 에너지} = \text{const}$

### 2.2 난류가 하는 일

난류 없는 vacuum beam은 거의 순수한 기본 모드:

$$U_{\text{vac}} \approx a_{00} \cdot \psi_{00} \quad \text{(Gaussian beam)}$$

난류가 에너지를 고차 모드로 분산시킨다:

$$U_{\text{turb}} = \underbrace{a'_{00}}_{\text{줄어듦}} \cdot \psi_{00} + \underbrace{a'_{01} \cdot \psi_{01} + a'_{10} \cdot \psi_{10} + \cdots}_{\text{새로 생김}}$$

$|a'_{00}|^2 < |a_{00}|^2$ — 기본 모드 에너지가 줄어듦.

### 2.3 렌즈 집속과 PIB

렌즈는 각 모드를 초점면에서 다른 패턴으로 변환한다:

| 모드 | 초점면 패턴 | 10μm bucket 내 에너지 |
|------|-----------|---------------------|
| $\psi_{00}$ (기본) | 중심에 집중 (Airy disk) | **높음** |
| $\psi_{01}, \psi_{10}$ | 두 갈래로 분산 | 낮음 |
| $\psi_{11}$ | 네 갈래로 분산 | 매우 낮음 |

따라서:

$$\text{PIB} \approx |a_{00}|^2 \cdot \eta_{00} + |a_{01}|^2 \cdot \eta_{01} + \cdots$$

여기서 $\eta_{mn}$은 모드 $mn$의 bucket 내 집속 효율. $\eta_{00} \gg \eta_{01} > \eta_{11} > \cdots$

**난류가 PIB를 낮추는 이유**: $|a_{00}|^2$ (잘 집속되는 모드)가 줄고, $|a_{mn}|^2$ (안 집속되는 모드)가 늘어남.

### 2.4 D2NN의 Mode Conversion

D2NN은 unitary 연산 $H$를 적용한다. 모드 공간에서:

$$H: \quad a_{mn} \rightarrow b_{mn} = \sum_{m',n'} H_{mn,m'n'} \cdot a_{m'n'}$$

에너지 보존: $\sum |b_{mn}|^2 = \sum |a_{mn}|^2$

**핵심**: D2NN은 에너지를 만들거나 없앨 수 없지만, **모드 간에 에너지를 재배분**할 수 있다.

$$|a_{00}|^2 = 0.17 \quad \xrightarrow{D2NN} \quad |b_{00}|^2 = 0.81$$

나머지 고차 모드에서 에너지를 "빼앗아" 기본 모드로 옮긴 것이다.

> **비유**: 주머니가 여러 개 달린 조끼를 입고 있다.
> - 난류 = 동전을 여러 주머니에 흩뿌림
> - D2NN = 조끼를 뒤집어 흔들면 특정 주머니에 동전이 모임
> - 총 동전 수는 같지만 (에너지 보존), 분포가 바뀜

### 2.5 왜 이것이 가능한가?

"고정 위상판이 랜덤 난류에 대해 어떻게 통계적으로 유리할 수 있는가?"

답: D2NN은 **통계적 앙상블**에 대해 최적화된다. 4000개의 서로 다른 난류 패턴을 학습하면서, **평균적으로** 기본 모드 에너지를 최대화하는 위상 변환을 찾는다.

수학적으로, 학습 목표는:

$$\min_{\{\phi_k\}} \mathbb{E}_{\text{turb}} \left[ 1 - \text{PIB}(H_\phi U_\text{turb}) \right]$$

이것은 특정 난류 하나를 교정하는 것이 아니라, **모든 가능한 난류에 대한 기대값**을 최적화한다.

---

## 3. WF RMS가 악화되는 이유

### 3.1 측정 방법

```
WF RMS = sqrt( Σ w(x,y) · [φ_residual(x,y)]² )
```

여기서:
- $\phi_{\text{residual}} = \angle(U_{\text{D2NN}}) - \angle(U_{\text{vac,D2NN}})$ (piston 제거 후)
- $w(x,y) = |U_{\text{vac,D2NN}}|^2 / \sum|U_{\text{vac,D2NN}}|^2$ (vacuum 세기 가중)

### 3.2 WF RMS ≠ L2 거리

유니터리 정리가 보존하는 것은 **L2 거리**이다:

$$||HU_t - HU_v||_2^2 = \sum_{x,y} |U_t^H(x,y) - U_v^H(x,y)|^2$$

이것은 진폭과 위상을 모두 포함한 **복소 필드의 차이**다.

반면 WF RMS는:
- **위상만** 본다 (진폭 차이 무시)
- **vacuum 세기로 가중**한다 (D2NN 통과 후 vacuum의 세기 분포가 바뀜)

따라서 **WF RMS는 유니터리 정리의 보존량이 아니다.** 변할 수 있으며, 변한다.

### 3.3 실험 결과

| | WF RMS | 표준편차 |
|---|--------|---------|
| Turbulent (no D2NN) | 349.5 nm | ±75.3 nm |
| D2NN | 423.4 nm | ±15.3 nm |

- **평균 증가** (349→423): D2NN이 위상을 의도적으로 재배열. "파면을 평탄하게" 하는 것이 아니라, "렌즈에 유리한 형태로" 재구성.
- **표준편차 감소** (75→15): 다양한 난류 입력에 대해 **거의 동일한 출력 위상 패턴**을 만듦. D2NN이 입력을 하나의 "최적 출력 모드"로 수렴시키는 증거.

### 3.4 직관적 비유

> 기타 소리를 마이크로 녹음한다고 하자.
>
> - **파면 교정 (AO)**: 방의 반향을 제거해서 원래 기타 소리를 복원 → WF RMS 감소
> - **D2NN (mode conversion)**: 반향은 그대로인데, 이퀄라이저를 고정 세팅으로 걸어서 특정 주파수 대역만 증폭 → 전체 음질(WF RMS)은 나빠질 수 있지만, 마이크에 잡히는 신호(PIB)는 증가

---

## 4. D2NN 출력면에서 관찰되는 현상

### 4.1 Residual Phase의 동심원 패턴

D2NN 출력의 residual phase에서 **규칙적인 동심원 패턴**이 관찰된다.

이것은 D2NN이 출력에 **2차 위상 (quadratic phase)**을 추가하고 있다는 뜻:

$$\phi_{\text{added}}(r) \approx \frac{\pi r^2}{\lambda f_{\text{eff}}}$$

이것은 사실상 **보조 렌즈** 역할이다. f=4.5mm 렌즈에 추가적인 포커싱 파워를 더해서, 10μm bucket에 에너지가 더 집중되도록 한다.

### 4.2 Irradiance 재분배

D2NN 출력에서 세기 분포가 한쪽으로 편중되어 있다. 이것은 D2NN이 **공간 필터링**을 수행하는 것:

- 넓게 퍼진 에너지 → 5층 회절을 거치며 특정 영역으로 집중
- 집중된 에너지가 렌즈를 통과하면 → 더 작은 focal spot

### 4.3 수치 요약

| 관측 | 물리적 의미 |
|------|-----------|
| CO 보존 (0.259→0.098) | ⚠ CO가 감소한 것처럼 보이지만, 이것은 CO(HU_t, U_v)이지 CO(HU_t, HU_v)가 아님. 정리의 보존량은 후자. |
| WF RMS 증가 (349→423nm) | 위상 재배열 — 파면 교정이 아닌 mode conversion |
| WF RMS 표준편차 감소 (75→15nm) | 다양한 입력 → 거의 동일한 출력 (통계적 수렴) |
| Residual phase 동심원 | 보조 렌즈 효과 (추가 포커싱 파워) |
| PIB@10μm: 17%→81% | 기본 모드 에너지 증가 (mode conversion 성공) |

---

## 5. Strehl Ratio 계산의 문제점과 올바른 해석

### 5.1 Strehl ratio의 정의와 상한

Strehl ratio는 다음과 같이 정의된다:

$$S = \frac{I(0)_{\text{aberrated}}}{I(0)_{\text{diffraction-limited}}} \leq 1$$

**S ≤ 1임은 물리적으로 보장된다.** Fraunhofer 평면에서 복소장은 pupil field의 푸리에 적분이므로:

$$E(\boldsymbol{f}) \propto \int_\Omega U(\boldsymbol{\rho})\,e^{-i2\pi \boldsymbol{f}\cdot\boldsymbol{\rho}}\,d\boldsymbol{\rho}$$

Cauchy-Schwarz 부등식에 의해:

$$|E(\boldsymbol{f})| \leq \int_\Omega |U(\boldsymbol{\rho})|\,d\boldsymbol{\rho} \leq \sqrt{|\Omega| \int_\Omega |U(\boldsymbol{\rho})|^2\,d\boldsymbol{\rho}}$$

등호는 위상이 평탄하고 진폭이 균일할 때만 성립한다. 따라서 **같은 aperture support와 같은 total flux라면, 어떤 위치의 speckle peak도 diffraction-limited peak보다 클 수 없다.** 이것은 약한 난류든 강한 난류(D/r₀=5)든 동일하다.

### 5.2 시뮬레이션에서 S > 1이 나온 원인: 수치적 언더샘플링

본 시뮬레이션에서 `strehl_ratio()` 함수가 S > 1을 반환한 것은 물리적 현상이 아니라 **수치적 아티팩트**이다.

**원인: PSF 언더샘플링**

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| dx_focal | 3.406 μm | focal plane pixel pitch |
| Airy first zero | 4.255 μm | 1.22λf/D |
| **Airy radius in pixels** | **1.25 px** | Nyquist 조건(≥2 px) **위반** |

Airy disk의 first zero가 1.25 pixel — 심각한 언더샘플링. 이산 FFT가 PSF의 진짜 peak을 포착하지 못한다.

실측 결과 (vacuum focal, energy-normalized):
```
3×3 around center:
  0.01046  0.00992  0.01046
  0.00992  0.00733  0.00992     ← 중심(512,512)이 오히려 가장 낮음!
  0.01046  0.00992  0.01046
```

**진짜 peak은 pixel 사이(sub-pixel 위치)에 있는데**, 이산 격자의 중심 pixel은 peak에서 벗어나 있다. 반면 turbulent speckle은 무작위 위치에서 발생하므로 우연히 pixel center에 가까이 떨어질 수 있다 → 샘플된 peak이 상대적으로 높아 보인다.

이것은 정확히 다음에 해당한다:
- ideal PSF를 analytic peak이 아니라 **언더샘플된 FFT peak으로 잡은 경우**
- turbulent frame의 local max와 reference의 샘플링 조건이 다른 경우

### 5.3 D2NN 시스템에서 Strehl 적용의 추가적 어려움

PSF 언더샘플링 문제를 해결하더라도 (zero-padding, sinc interpolation 등), D2NN 시스템에서 Strehl을 올바르게 정의하려면 추가적인 난제가 있다:

1. **D2NN이 effective aperture illumination을 바꿈**: 학습된 D2NN은 vacuum beam의 진폭/위상 분포를 근본적으로 재구성한다. 실측: D2NN(vac) focal peak이 zero-phase D2NN(vac) 대비 42배 집중. "같은 시스템에서 수차만 다른" 비교가 어렵다.

2. **Space-variant system**: D2NN은 고정 위상 마스크이므로 입사 위치에 따라 출력이 달라지는 space-variant system이다. 단일 PSF로 시스템을 설명할 수 없으므로 전통적 Strehl 정의가 직접 적용되지 않는다.

3. **어떤 reference를 쓸 것인가?**: D2NN(turb) vs D2NN(vac)? vs zero-phase D2NN(vac)? vs 직접 집속? 각각 다른 값을 주며, 합의된 정의가 없다.

### 5.4 올바른 해석

Strehl ratio 자체는 의미 있는 메트릭이다. 강한 난류(D/r₀=5)에서도 AO 시스템은 long-exposure Strehl로 보정 성능을 평가하고, lucky imaging에서는 단노출 Strehl로 frame selection을 수행한다.

초기 시뮬레이션에서 S=28이 나온 원인:
1. PSF 언더샘플링 (Airy = 1.25 px) → reference peak 과소평가
2. Vacuum wavefront curvature (R=2.2m) → vacuum 자체가 diffraction-limited가 아님
3. amax 비교 (on-axis가 아닌 전체 최대값 비교)

### 5.5 수정된 Strehl 계산 (구현 완료)

`strehl_ratio_correct()` 함수로 수정:
- 4× zero-padding (Airy = 5 px, Nyquist 만족)
- Flat-phase reference (vacuum 진폭 + 위상 0)
- Cauchy-Schwarz 검증: flat-phase uniform aperture → S = 1.000000, 0.5 rad RMS → S = 0.778 ≈ Maréchal 0.779

**수정된 결과:**

| | Strehl (correct) | PIB@10μm |
|---|---|---|
| Vacuum (curved WF) | 0.016 | 0.165 |
| Turbulent | 0.047 | 0.172 |
| D2NN (focal PIB) | **0.546** | **0.813** |

D2NN이 Strehl을 0.047 → 0.546으로 **11.6배** 개선. 이것은 D2NN의 mode conversion이 wavefront curvature도 부분적으로 보상함을 의미.

참고: Vacuum Strehl = 0.016은 입사 beam 자체의 wavefront curvature (1km 전파 후 R=2.2m) 때문. f=4.5mm 렌즈가 이 curvature를 보상하도록 설계되지 않았음.

**유효 메트릭 우선순위**: PIB > Strehl. PIB는 reference 정의에 무관하고, pixel 샘플링에 덜 민감.

---

## 6. 핵심 결론

### 할 수 있는 것과 할 수 없는 것

| | Adaptive Optics (AO) | Static D2NN |
|---|---|---|
| 파면 교정 | ✅ (센서+액추에이터) | ❌ (고정 마스크) |
| CO 개선 | ✅ | ❌ (유니터리 정리) |
| WF RMS 개선 | ✅ | ❌ |
| Strehl 개선 | ✅ (파면 교정 → Strehl ↑) | ❌ (Strehl 정의 자체 부적합, §5 참조) |
| Mode conversion | ✅ | ✅ |
| PIB 개선 | ✅ (파면 교정 경유) | ✅ (mode conversion 경유) |
| 유효 메트릭 | CO, WF RMS, Strehl, PIB | **PIB만 유효** |
| 전력 소모 | 높음 | **없음** (passive) |
| 지연 시간 | ms급 (센서 → 계산 → 구동) | **없음** (광속) |

### 한 문장 요약

> **D2NN은 난류를 "교정"하지 않는다. 대신, 5층 회절 위상판을 통해 광학 모드 간 에너지를 재배분하여, 초점면에서 작은 bucket 안에 더 많은 에너지가 집중되도록 한다. 이것은 파면 품질(WF RMS)을 희생하고 집속 효율(PIB)을 얻는 트레이드오프이다.**

---

## 부록: 파일 목록 (생성 순서)

| # | 파일 | 내용 |
|---|------|------|
| 01 | `01_model_checkpoint.pt` | 학습된 D2NN 모델 가중치 |
| 02 | `02_training_epoch_log.txt` | 전 epoch loss/lr 기록 |
| 03 | `03_learned_phase_masks_5layers.npy` | 5층 위상 마스크 (1024×1024 × 5) |
| 04 | `04_training_results_full.json` | 학습 완료 후 전체 결과 (메트릭, 히스토리) |
| 05 | `05_test_evaluation_pib_at_4radii.json` | 테스트셋 PIB@[5,10,25,50]μm 수치 |
| 06 | `06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png` | **Fig 1**: 초점면 3종 비교 (irradiance, log, 1D profile) |
| 07 | `07_fig2_pib_bar_chart_5um_10um_25um_50um.png` | **Fig 2**: PIB bar chart (4개 bucket 반지름) |
| 08 | `08_fig3_d2nn_output_plane_irradiance_phase_residual.png` | **Fig 3**: D2NN 출력면 분석 (irradiance, phase, residual phase, 1D) |
| 09 | `09_fig4_wavefront_rms_distribution_50samples.png` | **Fig 4**: WF RMS 히스토그램 (turbulent vs D2NN, 50 samples) |
| 10 | `10_report_d2nn_mode_conversion_mechanism.md` | 본 리포트 |
