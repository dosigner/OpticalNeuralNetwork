# 정적 회절 신경망의 대기 난류 보상 한계와 비선형 검출을 통한 집속효율 개선

## Static Diffractive Deep Neural Network: Fundamental Limits in Atmospheric Turbulence Compensation and Focusing Efficiency Enhancement via Nonlinear Detection

---

## 1. 서론 (Introduction)

자유공간 광통신(FSO)은 높은 대역폭과 면허 불필요성 등의 장점으로 차세대 통신 기술로 주목받고 있다 [1]. 그러나 대기 난류에 의한 빔 왜곡—파면 오차(wavefront error), 신틸레이션(scintillation), 빔 방황(beam wander)—은 FSO 링크의 성능을 심각하게 저하시킨다 [2].

적응광학(Adaptive Optics, AO)은 실시간 파면 센싱과 변형 거울을 통해 난류를 보상하는 표준 기술이지만, 높은 비용, 복잡한 제어 시스템, 제한된 보정 대역폭 등의 한계가 있다 [3]. 이러한 배경에서 **수동 광학 소자만으로 빔 품질을 개선**할 수 있다면 FSO 시스템의 실용화에 큰 기여를 할 수 있다.

회절 심층 신경망(Diffractive Deep Neural Network, D2NN)은 다층 위상 마스크와 자유공간 회절의 연쇄(cascade)로 구성된 수동 광학 프로세서로, Lin 등 [4]에 의해 처음 제안되었다. Tao 등 [5]은 D2NN을 빔 정화(beam cleanup)에 적용하여 결정론적(deterministic) 수차의 보정 가능성을 보였다.

그러나 기존 연구들은 **고정된(deterministic) 수차 패턴**을 대상으로 하였으며, 대기 난류와 같이 **매 순간 다른 랜덤 파면 왜곡**에 대한 정적 D2NN의 성능은 체계적으로 검증되지 않았다. 본 연구에서는 다음 세 가지 핵심 기여를 제시한다:

1. **유니터리 불변성 정리(Theorem 1–4)**: 정적 D2NN이 대기 난류의 복소 중첩(Complex Overlap, CO) 및 파면 오차(WF RMS)를 개선할 수 없음을 수학적으로 증명한다.

2. **비선형 검출 채널**: 광검출기의 제곱 비선형성(|E|²)이 유니터리 불변성을 깨트리며, 집속효율(Power-in-Bucket, PIB)과 같은 세기(intensity) 기반 지표만이 정적 D2NN에 의해 개선될 수 있음을 보인다.

3. **실험적 검증**: Cn² = 5×10⁻¹⁴ m⁻²/³, D/r₀ = 5.02 조건에서 PIB 손실함수를 적용하여 집속효율을 0.3%에서 83.4%로 247배 개선하는 실험 결과를 제시한다.

---

## 2. 이론 (Theory)

### 2.1 시스템 모델

본 연구의 FSO 시스템은 다음과 같이 구성된다:

- **송신기**: 파장 λ = 1.55 μm, 전발산각(full divergence) 0.3 mrad의 가우시안 빔
- **대기 채널**: 수평 경로 L = 1 km, Kolmogorov 난류 모델, 구조 상수 Cn² = 5×10⁻¹⁴ m⁻²/³
- **수신기**: 구경 D = 15 cm 망원경 → 75:1 빔 축소기 → 2.048 mm D2NN 입력 윈도우

대기 파라미터:

$$r_0 = \left(0.423 k^2 C_n^2 L\right)^{-3/5} = 2.99 \text{ cm}$$

$$D/r_0 = 15 \text{ cm} / 2.99 \text{ cm} = 5.02$$

이는 중등도 난류(moderate turbulence) 영역에 해당한다.

빔 전파는 Schmidt의 각스펙트럼 다중 위상스크린 방법(angular spectrum multi-phase-screen method) [6]으로 시뮬레이션하였다. 격자 크기 N = 4096, 위상스크린 18장을 사용하였으며, 전파 후 중앙 1024 × 1024 픽셀을 절단(crop)하여 15 cm 망원경 구경에 대응하는 수신 필드를 얻었다.

### 2.2 D2NN의 수학적 정의

D2NN은 K개의 위상 마스크 Mₖ와 자유공간 전파 연산자 Pₖ의 연쇄로 정의된다:

$$H = P_K \cdot M_K \cdot P_{K-1} \cdot M_{K-1} \cdots P_1 \cdot M_1$$

여기서 각 위상 마스크는:

$$M_k(\mathbf{x}) = e^{i\phi_k(\mathbf{x})}$$

이며, 자유공간 전파 연산자 Pₖ는 각스펙트럼법(angular spectrum method)에 의한 프레넬(Fresnel) 전파를 수행한다:

$$P_k[U](\mathbf{x}) = \mathcal{F}^{-1}\left\{\mathcal{F}\{U\}(\mathbf{f}) \cdot \exp\left(i 2\pi z_k \sqrt{\lambda^{-2} - |\mathbf{f}|^2}\right)\right\}(\mathbf{x})$$

위상 마스크 Mₖ는 유니터리(unitary)이다: $M_k^\dagger M_k = I$ (위상만 변조하므로 에너지 보존). 또한 자유공간 전파 Pₖ도 Parseval 정리에 의해 유니터리이다: $P_k^\dagger P_k = I$.

따라서 **D2NN 전체 연산자 H는 유니터리**이다:

$$\boxed{H^\dagger H = I}$$

이것이 본 논문의 모든 정리의 출발점이다.

### 2.3 정리 1: 내적 보존 (Inner Product Preservation)

**정리 1.** 유니터리 연산자 H에 대해, 임의의 두 필드 U₁, U₂의 내적은 보존된다:

$$\langle HU_1, HU_2 \rangle = \langle U_1, H^\dagger H U_2 \rangle = \langle U_1, U_2 \rangle$$

**따름정리 1a (CO 보존).** 복소 중첩(Complex Overlap)의 정의:

$$\text{CO}(U_1, U_2) = \frac{|\langle U_1, U_2 \rangle|}{\|U_1\| \cdot \|U_2\|}$$

유니터리 H에 의해 분자 $|\langle \cdot, \cdot \rangle|$와 분모 $\|\cdot\|$가 모두 보존되므로:

$$\boxed{\text{CO}(HU_1, HU_2) = \text{CO}(U_1, U_2) \quad \forall H: H^\dagger H = I}$$

**물리적 의미**: 어떤 정적 위상 마스크 조합을 사용하든, 난류 빔과 진공 빔을 **동일한 D2NN**에 통과시키면 두 빔 사이의 CO는 변하지 않는다. D2NN은 두 빔 사이의 "필드 수준 유사도"를 바꿀 수 없다.

### 2.4 정리 2: 거리 보존 (Metric Distance Preservation)

**정리 2.** 유니터리 연산자 H는 L2 거리를 보존한다:

$$\|HU_1 - HU_2\|^2 = \langle H(U_1-U_2), H(U_1-U_2) \rangle = \|U_1-U_2\|^2$$

$$\boxed{\|HU_\text{aberr} - HU_\text{vac}\|_2 = \|U_\text{aberr} - U_\text{vac}\|_2}$$

**따름정리 2a (WF RMS 근사 보존).** 소수차 근사(small aberration approximation, $|\Delta\phi| \ll 1$)에서:

$$U_\text{aberr} \approx U_\text{vac} \cdot e^{i\Delta\phi} \approx U_\text{vac}(1 + i\Delta\phi)$$

$$|U_\text{aberr} - U_\text{vac}|^2 \approx |U_\text{vac}|^2 |\Delta\phi|^2 = I_\text{vac} \cdot |\Delta\phi|^2$$

양변을 적분하면:

$$\int |U_\text{aberr} - U_\text{vac}|^2 \, d\mathbf{x} \approx \int I_\text{vac} \cdot |\Delta\phi|^2 \, d\mathbf{x} \propto \sigma_\phi^2$$

정리 2에 의해 좌변이 H에 의해 보존되므로, 세기 가중 파면 오차(WF RMS) 또한 근사적으로 보존된다:

$$\boxed{\sigma_\phi(HU_\text{aberr}, HU_\text{vac}) \approx \sigma_\phi(U_\text{aberr}, U_\text{vac})}$$

### 2.5 정리 3: 랜덤 난류에 대한 기대 CO 불변

**정리 3.** 난류 필드 $U_\text{turb} = U_\text{vac} \cdot e^{i\phi_\text{turb}}$에서 $\phi_\text{turb}$가 영평균(zero-mean) 랜덤 과정일 때:

$$E_\phi\left[\text{CO}(HU_\text{turb}, HU_\text{vac})\right] = E_\phi\left[\text{CO}(U_\text{turb}, U_\text{vac})\right] \quad \forall H: H^\dagger H = I$$

**증명**: 정리 1에 의해 각 실현(realization)에서 $\text{CO}(HU_\text{turb}, HU_\text{vac}) = \text{CO}(U_\text{turb}, U_\text{vac})$이므로, 기댓값도 동일하다.

**물리적 의미**: 학습 데이터를 아무리 많이 사용하고, 어떤 손실함수로 학습하더라도, 정적 D2NN이 달성할 수 있는 **기대 CO의 상한은 입력 그 자체의 기대 CO**와 동일하다. 이는 구조적 한계이며, 학습 알고리즘의 문제가 아니다.

### 2.6 비선형 검출 채널: 유니터리 불변성의 파괴

광검출기는 입사 필드의 세기를 측정한다:

$$I(\mathbf{x}) = |U(\mathbf{x})|^2$$

이것은 필드의 **이차 형식(quadratic form)**으로, 선형이 아닌 **비선형 연산**이다.

집속효율(Power-in-Bucket, PIB)은:

$$\text{PIB} = \frac{\int_B |U(\mathbf{x})|^2 \, d\mathbf{x}}{\int |U(\mathbf{x})|^2 \, d\mathbf{x}}$$

유니터리 H는 총 에너지를 보존한다: $\int |HU|^2 = \int |U|^2$ (분모 불변). 그러나 **국소적 세기 분포**는 변경할 수 있다: $|HU(\mathbf{x})|^2 \neq |U(\mathbf{x})|^2$ (일반적으로).

따라서:

$$\boxed{\text{PIB}(HU) \neq \text{PIB}(U) \quad \text{(일반적으로)}}$$

**반례 (Counterexample).** 2-픽셀 시스템에서:

$$a = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$$

Bucket = 픽셀 0: PIB(a) = 0.5, PIB(Ha) = 1.0. 동일한 유니터리 변환이 PIB를 2배로 증가시켰다.

이는 정리 1–3과 대비된다: 내적 기반 지표(CO, WF RMS)는 유니터리 변환에 불변이지만, **이차 형식 기반 지표(PIB, Strehl)는 유니터리 변환에 의해 변할 수 있다**.

**핵심 통찰**: 광검출의 비선형성($|E|^2$)이 유니터리 불변성을 파괴하며, 이것이 정적 D2NN이 빔 품질을 개선할 수 있는 **유일한 물리적 채널**이다.

---

## 3. 실험 방법 (Methods)

### 3.1 D2NN 구조

BeamCleanupD2NN 모델을 사용하였다:
- 층 수: K = 5
- 층간 간격: 10 mm (각스펙트럼 자유공간 전파)
- 격자: 1024 × 1024, 픽셀 크기 dx = 2.0 μm
- 윈도우: 2.048 mm × 2.048 mm
- 위상 제약: unconstrained ([-∞, +∞])
- 학습 가능 파라미터: 5 × 1024² = 5,242,880개

### 3.2 데이터 생성

Schmidt 각스펙트럼 방법으로 500개의 난류 실현(realization)을 생성하였다:
- 전파 격자: N = 4096, 위상스크린 18장
- 전파 후 중앙 1024 × 1024 절단 (= 15 cm 망원경 구경)
- 75:1 빔 축소기를 좌표 재해석으로 모델링
- 분할: 학습 400 / 검증 50 / 시험 50

### 3.3 학습 전략

두 단계로 나누어 실험하였다:

**Phase A — 필드 기반 손실함수 (CO 중심)**
- 손실함수: CO, CO+amplitude, CO+phasor, CO+full-field-phase, ROI
- 에폭: 100, 학습률: 5×10⁻⁴, 배치: 2
- 목적: 정리 1–3의 실험적 검증

**Phase B — 세기 기반 손실함수 (PIB 중심)**
- 손실함수: PIB, Strehl, intensity overlap, CO+PIB hybrid
- 에폭: 200, 학습률: 1×10⁻³ (코사인 감쇠)
- TV 정규화: 가중치 0.05 (위상 마스크 평활도 강제)
- 목적: 비선형 metric 개선 가능성 검증

### 3.4 결정론적 수차 검증

D2NN 구현의 정확성을 확인하기 위해, Zernike 수차(defocus Z₄, coma Z₇, astigmatism Z₅)에 대한 결정론적 교정 실험을 수행하였다:
- 단층 D2NN (전파 없음): 마스크가 직접 위상을 상쇄
- 다층 D2NN (10 mm 간격): 정리 2의 실험적 검증

### 3.5 평가 지표

지표를 두 범주로 구분한다:

| 범주 | 지표 | 정의 | 유니터리 불변? |
|------|------|------|--------------|
| 선형 (필드 기반) | CO | $\|\langle U_1, U_2\rangle\| / (\|U_1\|\|U_2\|)$ | **예** (정리 1) |
| 선형 (필드 기반) | WF RMS | $\sqrt{\int w \|\Delta\phi\|^2}$ | **예** (정리 2) |
| 비선형 (세기 기반) | PIB | $\int_B \|U\|^2 / \int \|U\|^2$ | **아니오** |
| 비선형 (세기 기반) | Strehl | $\max\|U\|^2 / \max\|U_\text{ref}\|^2$ | **아니오** |

---

## 4. 결과 (Results)

### 4.1 정리 1–2의 수치 검증: 유니터리 보존량의 직접 측정

정리 1–2가 보존한다고 주장하는 양을 직접 측정하여 검증하였다. 핵심은 **두 필드 모두에 동일한 D2NN 연산자 H를 적용**한 뒤 비교하는 것이다.

**정리 1 검증 (CO 보존).** 50개 시험 실현에 대해, 입력 평면의 $\text{CO}(U_\text{turb}, U_\text{vac})$와 D2NN 출력 평면의 $\text{CO}(HU_\text{turb}, HU_\text{vac})$를 비교하였다:

| 전략 | CO(U_t, U_v) 입력 | CO(HU_t, HU_v) D2NN | |차이| 평균 | |차이| 최대 |
|------|-------------------|---------------------|-------------|-------------|
| PIB only | 0.304446 | 0.304446 | 3.9×10⁻⁸ | 1.2×10⁻⁷ |
| CO+PIB hybrid | 0.304446 | 0.304446 | 4.5×10⁻⁸ | 1.5×10⁻⁷ |
| Strehl only | 0.304446 | 0.304446 | 6.0×10⁻⁸ | 1.8×10⁻⁷ |
| Intensity overlap | 0.304446 | 0.304446 | 1.1×10⁻⁷ | 3.3×10⁻⁷ |

모든 전략에서 CO의 차이는 10⁻⁷ 수준으로, **기계 정밀도(float32) 수준에서 정리 1이 정확히 성립**함을 확인하였다 (그림 7a). 이 결과는 학습된 위상 마스크의 형태에 무관하게 동일하다—유니터리 보존은 구조적 속성이기 때문이다.

**정리 2 검증 (L2 거리 보존).** 동일하게, $\|U_\text{turb} - U_\text{vac}\|_2$와 $\|HU_\text{turb} - HU_\text{vac}\|_2$를 비교하였다:

| 전략 | ||U_t − U_v|| 입력 | ||HU_t − HU_v|| D2NN | 상대 오차 |
|------|-------------------|---------------------|----------|
| PIB only | 24.8073 | 24.8073 | 1.2×10⁻⁶ |
| CO+PIB hybrid | 24.8073 | 24.8073 | 1.2×10⁻⁶ |
| Strehl only | 24.8073 | 24.8073 | 1.3×10⁻⁶ |
| Intensity overlap | 24.8073 | 24.8073 | 1.3×10⁻⁶ |

L2 거리의 상대 오차는 10⁻⁶ 수준으로, **정리 2가 수치적으로 정확히 성립**함을 확인하였다 (그림 7b). 이는 동시에 수치적 D2NN block이 이산 격자 위에서 거의 완벽한 유니터리 연산자임을 증명한다 ($H^\dagger H \approx I$, 오차 ~10⁻⁶).

**비고: CO 측정 시 비교 쌍의 구분.** 정리 1이 보존하는 양은 $\text{CO}(HU_\text{turb}, HU_\text{vac})$이다—즉, **난류 필드와 진공 필드 모두 동일한 D2NN을 통과**시킨 후의 CO이다. 이와 달리, D2NN 출력과 단순 전파된 진공 기준을 비교한 혼합 지표 $\text{CO}(HU_\text{turb}, U_\text{vac,det})$는 정리 1과 무관하며, 손실함수에 따라 크게 변한다 (Section 4.2 참조).

### 4.2 Phase A: 필드 기반 손실함수와 혼합 CO 지표

Section 4.1에서 정리 1이 보존하는 올바른 양을 확인하였다. 본 절에서는 **D2NN 출력과 진공 기준 사이의 혼합 CO**—즉 $\text{CO}(HU_\text{turb}, U_\text{vac,det})$—를 분석한다. 이 지표는 정리 1의 보존량이 아니며, D2NN이 난류 빔을 진공 빔에 얼마나 가깝게 변환하는지를 측정하는 **실용적 성능 지표**이다.

5가지 CO 기반 손실함수로 학습한 혼합 CO 결과:

| 손실함수 | 혼합 CO (시험) | 기준 CO | 변화 |
|---------|--------------|---------|------|
| co_ffp (최고) | 0.3295 | 0.3044 | +0.0250 (+8.2%) |
| baseline_co | 0.3286 | 0.3044 | +0.0241 (+7.9%) |
| co_amp | 0.3285 | 0.3044 | +0.0241 (+7.9%) |
| co_phasor | 0.2987 | 0.3044 | -0.0057 (-1.9%) |
| roi80 | 0.2733 | 0.3044 | -0.0311 (-10.2%) |

검증 집합에서는 혼합 CO가 0.294에서 0.357로 상승하였으나(+21%), 시험 집합에서는 +8%에 그쳤다. 이 차이는 과적합(overfitting)으로 설명된다: 학습 시 D2NN은 특정 난류 실현들에 대해 $HU_\text{turb} \approx U_\text{vac,det}$가 되도록 위상 마스크를 조정하지만, 보지 못한 실현에 대해서는 이 효과가 일반화되지 않는다. 50개 시험 실현의 다중 실현 통계에서는 모든 구성이 기준선 이하였다.

**혼합 CO와 정리 1의 관계**: 혼합 CO의 변화는 정리 1을 위반하는 것이 아니다. 정리 1은 $\text{CO}(HU_\text{turb}, HU_\text{vac})$의 보존을 말하며, $\text{CO}(HU_\text{turb}, U_\text{vac,det})$는 서로 다른 연산자를 적용한 필드 쌍의 비교이므로 정리의 적용 범위 밖이다.

WF RMS는 모든 구성에서 460 ± 5 nm로 변화가 미미하였다.

### 4.3 결정론적 수차 검증

| 구성 | 수차 | WF RMS 전 | WF RMS 후 | 개선 |
|------|------|----------|----------|------|
| 단층 (전파 없음) | Defocus 2 rad PV | 212.4 nm | 0.1 nm | **99.95%** |
| 단층 (전파 없음) | Coma 3 rad PV | 113.9 nm | — | 완벽 교정 |
| 다층 (10 mm) | Defocus 2 rad PV | 213.1 nm | 213.1 nm | **0%** |

단층 D2NN은 결정론적 수차를 완벽히 교정하였다 ($M_1 = e^{-i\Delta\phi}$를 학습). 그러나 다층 D2NN은 300 에폭 동안 WF RMS가 1 nm도 변하지 않았다. 이는 정리 2의 직접적 실험 검증이다: 유니터리 연산자 H를 통과한 후 L2 거리가 보존되므로, $\|HU_\text{aberr} - HU_\text{vac}\| = \|U_\text{aberr} - U_\text{vac}\|$이며 gradient가 0이 된다.

### 4.4 Phase B: 세기 기반 손실함수 (PIB 중심)

아래 표에서 "혼합 CO"는 Section 4.2에서 정의한 $\text{CO}(HU_\text{turb}, U_\text{vac,det})$이다. 정리 1이 보존하는 $\text{CO}(HU_\text{turb}, HU_\text{vac})$는 Section 4.1에서 확인한 바와 같이 모든 전략에서 0.3044로 불변이다.

| 전략 | PIB@50μm | 혼합 CO | 정리 CO | WF RMS [nm] |
|------|----------|---------|---------|-------------|
| 보정 없음 | 0.0034 | 0.3044 | 0.3044 | 460 |
| **PIB only** | **0.8340** | 0.0147 | 0.3044 | 449 |
| **CO+PIB hybrid** | **0.5082** | 0.1903 | 0.3044 | 461 |
| Strehl only | 0.0012 | 0.0231 | 0.3044 | 448 |
| Intensity overlap | 0.0024 | 0.1839 | 0.3044 | 455 |

**PIB 손실함수**: 집속효율이 0.34%에서 83.4%로 증가하였다 (절대값 기준 +83.1 percentage points). 이는 정적 D2NN이 세기 기반 지표를 극적으로 개선할 수 있음을 보여준다. 한편 혼합 CO는 0.304에서 0.015으로 크게 감소하였는데, 이는 D2NN이 난류 빔의 에너지를 검출 bucket에 집중시키는 과정에서 진공 빔과의 필드 유사도를 희생하기 때문이다. 정리 CO(두 필드 모두 D2NN 통과)는 0.3044로 불변이다.

**CO+PIB 하이브리드 손실함수**: PIB를 50.8%까지 개선하면서 혼합 CO를 0.190으로 유지하였다 (기준 대비 −37%). 정리 CO는 0.3044로 불변이다.

WF RMS는 모든 전략에서 448–461 nm 범위로, 기준선(460 nm)과 통계적으로 동일하였다. 이는 정리 2의 L2 거리 보존(Section 4.1)과 일관된다.

Throughput(처리량)은 모든 경우 1.0000으로, 에너지 손실 없이 공간적 재분배만 일어났음을 확인하였다.

---

## 5. 토론 (Discussion)

### 5.1 유니터리 불변성의 물리적 해석

정리 1–3의 물리적 의미는 다음과 같이 직관적으로 이해할 수 있다: 정적 D2NN은 모든 입력에 대해 **동일한 선형 변환** H를 적용한다. 난류 빔과 진공 빔이 같은 "신발"을 신으면, 두 빔 사이의 "키 차이"(필드 수준 유사도)는 변하지 않는다.

이 한계는 D2NN에 국한되지 않는다. **어떠한 유니터리 수동 광학 시스템**—Mach-Zehnder 간섭계 메시(MZI mesh), 다모드 도파관, 회절 광학 소자—도 동일한 제약을 받는다. 이는 양자역학의 유니터리 진화와 동일한 수학적 구조이며, 정보 이론적으로는 정적 유니터리 채널의 용량 한계에 해당한다.

### 5.2 비선형 채널과 공간 모드 필터링

PIB의 극적 개선(0.34% → 83.4%)은 D2NN이 **파면 교정기(wavefront corrector)**가 아닌 **공간 모드 필터(spatial mode filter)**로 작동함을 보여준다. D2NN은 입사 빔의 에너지를 회절을 통해 공간적으로 재분배하여, 검출 영역(bucket)에 에너지를 집중시킨다.

이 과정에서 혼합 CO($\text{CO}(HU_\text{turb}, U_\text{vac,det})$)는 0.304에서 0.015로 감소한다. 이는 D2NN이 난류 빔을 진공 빔과 다른 형태로 변환함을 의미한다. 그러나 정리 CO($\text{CO}(HU_\text{turb}, HU_\text{vac})$)는 0.3044로 불변이다 (Section 4.1). 즉, **D2NN은 두 빔 사이의 필드 수준 유사도를 변화시키지 않으면서(정리 1), 출력 평면에서의 세기 분포만을 재배치**한다. 세기 집중(PIB↑)과 혼합 필드 충실도(혼합 CO 유지)는 상충 관계에 있지만, 이 상충은 유니터리 불변성의 파괴가 아니라 유니터리 변환이 만들어내는 세기 재분배의 자연스러운 결과이다.

CO+PIB 하이브리드 손실함수는 이 트레이드오프의 절충점을 찾은 것으로 해석된다: PIB를 50.8%까지 개선하면서 혼합 CO를 0.190으로 유지하였다.

### 5.3 한계와 주의점

본 연구에는 다음과 같은 한계가 있다:

1. **유한 계산 영역**: 이론적으로 D2NN은 유니터리이지만, 유한 격자에서는 격자 경계에서의 에너지 누출이 발생할 수 있어 엄밀한 의미에서 준-유니터리(quasi-unitary)이다. 그러나 throughput = 1.0000 결과는 이 효과가 무시할 수 있음을 보여준다.

2. **단일 파장**: 본 연구는 단색광(1.55 μm)만을 고려하였다. 다색광(polychromatic) 조건에서는 파장 의존 회절으로 인해 결과가 달라질 수 있다.

3. **편광 무시**: 스칼라 근사를 사용하였다. 벡터 회절(vector diffraction) 효과는 고려하지 않았다.

4. **PIB 247배 증가의 해석**: 기준선 PIB가 극히 낮은 값(0.34%)에서 시작하였으므로, 절대적 개선은 크지만 이미 높은 PIB를 가진 시스템에서는 같은 배율의 개선이 불가능할 수 있다.

### 5.4 향후 연구 방향

정적 D2NN의 유니터리 불변성 한계를 극복하기 위한 세 가지 방향을 제시한다:

1. **적응형 D2NN (Adaptive D2NN)**: 합성곱 신경망(CNN)이 입력 세기 패턴에서 파면을 추정하고, 실현별로 다른 위상 마스크를 예측한다. 이는 유니터리 불변성을 깨트리는 **입력 의존적** 변환을 구현한다.

2. **비선형 D2NN**: 포화 흡수체(saturable absorber)와 같은 비선형 광학 소자를 층간에 삽입한다. 비선형 소자는 유니터리가 아니므로 ($H^\dagger H \neq I$), 필드 수준 지표의 개선도 가능하다.

3. **하이브리드 시스템**: 정적 D2NN으로 결정론적 수차(망원경 정렬 오차, 렌즈 수차)를 교정하고, 적응형 소자로 랜덤 난류를 보상하는 2단 구조.

---

## 참고문헌

[1] A. K. Majumdar, "Free Space Laser Communication," Springer, 2015.

[2] L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random Media," SPIE Press, 2005.

[3] R. K. Tyson, "Principles of Adaptive Optics," CRC Press, 2015.

[4] X. Lin et al., "All-optical machine learning using diffractive deep neural networks," Science 361, 1004–1008 (2018).

[5] J. Tao et al., "Diffractive deep neural network for optical vortex recognition," Opt. Lett. 44, 2650 (2019).

[6] J. D. Schmidt, "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB," SPIE Press, 2010.

---

## 부록: Figure 목록

- **그림 1**: 시스템 구성도 (TX → 대기 → 망원경 → 빔축소기 → D2NN → 검출기)
- **그림 2**: 유니터리 불변성의 실험적 증명 (에폭별 CO/WF RMS/PIB)
- **그림 3**: 결정론적 vs 랜덤 — 단층 성공 vs 다층 실패
- **그림 4**: 손실함수 전략별 성능 비교 (PIB/CO/WF RMS)
- **그림 5**: CO–PIB 트레이드오프 곡선 (Pareto 경계)
- **그림 6**: 에너지 공간 재분배 맵 (검출기 평면)
- **그림 7**: 정리 1–2의 수치 검증 (theorem1_co_verification.png, theorem2_l2_verification.png)
  - (a) CO(U_t,U_v) vs CO(HU_t,HU_v) 산점도 — 정리 1 (|diff| < 10⁻⁷)
  - (b) 혼합 CO 전략별 비교 — 정리 1의 양이 아님을 명시
  - (c) PIB-only 전략에서 세 가지 CO 정의 비교
- **그림 8**: L2 거리 보존 검증 (theorem2_l2_verification.png)
  - (a) ||U_t−U_v|| vs ||HU_t−HU_v|| 산점도 — 정리 2 (상대 오차 ~10⁻⁶)
  - (b) 상대 오차 분포 히스토그램
- **표 S1**: 정리 검증 요약표 (theorem_verification_table.png)
