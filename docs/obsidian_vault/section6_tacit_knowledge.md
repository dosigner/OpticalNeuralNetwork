---
title: "Section 6: 암묵지 총정리 — 논문이 말하지 않은 것들"
tags:
  - tacit-knowledge
  - reproduction
  - D2NN
aliases:
  - 암묵지
  - Tacit Knowledge
date: 2026-03-16
cssclasses:
  - nature-photonics
---

# 6. 암묵지 총정리 — 논문이 말하지 않은 것들

> [!abstract] Complete Tacit Knowledge Summary
> Luo et al., "Computational Imaging Without a Computer," *eLight* 2 (2022) 재현 과정에서 발견한 **30개의 암묵지 항목**을 **6개 범주**로 분류하여 정리한다. 각 항목은 **논문 서술 → 실제 구현 → 물리적/수학적 근거 → 재현자를 위한 교훈**의 4단 구조로 기술한다.

![[images/fig_tacit_knowledge_map.png|800]]

---

## 6.1 물리 모델링의 숨겨진 가정들

논문은 THz 영역에서의 자유공간 전파를 시뮬레이션한다고 기술하지만, scalar 근사, coherent 조명, evanescent wave 처리, 전파 거리의 비대칭 설계 등 핵심적인 물리 모델링 가정들은 명시적으로 정당화되지 않는다. 이 소절에서는 [[section2_diffuser_physics|섹션 2]]와 [[section3_system_design|섹션 3]]에서 다루었던 물리 모델의 기저에 깔린 4가지 암묵적 가정을 분석한다.

> [!important]+ TK-1. Scalar 근사와 편광 무시
>
> **논문 서술:** "THz 영역에서 자유공간 전파를 시뮬레이션한다"고만 기술하며, scalar wave 근사를 명시적으로 정당화하지 않는다.
>
> **실제 구현:** `optics.scalar_model: true`로 전체 파이프라인이 스칼라 복소 필드 $U(x,y)$ 하나로 동작한다. 편광 상태, 벡터 회절, cross-polarization coupling은 전혀 고려하지 않는다.
>
> **물리적 근거:** 논문의 그리드 피치(0.3 mm)와 파장(0.75 mm)의 비가 $\Delta x / \lambda = 0.4$이다. 이 정도 scale에서는 feature size가 파장과 비슷하여 본래 vector diffraction 효과가 무시할 수 없는 영역이다. 그러나 D2NN 레이어가 phase-only 박막이고, 디퓨저도 thin phase screen 모델이므로, 투과 시 편광 회전이 없다고 가정하면 scalar 근사가 성립한다.
>
> > [!tip] 재현자를 위한 교훈
> > Scalar 모델은 "물리적으로 정확하기 때문에" 쓰는 것이 아니라, "학습 가능한 수준의 계산 복잡도를 유지하면서도 논문의 핵심 메시지(디퓨저 다양성 → 일반화)를 재현하기에 충분하기 때문에" 쓰는 것이다. Vector 모델로 전환하면 GPU 메모리가 ~4배 증가하며, 재현 목적에서는 불필요하다.

> [!important]+ TK-2. Coherent illumination 가정
>
> **논문 서술:** 400 GHz 평면파 조명을 사용한다고 기술한다.
>
> **실제 구현:** 입력 amplitude를 `torch.complex64`로 캐스팅하여 위상이 0인 완전 coherent 평면파로 취급한다.
> ```python
> field = amplitude.to(torch.complex64)  # trainer.py line 139
> ```
>
> **물리적 근거:** THz 소스는 실제로 spatial coherence length가 유한하지만, 논문의 시뮬레이션 프레임워크에서는 완전 coherent를 가정한다. 이는 intensity 출력이 $|U|^2$로 계산되는 근거이며, partial coherence 하에서는 mutual coherence function을 전파해야 하므로 계산량이 $O(N^4)$로 폭증한다.
>
> > [!tip] 재현자를 위한 교훈
> > `detector_type: intensity`와 `coherent: true`의 조합은 "coherent imaging → intensity detection"을 의미한다. 이 조합을 바꾸면 loss landscape 전체가 달라지므로, 논문 재현 시 반드시 유지해야 한다.

> [!important]+ TK-3. Evanescent wave 처리
>
> **논문 서술:** BL-ASM (Band-Limited Angular Spectrum Method)을 사용한다고만 언급한다.
>
> **실제 구현:** `bl_asm.py`에서 $f_x^2 + f_y^2 \geq 1/\lambda^2$ 인 주파수 성분을 strict inequality로 마스킹한다:
> ```python
> propagating = f_sq < cutoff_sq  # strict < to avoid sqrt(0) edge
> ```
> 경계에서 $f_x^2 + f_y^2 = 1/\lambda^2$인 성분도 제거한다.
>
> **물리적 근거:** 경계 주파수를 포함하면 `sqrt(0)`이 되어 위상이 0이지만, 수치적으로 floating point 오류가 누적될 수 있다. Strict inequality로 제거하는 것이 안전하다.
>
> > [!tip] 재현자를 위한 교훈
> > `>=` vs `<`의 차이는 단일 전파에서는 미미하지만, D2NN처럼 5~6회 연속 전파하는 시스템에서는 경계 처리 불일치가 누적된다. 반드시 strict `<`를 사용해야 한다.

> [!important]+ TK-4. 전파 거리의 비대칭 설계
>
> **논문 서술:** D2NN의 기하 구조를 대략적으로만 서술한다.
>
> **실제 구현:** 거리가 비대칭이다:
> - Object → Diffuser: **40.0 mm** (원거리)
> - Diffuser → Layer 1: **2.0 mm** (근거리)
> - Layer → Layer: **2.0 mm**
> - Last Layer → Output: **7.0 mm** (다른 레이어 간 거리의 3.5배)
>
> **물리적 근거:** Object → Diffuser 40 mm는 물체의 Fresnel 회절이 디퓨저 면에서 충분히 발달하도록 보장한다. Last Layer → Output의 7.0 mm는 마지막 위상 레이어의 변조가 출력면에서 intensity pattern으로 변환되기 위한 충분한 자유공간 전파 거리이다. 2.0 mm 간격의 레이어는 $z/\lambda \approx 2.67$로, 근거리 Fresnel 영역에서 동작한다.
>
> > [!tip] 재현자를 위한 교훈
> > 이 거리값들은 논문의 핵심 하이퍼파라미터이다. 특히 마지막 ==7.0 mm==를 2.0 mm로 바꾸면, 위상-강도 변환이 불충분하여 출력 contrast가 급격히 떨어진다. 이 비대칭은 물리적 필연이지, 논문이 강조해야 했지만 하지 않은 설계 결정이다.

---

## 6.2 수치 구현의 비자명한 선택들

FFT 기반 전파, 주파수 그리드 정렬, 수치 정밀도, smoothing padding, transfer function 캐싱 등 논문이 전혀 언급하지 않는 수치적 선택들이 재현 결과에 유의미한 영향을 미친다. [[section3_system_design|섹션 3]]의 BL-ASM 구현과 직접적으로 관련된다.

> [!important]+ TK-5. Zero-padding 전략과 aliasing 방지
>
> **논문 서술:** 전파에 ASM을 사용한다고 기술한다. Padding에 대한 언급이 없다.
>
> **실제 구현:** ==`pad_factor=2`==로 240 → 480 그리드로 zero-padding 후 FFT를 수행한다:
> ```python
> N_pad = N * pad_factor  # 240 * 2 = 480
> padded = F.pad(field, (pad_lo, pad_hi, pad_lo, pad_hi), mode="constant", value=0)
> ```
>
> **물리적 근거:** FFT 기반 convolution은 cyclic convolution이므로, padding 없이 전파하면 한쪽 경계에서 나간 빛이 반대쪽에서 들어오는 wrap-around artifact가 발생한다. ==`pad_factor=2`==는 Nyquist 조건을 만족시키는 최소 패딩이다.
>
> > [!tip] 재현자를 위한 교훈
> > `pad_factor=1`로 실행하면 visually은 비슷해 보이지만, PCC가 0.02~0.05 정도 낮아지고 수렴이 불안정해진다. `pad_factor=3`은 메모리만 낭비하고 성능 차이가 미미하다. **2는 sweet spot이지만 논문에 명시되어 있지 않다.**

> [!important]- TK-6. 주파수 그리드의 fftfreq 순서 유지
>
> **논문 서술:** ASM 전파를 서술하지만 FFT shift 여부를 명시하지 않는다.
>
> **실제 구현:** Transfer function $H(f_x, f_y)$를 `fftfreq` 순서(DC 좌상단)로 생성하고, 전파 시에도 `fftshift`를 적용하지 않는다:
> ```python
> spectrum = torch.fft.fft2(padded)
> spectrum = spectrum * H  # 둘 다 fftfreq 순서
> propagated = torch.fft.ifft2(spectrum)
> ```
>
> **물리적 근거:** `fftshift`/`ifftshift` 쌍을 생략하면 추가 메모리 할당 없이 element-wise 곱셈만으로 전파가 완료된다. 다만 $H$를 생성할 때도 반드시 동일한 `fftfreq` 순서를 사용해야 한다.
>
> > [!tip] 재현자를 위한 교훈
> > Correlation length 추정(`correlation.py`)에서는 `fftshift`를 사용하여 autocorrelation을 중심에 놓는다. 전파와 분석에서 shift 규약이 다르므로, 혼동하면 결과가 완전히 틀어진다.

> [!important]- TK-7. float64 → complex64 정밀도 전략
>
> **논문 서술:** 수치 정밀도에 대한 언급이 없다.
>
> **실제 구현:** 디퓨저 height map은 `float64`로 생성하고, smoothing도 `float64`에서 수행한 뒤, 최종 transmittance만 `complex64`로 변환한다:
> ```python
> W = torch.randn(N, N, generator=gen, dtype=torch.float64) * sigma0_mm + mu_mm
> # ... smoothing in float64 ...
> transmittance = torch.exp(1j * phase_map).to(torch.complex64)
> ```
>
> **물리적 근거:** Phase wrapping($\phi \mod 2\pi$)은 수치적으로 민감하다. `float32`에서 $2\pi \cdot \Delta n \cdot D / \lambda$를 계산하면 $D$가 $25\lambda$ 수준일 때 유효 자릿수가 부족해진다. `float64`로 중간 계산을 수행하면 phase의 정밀도가 $\sim 10^{-14}$ rad 수준으로 유지된다.
>
> > [!tip] 재현자를 위한 교훈
> > 전체를 `float32`로 구현하면 디퓨저 간 상관길이 추정에서 ~5% 오차가 발생한다. 디퓨저 생성과 correlation 분석만 `float64`, 나머지는 `float32/complex64`가 최적 전략이다.

> [!important]- TK-8. Gaussian smoothing의 reflect padding
>
> **논문 서술:** 디퓨저 height map에 Gaussian smoothing을 적용한다고만 기술한다.
>
> **실제 구현:** Separable 1-D convolution을 `mode="reflect"` padding으로 수행한다:
> ```python
> out = F.pad(inp, (pad_size, pad_size, 0, 0), mode="reflect")
> ```
>
> **물리적 근거:** `mode="zero"`로 padding하면 경계에서 height가 갑자기 0으로 떨어져 비물리적인 phase jump가 생긴다. `mode="reflect"`는 경계 근처의 통계적 특성을 보존한다.
>
> > [!tip] 재현자를 위한 교훈
> > Padding mode 차이는 디퓨저 중앙부에는 거의 영향이 없지만, correlation length를 radial average로 추정할 때 경계 효과가 fitting에 영향을 준다. 논문의 $L \approx 10\lambda$ 값을 재현하려면 reflect padding이 필수적이다.

> [!important]- TK-9. Transfer function 캐싱
>
> **논문 서술:** 언급 없음.
>
> **실제 구현:** `bl_asm.py`에서 `(N, dx, λ, z, pad_factor, device, dtype)` 튜플을 키로 하는 딕셔너리 캐시를 운영한다.
>
> **물리적 근거:** D2NN forward pass에서 동일한 전파 거리의 transfer function이 매 batch마다 재사용된다. 캐싱 없이는 240×240 grid에서 epoch당 ~20초의 불필요한 오버헤드가 발생한다.
>
> > [!tip] 재현자를 위한 교훈
> > 멀티 GPU 학습 시 캐시가 device별로 분리되지 않으면 device mismatch 에러가 발생한다. 캐시 키에 `dev_str`이 포함된 이유이다.

---

## 6.3 학습 파이프라인의 암묵적 설계 결정

배치 확장 전략, 디퓨저 갱신 주기, 시드 체계, loss 함수 구조, PCC 계산 방식, learning rate schedule 등 [[section4_5_training_results|섹션 4-5]]에서 다루었던 학습 파이프라인의 핵심 설계 결정들을 분석한다. 이 항목들은 논문의 "Methods" 섹션에서도 충분히 기술되지 않았다.

> [!important]- TK-10. B × n 배치 확장 전략
>
> **논문 서술:** "각 epoch에서 $n$개의 diffuser를 사용한다"고만 기술한다.
>
> **실제 구현:** 배치 크기 $B$의 각 object에 대해 $n$개 diffuser를 모두 적용하여 $B \times n$ 개의 output을 생성한 뒤, target과 mask를 $n$번 복제하여 loss를 계산한다:
> ```python
> # (n, B, N, N) -> (B*n, N, N) via permute+reshape
> stacked = torch.stack(outputs, dim=0)
> result = stacked.permute(1, 0, 2, 3).reshape(B * n, N, N)
> ```
> Target 복제:
> ```python
> target_dup = target.unsqueeze(1).expand(B, n, ...).reshape(B * n, ...)
> ```
>
> **물리적 근거:** 이 전략은 동일 object가 서로 다른 산란 조건을 통과한 결과를 **하나의 loss에 동시에 반영**한다. 이것이 논문의 핵심 메커니즘인 "diffuser diversity as regularization"의 구현이다.
>
> > [!tip] 재현자를 위한 교훈
> > $n=20$, $B=4$이면 유효 배치가 80이다. GPU 메모리가 부족하면 $B$를 줄여야 하지, $n$을 줄여서는 안 된다. $n$은 물리적 실험 설계이고, $B$는 수치적 편의이다.

> [!important]- TK-11. Epoch 단위 디퓨저 갱신 — batch 단위가 아닌 이유
>
> **논문 서술:** "매 epoch마다 새로운 diffuser를 생성한다"고 기술한다.
>
> **실제 구현:** `train_epoch()` 시작 시 $n$개 디퓨저를 생성하고, 해당 epoch의 모든 batch에서 동일한 디퓨저 세트를 사용한다. Batch마다 갱신하지 않는다.
>
> **물리적 근거:** 실제 실험에서 한 epoch 동안 diffuser를 교체하지 않고 여러 object를 통과시키는 것에 대응한다. 또한, batch마다 diffuser를 바꾸면 gradient의 분산이 과도하게 커져 수렴이 불안정해진다.
>
> > [!tip] 재현자를 위한 교훈
> > Batch 단위 갱신을 시도하면 수렴은 하지만 PCC가 ~0.02 낮고 학습 곡선의 진동이 심하다. Epoch 단위 갱신이 논문 재현의 핵심이다.

> [!important]- TK-12. 디퓨저 시드 체계: `epoch_seed * 1000 + i`
>
> **논문 서술:** 재현성을 위한 시드 전략에 대한 언급이 없다.
>
> **실제 구현:**
> ```python
> epoch_seed = self.base_seed + epoch
> seed = epoch_seed * 1000 + i  # i = 0..n-1
> ```
>
> **물리적 근거:** 이 체계는 (1) epoch 간 디퓨저가 겹치지 않고, (2) 같은 epoch 내 디퓨저끼리도 겹치지 않으며, (3) 평가용 "blind" 디퓨저(`seed = 77777777 + i`)가 학습용과 절대 겹치지 않도록 보장한다.
>
> > [!tip] 재현자를 위한 교훈
> > 시드 충돌이 발생하면 "known" vs "new" 디퓨저 비교가 무의미해진다. 특히 `base_seed + last_epoch`의 known 디퓨저와 blind 시드(`77777777`)의 간격이 충분한지 확인해야 한다. $30 \text{ epochs} \times 1000 + 20 < 77777777$ 이므로 안전하다.

> [!important]- TK-13. Loss 함수의 비자명한 구조
>
> **논문 서술:** PCC를 최적화한다고 기술한다.
>
> **실제 구현:** 실제 loss는 `-PCC + energy_penalty`의 합이다:
> ```python
> def pcc_energy_loss(output, target, mask, alpha=1.0, beta=0.5):
>     return -pearson_correlation(output, target) + energy_penalty(output, mask, alpha, beta)
> ```
> Energy penalty는 mask 외부의 에너지를 억제하고 mask 내부의 에너지를 장려한다:
> $$E = \frac{\alpha \sum (1-m) \cdot I - \beta \sum m \cdot I}{\sum m}$$
>
> **물리적 근거:** PCC만으로는 출력의 에너지가 object 영역 밖으로 퍼져도 상관관계가 높을 수 있다. Energy penalty는 "에너지를 올바른 위치에 집중시켜라"는 물리적 제약이다.
>
> > [!tip] 재현자를 위한 교훈
> > ==`alpha=1.0`==, ==`beta=0.5`==는 논문에 명시되지 않은 하이퍼파라미터이다. `beta`를 0으로 놓으면 출력이 어둡고 contrast가 낮아지며, `beta > 1.0`이면 energy explosion이 발생한다. 이 값들은 시행착오로 찾아야 한다.

> [!important]- TK-14. PCC 계산에서의 전역 평균 vs 국소 평균
>
> **논문 서술:** PCC를 사용한다고만 기술한다.
>
> **실제 구현:** 전체 이미지를 flatten하여 전역 PCC를 계산한다:
> ```python
> o = output.reshape(output.shape[0], -1)  # (B, N*N)
> o_mean = o.mean(dim=1, keepdim=True)     # 전역 평균
> ```
>
> **물리적 근거:** 국소 PCC(patch 단위)를 사용하면 고주파 디테일을 더 잘 살릴 수 있지만, 전역 PCC가 논문의 정의이며 D2NN의 전역적 위상 변조 특성과도 잘 맞는다.
>
> > [!tip] 재현자를 위한 교훈
> > SSIM이나 국소 PCC로 대체하면 학습 역학이 완전히 달라진다. 논문 재현에는 반드시 전역 PCC를 사용해야 한다.

> [!important]- TK-15. Learning rate schedule의 보수적 설정
>
> **논문 서술:** Adam optimizer를 사용한다고만 기술한다.
>
> **실제 구현:** ==`gamma=0.99`==의 epoch 단위 multiplicative decay. 30 epochs 후 LR은 초기의 $0.99^{30} \approx 0.74$ 수준으로, 거의 줄어들지 않는다.
>
> **물리적 근거:** D2NN의 위상 파라미터는 $[0, 2\pi)$ 범위에서 자유롭게 움직여야 하므로, 공격적인 LR decay는 국소 최적해에 조기 수렴을 유발한다. 보수적 decay가 디퓨저 다양성에 대한 적응을 돕는다.
>
> > [!tip] 재현자를 위한 교훈
> > Cosine annealing이나 step decay를 시도하면 $n \geq 10$에서 수렴이 불안정해진다. ==`gamma=0.99`==는 "거의 안 줄이면서 형식적으로 decay한다"는 의미이다.

---

## 6.4 디퓨저 생성의 숨겨진 물리학

Random phase diffuser의 height map 통계, smoothing sigma와 correlation length 사이의 비선형 관계, correlation length 추정을 위한 fitting 전략 등 [[section2_diffuser_physics|섹션 2]]에서 다루었던 디퓨저 물리학의 미묘한 구현 세부사항을 분석한다.

> [!important]- TK-16. Height map 통계와 phase wrapping의 관계
>
> **논문 서술:** "random phase diffuser"를 사용한다고 기술한다.
>
> **실제 구현:** Height map의 mean이 $25\lambda$이고 std가 $8\lambda$이다. Phase는 $\phi = 2\pi \Delta n \cdot D / \lambda$이므로, mean phase $= 2\pi \times 0.74 \times 25 \approx 116.2$ rad, 즉 $\sim 18.5$ 바퀴를 돌고도 남는다.
>
> **물리적 근거:** 평균 높이가 이렇게 크면 phase가 $[0, 2\pi)$ 내에서 거의 균일 분포하게 되어, 특정 위상 값에 편향되지 않는 "완전한" random phase screen이 된다. std $= 8\lambda$는 인접 픽셀 간 phase jump가 $\sim 2\pi \times 0.74 \times 8 / \sqrt{2} \approx 26$ rad (smoothing 전) 수준임을 의미한다.
>
> > [!tip] 재현자를 위한 교훈
> > `height_mean_lambda`를 1~5 수준으로 줄이면 phase가 $[0, 2\pi)$를 고르게 덮지 못하여 디퓨저의 산란 강도가 약해지고, D2NN이 학습할 "산란 보정" 자체가 줄어든다.

> [!important]- TK-17. Smoothing sigma와 correlation length의 비선형 관계
>
> **논문 서술:** Correlation length $L \approx 10\lambda$라고 기술한다.
>
> **실제 구현:** Smoothing sigma를 $4\lambda$로 설정하고, target correlation length를 $10\lambda$로 기록한다. 하지만 실제 correlation length는 autocorrelation fitting으로 사후 추정한다.
>
> **물리적 근거:** Gaussian smoothing의 sigma와 autocorrelation의 correlation length는 정확히 같지 않다. 논문의 Eq. 5에서 정의하는 $R_d(r) = \exp(-\pi r^2 / L^2)$에 따르면, white noise에 $\sigma_{smooth}$의 Gaussian smoothing을 적용한 결과의 autocorrelation length는 대략 $L \approx 2\sigma_{smooth} \cdot \sqrt{\pi / 2} \approx 2.5 \sigma_{smooth}$이다. $\sigma = 4\lambda$이면 $L \approx 10\lambda$가 된다.
>
> > [!tip] 재현자를 위한 교훈
> > "Correlation length $10\lambda$"를 직접 달성하려면 sigma를 $4\lambda$로 설정해야 한다는 것이 논문에 명시되어 있지 않다. FigS5에서 $L \approx 5\lambda$를 얻기 위해 `smoothing_sigma_lambda=2.0`을 사용하는 것도 같은 관계에 기반한다.

> [!important]- TK-18. Correlation length 추정의 fitting 전략
>
> **논문 서술:** Correlation length를 사용하지만 추정 방법을 명시하지 않는다.
>
> **실제 구현:** Autocorrelation의 radial average를 $\ln R(r) = -\pi r^2 / L^2$에 least-squares fitting한다. 핵심 선택들:
> 1. `max_r = min(mid, 60)`: fitting 범위를 60 픽셀로 제한
> 2. `valid = r_avg > 0.05`: 노이즈가 큰 tail을 제거
> 3. `valid[0] = False`: $r=0$ 점(항상 1.0) 제외
> 4. Intercept 없는 fitting: $\log R = \text{slope} \cdot r^2$ (forced through origin)
>
> **물리적 근거:** Forced-through-origin fitting은 $R(0) = 1$이라는 autocorrelation의 정의를 활용하여 자유도를 줄인다. Tail 제거는 유한 격자 효과와 수치 노이즈를 방지한다.
>
> > [!tip] 재현자를 위한 교훈
> > $R > 0.05$ threshold를 $R > 0.01$로 바꾸면 $L$ 추정값이 ~15% 변할 수 있다. 논문의 $L \approx 10\lambda$ 값은 이 특정 fitting 전략에 의존한다.

---

## 6.5 재현 과정에서 발견한 민감도 분석

n-sweep 포화 곡선, known vs new 디퓨저 역전 현상, grating period 추정의 불안정성, contrast enhancement, input encoding 방식 등 [[section4_5_training_results|섹션 4-5]]의 실험 결과를 해석하는 데 필수적인 민감도 분석 결과를 정리한다.

> [!important]- TK-19. n-sweep에서의 포화 곡선과 4-layer 용량 한계
>
> **논문 서술:** $n$이 증가하면 일반화가 개선된다고 정성적으로 기술한다.
>
> **실제 구현:** 재현 실험에서 training PCC의 단조 감소를 확인:
> - $n=1$: 0.907, $n=10$: 0.884, $n=15$: 0.879, $n=20$: 0.878
> - $n=10 \to 15$: $\Delta = -0.005$, $n=15 \to 20$: $\Delta = -0.0008$
>
> **물리적 근거:** 4-layer D2NN의 학습 파라미터 수는 $4 \times 240^2 = 230{,}400$개이다. 이 자유도로 동시에 만족시킬 수 있는 독립적인 산란 조건의 수에는 상한이 있으며, $n \approx 15$에서 사실상 포화된다.
>
> > [!tip] 재현자를 위한 교훈
> > $n=20$은 "성능 향상"이 아니라 "포화 확인" 실험이다. 비용 효율이 중요하면 $n=15$가 최적이고, 결론의 강도가 중요하면 $n=20$까지 돌려야 한다.

> [!important]- TK-20. Known vs New 디퓨저 역전 현상
>
> **논문 서술:** Known diffuser에서의 성능이 new diffuser보다 높을 것으로 암묵적으로 기대한다.
>
> **실제 구현:** $n=15$에서 known PCC = 0.860, new PCC = 0.878로 ==**new가 더 높다**==.
>
> **물리적 근거:** 네트워크가 마지막 epoch의 특정 디퓨저 집합을 "암기"한 것이 아니라, diffuser distribution 전체에 대한 평균적 산란 보정을 학습했기 때문이다. 마지막 epoch의 디퓨저는 학습의 마지막 gradient update에서 약간의 과적합 편향이 남아 있을 수 있다.
>
> > [!tip] 재현자를 위한 교훈
> > 이 역전 현상은 논문의 핵심 주장("diffuser diversity enables generalization")의 가장 강력한 증거 중 하나이다. Known > New가 나오면 과적합을 의심해야 한다.

> [!important]- TK-21. Grating period 추정의 불안정성
>
> **논문 서술:** Resolution target으로 일반화를 평가한다고 기술한다.
>
> **실제 구현:** 1D profile에서 peak detection 후, 실패 시 3-Gaussian fitting으로 fallback한다. 그러나 작은 period (7.2 mm, 8.4 mm)에서는:
> - Bar 간격이 좁아 peak가 merge될 수 있음
> - `find_peaks`의 height/prominence threshold가 0.25/0.10으로 하드코딩됨
>
> **물리적 근거:** 7.2 mm period에서 bar width = 12 pixels, gap = 12 pixels이다. Diffuser에 의한 산란과 D2NN 전파를 거치면 PSF가 ~5 pixels 수준으로 퍼지므로, 이 정도 period에서는 peak separation이 불확실해진다.
>
> > [!tip] 재현자를 위한 교훈
> > Fig.3의 작은 period 영역에서 에러바가 큰 것은 D2NN의 해상 한계와 period 추정 알고리즘의 한계가 모두 반영된 결과이다. 이 둘을 분리하기 어렵다는 점을 인지해야 한다.

> [!important]- TK-22. Contrast enhancement가 시각적 품질과 PCC를 분리한다
>
> **논문 서술:** Reconstruction 이미지를 보여주지만 후처리 여부를 명시하지 않는다.
>
> **실제 구현:** PCC는 raw intensity로 계산하고, display 이미지는 percentile stretch (1%~99%)를 적용한다:
> ```yaml
> contrast_enhancement:
>   type: percentile_stretch
>   lower_percentile: 1.0
>   upper_percentile: 99.0
> ```
>
> **물리적 근거:** D2NN 출력의 dynamic range는 매우 넓어서, 몇 개의 bright spot이 나머지를 압도한다. Percentile stretch 없이는 이미지가 거의 검게 보인다.
>
> > [!tip] 재현자를 위한 교훈
> > 논문의 "보기 좋은" reconstruction 이미지와 PCC 수치 사이에는 간극이 있다. PCC가 0.88이어도 raw 이미지는 상당히 흐려 보일 수 있으며, 이는 정상이다. ==**Display와 metric을 혼동하면 안 된다.**==

> [!important]- TK-23. Input encoding: amplitude vs intensity
>
> **논문 서술:** MNIST digits를 입력으로 사용한다고 기술한다.
>
> **실제 구현:** 28$\times$28 MNIST를 160$\times$160으로 bilinear resize 후 240$\times$240으로 zero-pad한다. Grayscale 값 $[0,1]$을 **amplitude**로 사용한다 (intensity가 아님).
>
> **물리적 근거:** Amplitude encoding은 $U_{in} = a(x,y) \cdot e^{j \cdot 0}$를 의미한다. 즉 물체가 intensity transmittance $t = a^2$이 아니라 amplitude transmittance $t = a$를 가진다고 가정한다. 이 선택은 물체의 대비를 높여 D2NN의 학습을 돕는다.
>
> > [!tip] 재현자를 위한 교훈
> > Intensity encoding ($I_{in} = a^2$)으로 바꾸면 입력 contrast가 낮아지고 PCC가 ~0.03 떨어진다. 논문의 수치를 재현하려면 반드시 amplitude encoding을 사용해야 한다.

---

## 6.6 논문의 모호한 서술과 실제 구현의 차이

논문 본문의 모호한 서술이 구현 시 어떻게 해석되어야 하는지를 분석한다. "20 diffusers per epoch"의 이중 의미, resolution target의 출처, pruning 실험의 세부사항, phase parameterization, resize pipeline, blind 평가 시드 분리, correlation length 변경 방법 등 [[section4_5_training_results|섹션 4-5]]의 실험 설계에서 논문과 구현 사이의 간극이 가장 두드러지는 항목들이다.

> [!important]- TK-24. "20 diffusers per epoch"의 이중 의미
>
> **논문 서술:** Training에 $n$개의 diffuser를 사용한다고 기술한다.
>
> **실제 구현:** $n$은 두 가지 맥락에서 사용된다:
> 1. **Training:** epoch당 사용하는 diffuser 수 (학습 중 다양성)
> 2. **Evaluation:** 평가 시 사용하는 diffuser 수 (통계적 신뢰도)
>
> Evaluation의 blind diffuser 수(`blind_test_new_diffuser_count: 20`)는 training의 $n$과 독립적이다.
>
> > [!tip] 재현자를 위한 교훈
> > $n=1$ 모델도 20개의 blind diffuser로 평가한다. 평가의 $n$과 학습의 $n$을 혼동하면 Fig.3의 해석이 완전히 달라진다.

> [!important]- TK-25. "Resolution target"은 MNIST에 없다
>
> **논문 서술:** MNIST로 학습하고 resolution target으로 평가한다.
>
> **실제 구현:** Resolution target(3-bar grating)은 학습 데이터에 전혀 포함되지 않는다. 이것이 Fig.3의 핵심 의미이다: "MNIST 숫자만 보고 학습한 네트워크가, 학습 때 보지 못한 선형 해상도 타깃에도 일반화되는가?"
>
> **물리적 근거:** D2NN이 학습하는 것은 "특정 object class의 복원"이 아니라 "산란 보정 자체"이다. 따라서 학습 데이터의 형태와 무관하게 공간주파수 보존이 일반화된다.
>
> > [!tip] 재현자를 위한 교훈
> > Fig.3의 해석 포인트는 "period recovery accuracy"이지 "classification accuracy"가 아니다. x축은 입력 grating의 true period이고, y축은 출력에서 추정한 period이다. 막대가 초록선에 가까울수록 좋다.

> [!important]- TK-26. Pruning 실험(FigS4)의 mask 생성 파이프라인
>
> **논문 서술:** 학습된 레이어의 일부를 pruning한다고 기술한다.
>
> **실제 구현:** 6단계의 pruning condition이 존재하며, 각각 다른 마스크 생성 방법을 사용한다:
> 1. `full_layers`: 마스크 없음 (baseline)
> 2. `no_layers`: 모든 위상을 0으로 (diffractive layer 제거)
> 3. `islands_only`: phase island detection으로 추출
> 4. `dilated_islands`: island를 3$\times$3 binary dilation
> 5. `inside_contour`: convex hull + dilation
> 6. `aperture_80lambda`: $80\lambda$ 반경 원형 aperture
>
> 특히 `no_layers` 조건에서는 D2NN model을 통과하지 않고, diffuser → output까지의 총 자유공간 거리($2.0 + 3 \times 2.0 + 7.0 = 15.0$ mm)에 해당하는 단일 transfer function으로 전파한다.
>
> > [!tip] 재현자를 위한 교훈
> > `no_layers`는 단순히 "phase를 0으로 설정"하는 것이 아니라, 물리적으로 레이어 자체를 제거하는 것이다. 이 두 가지는 다른 결과를 낳는다 (레이어가 존재하되 phase가 0인 경우, 각 레이어 면에서의 회절이 여전히 발생한다).

> [!important]- TK-27. Phase parameterization: wrapped vs unwrapped
>
> **논문 서술:** 학습 가능한 위상 레이어를 사용한다고 기술한다.
>
> **실제 구현:** `phase_parameterization: wrapped_phase`로 설정되어 있지만, 실제 `PhaseLayer`는 `nn.Parameter`로 raw float32 값을 저장하고, `exp(j * phase)`를 적용할 때 자연스럽게 wrapping된다:
> ```python
> t_m = torch.exp(1j * self.phase.to(field.dtype))
> ```
> Phase 값은 $[0, 2\pi)$로 초기화되지만, 학습 중 제한 없이 성장할 수 있다 (e.g., $-3\pi$ or $+5\pi$).
>
> **물리적 근거:** $e^{j\phi}$는 $2\pi$ 주기이므로 어떤 실수값이든 유효한 위상이 된다. 하지만 gradient 전파 시 $\phi$가 매우 크면 학습률에 대한 실효 감도가 달라진다.
>
> > [!tip] 재현자를 위한 교훈
> > 학습 후 위상을 분석할 때는 반드시 `phase % (2*pi)`로 wrapping해야 한다. Raw 값은 $10\pi$ 이상일 수 있으며, 이는 물리적으로 의미가 없다.

> [!important]- TK-28. Resize pipeline의 3단계 구조
>
> **논문 서술:** MNIST 28$\times$28을 사용한다고만 기술한다.
>
> **실제 구현:** 3단계 resize가 필요하다:
> 1. $28 \to 160$ (bilinear interpolation): 원본 해상도를 물리 그리드에 맞춤
> 2. $160 \to 240$ (zero-padding): 전파에 필요한 여유 공간 확보
> 3. $240 \to 480$ (BL-ASM 내부 padding): aliasing 방지
>
> 단계 1의 resize 크기(160)와 단계 2의 최종 크기(240)의 차이가 곧 "active region"과 "guard band"를 정의한다. PCC 계산과 Fig.2 display는 중앙 160$\times$160 영역에서만 수행된다.
>
> > [!tip] 재현자를 위한 교훈
> > 160$\times$160 active region 밖에서의 intensity는 평가에서 무시된다. 이 crop을 생략하면 PCC가 ~0.05 낮게 나온다 (guard band의 어두운 영역이 correlation을 희석시키므로).

> [!important]- TK-29. "Blind" 평가의 시드 분리
>
> **논문 서술:** "new diffusers"로 평가한다고 기술한다.
>
> **실제 구현:** Blind 평가의 시드는 `77777777 + i`로, training 시드 체계(`base_seed + epoch * 1000 + i`)와 완전히 분리된다. 이 분리가 없으면 "new" 디퓨저가 실제로는 학습에 사용된 것일 수 있다.
>
> > [!tip] 재현자를 위한 교훈
> > 시드 분리를 확인하지 않고 논문을 재현하면, known과 new의 PCC 차이가 비정상적으로 작아지는 원인 불명의 버그를 만날 수 있다.

> [!important]- TK-30. FigS5의 correlation length 변경 방법
>
> **논문 서술:** "다른 correlation length의 diffuser로 테스트한다"고 기술한다.
>
> **실제 구현:** $L \approx 5\lambda$를 얻기 위해 `smoothing_sigma_lambda=2.0`을 사용한다 ($L \approx 2.5 \times 2.0\lambda = 5\lambda$). Height map의 다른 파라미터(mean, std, $\Delta n$)는 동일하게 유지한다.
>
> **물리적 근거:** Correlation length만 바꾸면 산란의 "angular spread"가 변한다. $L$이 작으면 고주파 위상 변동이 강해져 빛이 더 넓은 각도로 산란된다. D2NN이 이러한 분포 외 산란에 대해서도 작동하는지가 FigS5의 핵심 질문이다.
>
> > [!tip] 재현자를 위한 교훈
> > `smoothing_sigma_lambda`만 바꾸되 나머지를 고정하는 것이 공정한 비교의 조건이다. $\Delta n$까지 바꾸면 phase dynamic range와 correlation length가 동시에 변하여 해석이 불가능해진다.

---

## 6.7 종합: 암묵지 의존도 매핑

아래 표는 각 암묵지 항목이 영향을 미치는 재현 결과물을 정리한 것이다. ●는 해당 TK 항목이 해당 결과물의 재현에 직접적으로 영향을 미침을 의미한다.

| 암묵지 | Fig.2 | Fig.3 | FigS4 | FigS5 | Training |
|--------|:-----:|:-----:|:-----:|:-----:|:--------:|
| TK-1 (Scalar) | ● | ● | ● | ● | ● |
| TK-2 (Coherent) | ● | ● | ● | ● | ● |
| TK-3 (Evanescent) | ● | ● | ● | ● | ● |
| TK-4 (비대칭 거리) | ● | ● | ● | ● | ● |
| TK-5 (Zero-padding) | ● | ● | ● | ● | ● |
| TK-6 (fftfreq 순서) | ● | ● | ● | ● | ● |
| TK-7 (f64 정밀도) | | | | | ● |
| TK-8 (Reflect pad) | | | | ● | ● |
| TK-9 (TF 캐싱) | | | | | ● |
| TK-10 (B×n 배치) | | | | | ● |
| TK-11 (Epoch 갱신) | | | | | ● |
| TK-12 (시드 체계) | ● | ● | ● | ● | ● |
| TK-13 (Loss 구조) | | | | | ● |
| TK-14 (전역 PCC) | ● | ● | ● | ● | ● |
| TK-15 (LR schedule) | | | | | ● |
| TK-16 (Height 통계) | ● | ● | ● | ● | ● |
| TK-17 (Sigma-L 관계) | | | | ● | ● |
| TK-18 (Fitting 전략) | | | | ● | |
| TK-19 (n-sweep 포화) | | | | | ● |
| TK-20 (Known vs New) | ● | | | | |
| TK-21 (Period 추정) | | ● | | | |
| TK-22 (Contrast) | ● | | ● | ● | |
| TK-23 (Amplitude) | ● | ● | ● | ● | ● |
| TK-24 (n 이중 의미) | ● | ● | | | ● |
| TK-25 (Resolution) | | ● | | | |
| TK-26 (Pruning) | | | ● | | |
| TK-27 (Phase param) | | | ● | | ● |
| TK-28 (Resize 3단계) | ● | ● | ● | ● | ● |
| TK-29 (Blind 시드) | ● | ● | ● | ● | |
| TK-30 (L 변경 방법) | | | | ● | |

> [!quote] 요약
> 본 섹션에서 정리한 30개의 암묵지 항목은 Luo et al. 2022 논문의 성공적 재현에 필수적이지만, 논문 본문에는 명시되어 있지 않거나 모호하게 서술된 것들이다. 이 중 약 절반은 물리적 근거가 분명하여 "당연히 그래야 하는" 것이고, 나머지 절반은 시행착오를 통해서만 발견할 수 있는 수치적/구현적 선택이다. 재현자는 이 목록을 체크리스트로 활용하여 구현 초기 단계에서의 디버깅 시간을 크게 줄일 수 있을 것이다.

---

[^1]: 모든 TK 항목의 코드 참조는 `luo2022_random_diffusers_d2nn/src/` 디렉토리 기준이다.
