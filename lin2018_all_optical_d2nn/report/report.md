---
title: |
  Diffractive Deep Neural Networks (D$^2$NN)을 이용한\
  전광학 기계학습 시뮬레이션 재현
subtitle: "Lin et al. (*Science*, 2018) 논문 재현 보고서"
date: 2026-02-27
lang: ko
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Noto Sans CJK KR"
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{graphicx}
  - \usepackage{caption}
---

\newpage

# 초록

본 보고서는 Lin et al.이 *Science* (2018)에 발표한 "All-optical machine learning using diffractive deep neural networks" 논문의 핵심 결과를 수치 시뮬레이션으로 재현한 내용을 기술한다. Diffractive Deep Neural Network (D$^2$NN)은 빛의 회절 현상을 이용하여 기존 전자식 신경망의 연산을 광학적으로 수행하는 물리적 신경망 구조이다.

본 재현 실험에서는 Angular Spectrum Method (ASM)를 기반으로 한 파동 전파 시뮬레이션을 구현하고, MNIST 손글씨 숫자 분류(검증 정확도 **96.9%**), Fashion-MNIST 의류 분류(검증 정확도 **78.4%**), 그리고 ImageNette 이미지를 활용한 회절 이미징 렌즈(SSIM **0.734**) 실험을 수행하였다. 모든 실험은 NVIDIA A100-SXM4-40GB GPU에서 PyTorch 기반으로 수행되었으며, 결정론적(deterministic) 모드를 통해 완전한 재현성을 보장한다.

\newpage

# 서론

## 배경

전통적인 심층 신경망(Deep Neural Network)은 전자 회로에서 행렬 곱셈과 비선형 활성화를 반복 수행하여 입력 데이터를 분류하거나 변환한다. 이 과정은 본질적으로 디지털 연산에 의존하며, 에너지 소비와 연산 속도의 물리적 한계가 존재한다.

D$^2$NN은 이러한 한계를 극복하기 위해 **빛의 회절(diffraction)**을 연산 메커니즘으로 활용하는 새로운 패러다임을 제시한다. D$^2$NN에서 각 "레이어"는 물리적인 회절판(diffractive layer)이며, 각 픽셀은 입사하는 빛의 위상을 독립적으로 변조한다. 레이어를 통과한 빛은 자유공간에서 회절하며 다음 레이어로 전파되고, 이 과정이 반복되면서 입력 광학 신호가 변환된다. 최종 출력면에서 검출기가 광 에너지를 측정하여 분류 결과를 얻는다.

이 구조의 핵심적 장점은 다음과 같다:

1. **빛의 속도로 연산**: 추론(inference)이 빛의 전파 시간 내에 완료됨
2. **에너지 효율**: 연산에 전자적 에너지가 불필요하며, 광원의 에너지만으로 동작
3. **병렬 처리**: 2차원 광학 필드의 모든 픽셀이 동시에 처리됨

## 재현 동기

Lin et al. (2018) 논문은 0.4 THz ($\lambda = 0.75$ mm) 대역에서 3D 프린팅된 회절판으로 MNIST 숫자 분류 91.75%, Fashion-MNIST 의류 분류, 그리고 이미징 렌즈 기능을 실험적으로 시연하였다. 본 보고서에서는 논문의 물리적 파라미터를 그대로 사용하여 수치 시뮬레이션을 수행하고, 논문 결과와의 정합성을 검증한다.

\newpage

# 이론적 배경

## Angular Spectrum Method (ASM)

D$^2$NN에서 레이어 간 빛의 전파는 **Angular Spectrum Method** (각 스펙트럼 방법)으로 계산된다. 이 방법은 임의의 단색 파동장(monochromatic wave field)을 평면파(plane wave)의 중첩으로 분해한 뒤, 각 평면파 성분이 거리 $z$만큼 전파할 때 겪는 위상 지연을 독립적으로 계산하고, 다시 합성하는 기법이다.

### 물리적 직관

공간에서 전파하는 빛을 다양한 방향으로 진행하는 평면파의 합으로 생각해보자. 각 평면파는 공간 주파수 $(f_x, f_y)$로 특징지어지며, 이는 해당 평면파가 $x$, $y$ 방향으로 얼마나 빠르게 진동하는지를 나타낸다. 거리 $z$를 전파하면, 각 평면파는 자신의 방향에 따라 고유한 위상 지연을 겪는다. 이것이 회절의 본질이다.

### 수학적 정의

전기장 $E(x, y)$가 거리 $z$를 전파한 후의 결과 $E'(x, y)$는 다음과 같이 계산된다:

$$E'(x, y) = \mathcal{F}^{-1}\!\Big[\mathcal{F}\big[E(x,y)\big] \cdot H(f_x, f_y)\Big]$$

여기서 $\mathcal{F}$는 2차원 푸리에 변환, $\mathcal{F}^{-1}$은 역변환이며, **전달 함수(transfer function)** $H$는:

$$H(f_x, f_y) = \exp\!\left(i\,2\pi z \sqrt{\left(\frac{n}{\lambda}\right)^{\!2} - f_x^2 - f_y^2}\right)$$

각 기호의 의미:

- $\lambda$: 파장 (본 시뮬레이션에서 0.75 mm)
- $n$: 매질의 굴절률 (공기, $n = 1$)
- $z$: 전파 거리
- $(f_x, f_y)$: 공간 주파수

### 전파(propagating) 모드와 소멸(evanescent) 모드

제곱근 내부의 값에 따라 두 가지 물리적 상황이 구분된다:

- $\left(\frac{n}{\lambda}\right)^2 > f_x^2 + f_y^2$인 경우: **전파 모드**. 해당 주파수 성분은 실수 위상 지연을 가지며, 에너지 손실 없이 전파된다.
- $\left(\frac{n}{\lambda}\right)^2 < f_x^2 + f_y^2$인 경우: **소멸(evanescent) 모드**. 제곱근이 순허수가 되어 해당 성분은 지수적으로 감쇠한다. 본 구현에서는 **Band-Limited ASM**을 사용하여 이러한 고주파 성분을 제거함으로써 aliasing artifact를 방지한다.

### 계산 절차

실제 수치 계산은 FFT를 활용하여 효율적으로 수행된다:

1. 입력 필드를 2D FFT로 공간 주파수 영역으로 변환
2. 전달 함수 $H$를 곱하여 위상 지연 적용
3. 역 FFT로 공간 영역으로 복원

이 3단계 과정이 하나의 "전파(propagation)" 연산에 해당하며, 계산 복잡도는 $O(N^2 \log N)$이다.

## 회절 레이어 (Diffractive Layer)

### 물리적 구조

각 회절 레이어는 $N \times N$ 격자의 픽셀로 구성되며, 각 픽셀은 고유한 **위상 변조값** $\phi_l(x, y)$를 가진다. 물리적으로 이는 3D 프린팅된 투명 재료의 두께(높이) 차이로 구현되며, 빛이 서로 다른 두께의 재료를 통과할 때 경험하는 광경로차(optical path difference)가 위상 차이를 만든다.

### 복소 투과 함수

레이어 $l$의 복소 투과 함수(complex transmission function)는:

$$t_l(x, y) = A_l(x, y) \cdot e^{i\phi_l(x, y)}$$

여기서:

- $A_l(x, y) \in [0, 1]$: 진폭 투과율 (phase-only 모드에서는 $A_l = 1$)
- $\phi_l(x, y)$: 위상 변조 (학습 대상 파라미터)

### 위상 제약 조건

물리적으로 실현 가능한 위상 범위를 보장하기 위해 **symmetric tanh** 제약을 적용한다:

$$\phi_l = \frac{\phi_{\max}}{2} \cdot \tanh(\theta_l)$$

여기서 $\theta_l$은 제약이 없는 원시(raw) 파라미터이고, $\phi_{\max} = 2\pi$이다. tanh 함수를 사용함으로써 gradient가 끊기지 않으면서 위상이 유한 범위 $[-\pi, \pi]$에 제한된다.

### Forward 연산

각 DiffractionLayer의 forward 연산은 다음 두 단계로 구성된다:

1. **전파**: 입사 필드를 거리 $z$만큼 ASM으로 전파
2. **변조**: 전파된 필드에 투과 함수를 곱함

$$E_{l}^{\text{out}}(x,y) = \underbrace{\text{ASM}\big(E_{l-1}^{\text{out}},\; z\big)}_{\text{전파}} \;\cdot\; \underbrace{t_l(x,y)}_{\text{변조}}$$

물리적으로 이는 "이전 레이어에서 나온 빛이 자유공간을 거리 $z$만큼 회절하며 전파된 후, 현재 레이어의 투과 마스크를 통과한다"는 과정을 기술한다.

## 전체 D$^2$NN 모델

### 순전파 (Forward Model)

5개의 회절 레이어와 최종 출력 전파로 구성된 전체 모델의 순전파 과정은:

$$E_{\text{in}} \xrightarrow{\text{ASM}(z)} \odot\, t_1 \xrightarrow{\text{ASM}(z)} \odot\, t_2 \xrightarrow{\text{ASM}(z)} \odot\, t_3 \xrightarrow{\text{ASM}(z)} \odot\, t_4 \xrightarrow{\text{ASM}(z)} \odot\, t_5 \xrightarrow{\text{ASM}(z_{\text{out}})} E_{\text{out}}$$

여기서 $\odot$는 element-wise 곱셈(Hadamard product)을 나타낸다.

최종 출력면에서의 **강도 분포(intensity)**는:

$$I(x, y) = |E_{\text{out}}(x, y)|^2$$

### 학습 가능 파라미터

위상 변조 $\phi_l(x,y)$의 원시 파라미터 $\theta_l$이 역전파(backpropagation)를 통해 학습된다:

- 5개 레이어 $\times$ $N^2$개 위상 값 = **200,000개 파라미터** ($N = 200$ 기준)
- Phase-only 모드: 진폭 $A_l = 1$로 고정

## 입력 인코딩

### MNIST: 진폭 인코딩 (Amplitude Encoding)

MNIST 손글씨 숫자 이미지를 D$^2$NN에 입력하기 위해, 이미지 밝기를 전기장의 **진폭**으로 변환한다:

$$E_0(x, y) = A(x, y) \cdot e^{i \cdot 0} = A(x, y)$$

여기서 $A(x,y) \in \{0, 1\}$ (이진화). 이는 논문의 실험에서 알루미늄 포일을 잘라 binary transmittance mask를 만든 것과 물리적으로 대응된다. $28 \times 28$ 이미지를 $80 \times 80$ 픽셀로 확대한 후, $200 \times 200$ 격자 중앙에 배치한다.

### Fashion-MNIST: 위상 인코딩 (Phase Encoding)

Fashion-MNIST 이미지는 전기장의 **위상**으로 인코딩된다:

$$E_0(x, y) = 1 \cdot e^{i \cdot 2\pi \cdot g(x,y)}$$

여기서 $g(x,y) \in [0, 1]$은 정규화된 그레이스케일 값이다. 진폭은 균일하게 1이며, 이미지 정보는 위상에만 담겨 있다. 이 방식은 SLM(Spatial Light Modulator)을 통한 위상 변조 실험에 대응된다.

## 검출기 분류 (Detector Classification)

출력면에 10개의 검출기 영역 $R_0, R_1, \ldots, R_9$를 정의한다. 각 영역은 $8 \times 8$ mm 크기이며, $80 \times 80$ mm 출력면 위에 3-4-3 패턴으로 배치된다.

각 검출기 $k$의 광 에너지는:

$$P_k = \iint_{R_k} |E_{\text{out}}(x, y)|^2 \, dA \approx \sum_{(i,j) \in R_k} I_{i,j} \cdot (\Delta x)^2$$

분류 결과는 최대 에너지를 가진 검출기의 인덱스로 결정된다:

$$\hat{y} = \arg\max_k \; P_k$$

이는 논문의 "maximum optical signal" 기준과 정확히 일치한다.

## 손실 함수

### 분류 손실 (Classification Loss)

검출기 에너지 벡터를 softmax logit으로 사용하는 교차 엔트로피(cross-entropy) 손실:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=0}^{9} y_k \log\left(\frac{e^{P_k / \tau}}{\sum_{j=0}^{9} e^{P_j / \tau}}\right)$$

여기서 $y_k$는 one-hot 인코딩된 정답 레이블, $\tau$는 temperature 파라미터 (기본값 1.0)이다.

추가적으로 검출기 영역 밖으로 새어나가는 에너지를 억제하는 **누출 페널티(leakage penalty)**:

$$\mathcal{L}_{\text{leak}} = 1 - \frac{\sum_{(i,j) \in \bigcup_k R_k} I_{i,j}}{\sum_{i,j} I_{i,j}}$$

전체 손실:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{leak}}, \quad \alpha = 0.1$$

### 이미징 손실 (Imaging Loss)

출력 강도와 목표 강도 간의 평균 제곱 오차(MSE):

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N^2} \sum_{i,j} \left(\hat{I}_{i,j} - I^{\text{target}}_{i,j}\right)^2$$

여기서 $\hat{I}$와 $I^{\text{target}}$은 각각 최댓값으로 정규화된 출력 및 목표 강도이다.

## 위상-높이 변환 (Phase-to-Height)

학습된 위상 마스크를 물리적 구조물로 제작하기 위해, 위상을 재료 두께로 변환한다:

$$\Delta z = \frac{\lambda}{2\pi} \cdot \frac{\phi}{\Delta n}$$

여기서 $\Delta n = n_{\text{material}} - n_{\text{air}}$이다. 논문에서 사용한 VeroBlackPlus RGD875 재료의 경우 $\Delta n = 0.7227$ (0.4 THz 대역)이다.

\newpage

# 시뮬레이션 설정

## 물리 파라미터

Table 1은 분류기와 이미징 렌즈 실험에 사용된 물리 파라미터를 정리한 것이다. 모든 값은 Lin et al. (2018) 논문에서 명시한 값과 일치한다.

| **파라미터** | **분류기 (MNIST/Fashion)** | **이미징 렌즈** | **단위** | **물리적 의미** |
|:---|:---:|:---:|:---:|:---|
| $\lambda$ | 0.75 | 0.75 | mm | THz 대역 파장 (0.4 THz) |
| $N$ | 200 | 300 | px | 격자 해상도 |
| $\Delta x$ | 0.4 | 0.3 | mm | 픽셀 간격 (피치) |
| 개구 크기 $(N \times \Delta x)$ | 80 | 90 | mm | 레이어의 물리적 크기 |
| 레이어 간격 $z$ | 30 | 4 | mm | 인접 레이어 간 전파 거리 |
| 출력 거리 $z_{\text{out}}$ | 30 | 7 | mm | 마지막 레이어 $\rightarrow$ 검출면 |
| 회절 레이어 수 | 5 | 5 | -- | 학습 가능한 위상 변조 층 수 |
| $\phi_{\max}$ | $2\pi$ | $2\pi$ | rad | 최대 위상 변조 범위 |
| 굴절률 $n$ | 1.0 | 1.0 | -- | 매질 (공기) |
| Band-limit | 적용 | 적용 | -- | Evanescent 모드 제거 |
| 학습 파라미터 수 | 200,000 | 450,000 | -- | $N^2 \times 5$ 레이어 |

: Table 1. D$^2$NN 물리 파라미터 비교

## 학습 설정

| **항목** | **MNIST** | **Fashion-MNIST** | **이미징 렌즈** |
|:---|:---:|:---:|:---:|
| Epochs | 10 | 10 | 36 |
| Batch size | 128 | 128 | 64 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Optimizer | Adam | Adam | Adam |
| 손실 함수 | CE + Leakage | CE + Leakage | MSE |
| Leakage weight $\alpha$ | 0.1 | 0.1 | -- |
| Temperature $\tau$ | 1.0 | 1.0 | -- |
| 입력 인코딩 | 진폭 (이진화) | 위상 $[0, 2\pi]$ | 진폭 |
| 입력 크기 | 80 px | 80 px | 140 px |
| 위상 제약 | symmetric\_tanh | symmetric\_tanh | symmetric\_tanh |
| 위상 초기값 | zeros | zeros | zeros |
| Random seed | 1234 | 1234 | 1234 |

: Table 2. 학습 하이퍼파라미터

## 학습 환경

| **항목** | **상세** |
|:---|:---|
| GPU | NVIDIA A100-SXM4-40GB |
| Framework | PyTorch $\geq$ 2.2 |
| Python | $\geq$ 3.10 |
| Deterministic 모드 | 활성화 (완전 재현성 보장) |
| DataLoader workers | 8 (persistent, pinned memory) |
| 정밀도 | complex64 (float32 복소수) |

: Table 3. 학습 환경

\newpage

# 결과

## MNIST 손글씨 숫자 분류

MNIST 데이터셋(60,000 학습 / 10,000 검증)을 사용하여 5-layer phase-only D$^2$NN 분류기를 학습시켰다. $28 \times 28$ 손글씨 숫자 이미지를 이진화된 진폭 마스크로 인코딩하여 입력하였다.

### 학습 결과

- **학습 정확도**: 98.3%
- **검증 정확도**: 96.9%
- **학습 손실 (최종)**: 0.149
- **검증 손실 (최종)**: 0.184

이는 논문의 실험 결과(91.75%)보다 높은 수치인데, 시뮬레이션에서는 물리적 실험에서 발생하는 레이어 정렬 오차(misalignment), 재료 불완전성, 표면 거칠기, 측정 노이즈 등이 부재하기 때문이다.

### 학습 곡선

![MNIST 분류기 학습 곡선. 10 epoch에 걸친 학습/검증 손실 및 정확도의 변화. 학습 초기에 급격한 정확도 상승 후 안정적으로 수렴하는 전형적인 학습 패턴을 보인다.](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/training_curves.png){ width=90% }

### 혼동 행렬

![MNIST 혼동 행렬 (절대 개수). 대각선 성분이 지배적이며, 대부분의 숫자가 정확하게 분류됨을 보여준다.](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/confusion_matrix_counts.png){ width=75% }

![MNIST 혼동 행렬 (정규화). 각 클래스별 분류 정확도를 비율로 표시. 대부분 95% 이상의 클래스별 정확도를 달성.](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/confusion_matrix_normalized.png){ width=75% }

### 검출기 에너지 분포

![검출기별 에너지 분포 히트맵. 각 행은 입력 클래스(0--9), 각 열은 검출기 번호. 대각선 방향으로 에너지가 집중되어 있어, D$^2$NN이 각 숫자에 해당하는 검출기로 빛을 효과적으로 집중시키고 있음을 확인할 수 있다.](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/energy_distribution_heatmap.png){ width=75% }

### 학습된 위상 마스크

5개 레이어의 학습된 위상 분포는 각 레이어가 서로 다른 공간 주파수 패턴을 형성하고 있음을 보여준다. 초기 레이어(L1--L2)는 비교적 완만한 저주파 패턴을, 후기 레이어(L4--L5)는 미세한 고주파 구조를 가지는 경향이 있다.

![MNIST -- Layer 1 위상 마스크](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/phase_layer_1.png){ width=48% }
![MNIST -- Layer 2 위상 마스크](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/phase_layer_2.png){ width=48% }

![MNIST -- Layer 3 위상 마스크](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/phase_layer_3.png){ width=48% }
![MNIST -- Layer 4 위상 마스크](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/phase_layer_4.png){ width=48% }

![MNIST -- Layer 5 위상 마스크](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/phase_layer_5.png){ width=48% }

### 출력면 및 추론 예시

![출력면 강도 분포와 검출기 영역. 10개의 검출기 영역(빨간 사각형)이 출력면 위에 표시되어 있으며, D$^2$NN이 입력 숫자에 해당하는 검출기 위치에 빛 에너지를 집중시키는 것을 관찰할 수 있다.](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/sample_output_with_detectors.png){ width=85% }

![MNIST 추론 요약. 여러 테스트 샘플에 대한 입력 이미지, 출력 강도 분포, 예측 결과를 한눈에 보여주는 종합 그림.](../runs/mnist_phase_only_5l_a100/be19548ba636/figures/sample_inference_summary.png){ width=90% }

\newpage

## Fashion-MNIST 의류 분류

Fashion-MNIST 데이터셋(60,000 학습 / 10,000 검증)을 사용하여 동일한 5-layer D$^2$NN 구조로 10종 의류 분류를 수행하였다. MNIST와 달리 **위상 인코딩**을 사용하여 그레이스케일 이미지를 $[0, 2\pi]$ 범위의 위상으로 변환한다.

### 학습 결과

- **학습 정확도**: 83.4%
- **검증 정확도**: 78.4%
- **학습 손실 (최종)**: 1.174
- **검증 손실 (최종)**: 1.787

Fashion-MNIST는 MNIST보다 시각적으로 복잡하고 클래스 간 유사성이 높아(예: T-shirt vs Shirt, Pullover vs Coat), 분류 난이도가 상당히 높다.

### 학습 곡선

![Fashion-MNIST 분류기 학습 곡선. MNIST에 비해 수렴이 느리고 학습/검증 손실 간 격차가 큰 것은 데이터셋의 높은 복잡도를 반영한다.](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/training_curves.png){ width=90% }

### 혼동 행렬

![Fashion-MNIST 혼동 행렬 (절대 개수).](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/confusion_matrix_counts.png){ width=75% }

![Fashion-MNIST 혼동 행렬 (정규화). 시각적으로 유사한 클래스(예: Shirt, T-shirt, Pullover) 간 혼동이 두드러짐.](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/confusion_matrix_normalized.png){ width=75% }

### 검출기 에너지 분포

![Fashion-MNIST 검출기별 에너지 분포. MNIST에 비해 에너지가 더 분산되어 있어, 위상 인코딩 입력의 분류가 더 어려운 과제임을 시사한다.](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/energy_distribution_heatmap.png){ width=75% }

### 학습된 위상 마스크

![Fashion-MNIST -- Layer 1 위상 마스크](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/phase_layer_1.png){ width=48% }
![Fashion-MNIST -- Layer 2 위상 마스크](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/phase_layer_2.png){ width=48% }

![Fashion-MNIST -- Layer 3 위상 마스크](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/phase_layer_3.png){ width=48% }
![Fashion-MNIST -- Layer 4 위상 마스크](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/phase_layer_4.png){ width=48% }

![Fashion-MNIST -- Layer 5 위상 마스크](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/phase_layer_5.png){ width=48% }

### 출력면 및 추론 예시

![Fashion-MNIST 출력면 강도 분포와 검출기 영역.](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/sample_output_with_detectors.png){ width=85% }

![Fashion-MNIST 추론 요약. 위상 인코딩 입력에 대한 D$^2$NN의 분류 결과.](../runs/fashion_phase_only_5l_a100/bb8baf380982/figures/sample_inference_summary.png){ width=90% }

\newpage

## 이미징 렌즈 (Diffractive Imaging)

D$^2$NN은 분류 외에도 **광학 이미징 렌즈** 기능을 수행할 수 있다. ImageNette 데이터셋의 자연 이미지를 입력하여, 5개 회절 레이어를 통과한 후 출력면에 입력 이미지가 재구성되도록 학습시켰다. 이미징 실험에서는 격자 해상도 $N = 300$, 픽셀 간격 $\Delta x = 0.3$ mm, 레이어 간격 $z = 4$ mm로 설정하였다.

### 학습 결과

- **SSIM (D$^2$NN)**: $0.734 \pm 0.038$
- **SSIM (자유공간)**: $0.717 \pm 0.039$
- **D$^2$NN이 자유공간보다 우수한 비율**: 94.1% (전체 SSIM 기준)
- **학습 손실 (최종)**: 0.017
- **검증 손실 (최종)**: 0.018

D$^2$NN 이미징 렌즈는 단순 자유공간 전파(레이어 없음)보다 대부분의 이미지에서 더 높은 SSIM을 달성하여, 학습된 위상 마스크가 실질적인 이미징 기능을 수행하고 있음을 확인하였다.

### 학습 곡선

![이미징 렌즈 학습 곡선. 36 epoch에 걸쳐 MSE 손실이 안정적으로 감소.](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/training_curve.png){ width=85% }

### 이미징 결과 비교

![이미징 결과 비교. 왼쪽부터 입력 이미지, D$^2$NN 출력, 자유공간(레이어 없음) 출력을 보여준다. D$^2$NN이 자유공간보다 더 선명한 이미지를 재구성함.](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/imaging_comparison.png){ width=95% }

### 학습된 위상 마스크

![이미징 -- Layer 1 위상 마스크](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/phase_layer_1.png){ width=48% }
![이미징 -- Layer 2 위상 마스크](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/phase_layer_2.png){ width=48% }

![이미징 -- Layer 3 위상 마스크](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/phase_layer_3.png){ width=48% }
![이미징 -- Layer 4 위상 마스크](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/phase_layer_4.png){ width=48% }

![이미징 -- Layer 5 위상 마스크. 이미징 레이어의 위상 패턴은 분류기와 달리 동심원 형태의 렌즈 구조를 형성하는 경향이 있다.](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/phase_layer_5.png){ width=48% }

### 전파 과정 시각화

![층별 진폭 분포 (propagation stack). 입력에서 출력까지 각 레이어를 통과하면서 전기장 진폭이 어떻게 변화하는지 보여준다.](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/propagation_stack_amp.png){ width=95% }

![층별 위상 분포 (propagation stack). 각 레이어에서의 위상 변조 효과를 시각화.](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/propagation_stack_phase.png){ width=95% }

![xz 평면 진폭 전파. 수직 단면에서 빛이 레이어를 통과하며 집속되는 과정을 보여준다. 각 레이어 위치에서 진폭의 급격한 변화가 관찰된다.](../runs/imaging_lens_imagenette_large_a100_n300/260218_154328/figures/propagation_xz_amp.png){ width=95% }

\newpage

## 파동 전파 시각화

학습된 D$^2$NN과 자유공간(레이어 없음)의 파동 전파를 비교하는 상세 시각화를 수행하였다. 이 분석을 통해 회절 레이어가 빛의 전파에 미치는 구체적인 영향을 관찰할 수 있다.

### 논문 Fig. S6 스타일 재현

![논문의 보충자료 Figure S6 스타일로 재현한 파동 전파 시각화. 입력면에서 출력면까지의 전기장 진폭과 위상 변화를 종합적으로 보여준다.](../runs/wave_panels_s6_final/figure_s6_style.png){ width=95% }

### D$^2$NN vs 자유공간 xz 단면 비교

![D$^2$NN xz 평면 진폭 전파. 5개의 회절 레이어가 빛을 적극적으로 재분배하여 출력면에서 원하는 패턴을 형성한다.](../runs/wave_panels_s6_final/d2nn_xz_amplitude.png){ width=95% }

![자유공간 xz 평면 진폭 전파 (레이어 없음). 회절 레이어가 없으면 빛이 단순히 확산되며 특정 패턴을 형성하지 못한다.](../runs/wave_panels_s6_final/free_space_xz_amplitude.png){ width=95% }

### xy 평면 진폭/위상 비교

![D$^2$NN과 자유공간의 xy 평면 진폭 및 위상 비교 (stacked view). 각 레이어 위치에서의 필드 분포를 비교하여, 회절 레이어가 빛의 공간 분포를 어떻게 조작하는지 직관적으로 보여준다.](../runs/wave_panels_s6_final/stacked_xy_amp_phase_comparison.png){ width=95% }

### xz 진폭 비교 (나란히)

![xz 평면 진폭 비교. D$^2$NN(위)과 자유공간(아래)을 나란히 배치하여, 회절 레이어의 효과를 직접 비교할 수 있다. 레이어가 있는 경우 빛의 에너지가 특정 영역으로 집중되는 반면, 자유공간에서는 균일하게 확산됨.](../runs/wave_panels_s6_final/xz_amplitude_comparison.png){ width=95% }

\newpage

# 논의

## 논문 결과와의 비교

Table 4는 본 시뮬레이션 결과와 논문의 실험/시뮬레이션 결과를 비교한 것이다.

| **실험** | **본 시뮬레이션** | **논문 (실험)** | **논문 (시뮬레이션)** |
|:---|:---:|:---:|:---:|
| MNIST 분류 정확도 | **96.9%** | 91.75% | 93--98% |
| Fashion-MNIST 분류 정확도 | **78.4%** | -- | 약 81% |
| 이미징 (D2NN vs 자유공간) | 94.1% 우수 | 정성적 비교 | 정성적 비교 |

: Table 4. 논문 결과 대비 비교

### MNIST: 시뮬레이션이 실험보다 높은 이유

본 시뮬레이션의 MNIST 분류 정확도(96.9%)가 논문의 실험 정확도(91.75%)보다 높은 것은 다음과 같은 이유로 설명된다:

1. **물리적 불완전성 부재**: 시뮬레이션에서는 3D 프린팅 오차, 표면 거칠기, 재료 비균질성이 존재하지 않음
2. **정렬 오차 부재**: 실제 실험에서는 레이어 간 정렬이 완벽하지 않으나, 시뮬레이션에서는 완벽한 정렬 가정
3. **측정 노이즈 부재**: 실험에서의 검출기 노이즈, 배경 복사 등이 부재
4. **이상적인 입력**: 시뮬레이션의 이진화된 입력은 완벽한 0/1 투과율을 가정

### Fashion-MNIST: 논문보다 약간 낮은 이유

검증 정확도(78.4%)가 논문의 시뮬레이션 결과(약 81%)보다 다소 낮은 것은 위상 제약 방식(symmetric\_tanh vs 논문의 미상), 학습 epoch 수, batch size 등 학습 하이퍼파라미터의 차이에 기인할 수 있다. 더 긴 학습이나 하이퍼파라미터 튜닝을 통해 격차를 줄일 수 있을 것으로 예상된다.

### 이미징: D$^2$NN의 효과

이미징 실험에서 D$^2$NN은 94.1%의 테스트 이미지에서 자유공간보다 높은 SSIM을 달성하였다. 이는 학습된 위상 마스크가 회절 렌즈 역할을 수행하여, 단순 자유공간 전파에서 발생하는 빛의 확산을 보상하고 이미지를 집속(focusing)시키는 효과를 가짐을 의미한다.

## 물리적 타당성 검증

구현의 물리적 정확성을 검증하기 위해 다음 테스트를 수행하였다:

1. **ASM 항등성 테스트**: $z = 0$에서 전파 시 입력 필드가 변하지 않음을 확인
2. **에너지 보존 테스트**: Band-limit 미적용 시 전파 전후 총 에너지의 상대 오차 $< 0.1\%$
3. **위상 제약 범위 테스트**: 학습된 위상이 $[-\pi, \pi]$ 범위 내에 있음을 확인
4. **재현성 테스트**: 동일 seed에서 완전히 동일한 결과를 생성함을 확인

\newpage

# 결론

본 보고서에서는 Lin et al. (2018)의 Diffractive Deep Neural Network (D$^2$NN) 논문을 Angular Spectrum Method 기반의 수치 시뮬레이션으로 재현하였다.

**주요 성과:**

1. **MNIST 분류**: 5-layer phase-only D$^2$NN으로 96.9%의 검증 정확도를 달성하여, 논문의 실험 결과(91.75%)를 상회하는 시뮬레이션 성능을 확인
2. **Fashion-MNIST 분류**: 위상 인코딩 입력을 사용하여 78.4%의 검증 정확도 달성
3. **이미징 렌즈**: 학습된 D$^2$NN이 94.1%의 이미지에서 자유공간 대비 우수한 SSIM을 보여, 회절 레이어가 효과적인 광학 이미징 기능을 수행함을 확인
4. **파동 전파 시각화**: D$^2$NN과 자유공간의 전파 과정을 상세히 비교하여, 회절 레이어가 빛의 공간 분포를 적극적으로 조작하는 메커니즘을 시각적으로 확인

본 구현은 논문의 핵심 물리 파라미터($\lambda = 0.75$ mm, 5개 레이어, 30/4 mm 간격 등)를 충실히 재현하였으며, ASM 전파, 위상 변조, 검출기 분류의 물리적 정확성을 단위 테스트를 통해 검증하였다.

---

**참고문헌**

1. X. Lin, Y. Rivenson, N. T. Yardimci, M. Veli, Y. Luo, M. Jarrahi, and A. Ozcan, "All-optical machine learning using diffractive deep neural networks," *Science* **361**, 1004--1008 (2018).
2. J. W. Goodman, *Introduction to Fourier Optics*, 4th ed. (W.H. Freeman, 2017).
