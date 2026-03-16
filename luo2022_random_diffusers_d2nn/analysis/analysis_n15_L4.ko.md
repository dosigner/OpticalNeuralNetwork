---
title: 실험 분석 - n15_L4 (고다양성 디퓨저, 포화 직전 구간)
aliases:
  - analysis_n15_L4 한국어
tags:
  - d2nn
  - luo2022
  - analysis
  - obsidian
  - korean
date: 2026-03-13
source_note: "[[analysis_n15_L4]]"
status: translated
---

# 실험 분석: n15_L4 (고다양성 디퓨저, 포화 직전 구간)

**Run ID:** `n15_L4`  
**Date:** `2026-03-13`  
**Reference:** Luo et al., "Computational Imaging Without a Computer: Diffractive Networks," *eLight* 2 (2022)

---

## 1. 실험 개요

> [!info] 실험 요약
> **모델**: 4-레이어 위상 전용 D2NN (240×240 그리드)  
> **디퓨저 수**: epoch당 15개 (높은 다양성)  
> **학습**: 30 epochs, B=64 (유효 배치 B×n=960), A100 TF32  
> **최종 PCC**: ==0.879== | **최종 Loss**: -0.953 | **소요 시간**: 79.5분

### 모델 설정

| 파라미터 | 값 |
|---|---|
| 아키텍처 | Phase-only D2NN (`d2nn_phase_only`) |
| 레이어 수 | 4 |
| 그리드 | 240 × 240 |
| 픽셀 피치 | 0.3 mm |
| 파장 | 0.75 mm (400 GHz, THz regime) |
| 전파 방식 | BL-ASM (Band-Limited Angular Spectrum Method) |
| 패딩 배율 | 2x (480 × 480 padded grid) |
| 위상 초기화 | Uniform [0, 2pi) |
| 학습 파라미터 수 | 230,400 (4 × 57,600) |

### 기하 구조

- Object-to-diffuser: 40.0 mm
- Diffuser-to-layer-1: 2.0 mm
- Layer-to-layer: 2.0 mm
- Last-layer-to-output: 7.0 mm

### 학습 파라미터

| 파라미터 | 값 |
|---|---|
| Epochs | 30 |
| Diffusers per epoch (n) | **15** |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| LR schedule | Multiplicative decay, gamma = 0.99/epoch |
| Loss function | -PCC + energy penalty (alpha=1.0, beta=0.5) |
| Hardware | NVIDIA A100 (TF32 enabled) |
| Dataset | MNIST (50k train / 10k val) |
| Input encoding | Amplitude (grayscale), resized 28 -> 160 -> 240 px |

### 디퓨저 모델

얇은 랜덤 위상 스크린:

- delta_n = 0.74, mean height = 25 lambda, std = 8 lambda
- Smoothing sigma = 4 lambda, correlation length = 10 lambda
- epoch마다 15개의 고유한 diffuser realization을 새로 샘플링

> [!note] 설정 파일 메모
> 저장된 [`config.yaml`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n15_L4/config.yaml)에는 `batch_size_objects: 4`가 기록되어 있습니다. 이 노트의 요약 표기는 원본 분석 노트의 배치 표기를 유지합니다.

---

## 2. 학습 동역학

### 수렴 요약

| 구간 | Epoch | PCC | 관찰 |
|---|---|---|---|
| 급속 상승 | 1→5 | 0.621 → 0.856 | 기본 산란 보정 학습 |
| 안정화 | 5→10 | 0.856 → 0.874 | 디퓨저 다양성에 적응 |
| 고원기 | 10→20 | 0.874 → 0.878 | 미세 조정 단계 |
| 최종 | 20→30 | 0.878 → 0.879 | 수렴 완료 |

**최종 지표:** loss = -0.9532, PCC = 0.8793  
**총 학습 시간:** 4,771.6 s (79.5 min), 약 159.1 s/epoch

### 수렴 곡선 해석

1. **급속 상승 (epochs 1-5):** PCC가 0.621에서 0.856으로 크게 상승합니다. 이 구간에서 네트워크는 디퓨저별 세부 차이보다, 여러 디퓨저에 공통적인 저주파 산란 보정을 먼저 학습합니다.

2. **감속 수렴 (epochs 5-10):** PCC 증가는 +0.018 수준으로 둔화됩니다. `n=10`보다 더 많은 디퓨저를 동시에 만족시켜야 하므로, 각 업데이트가 특정 디퓨저 하나에 특화되는 대신 앙상블 평균 방향으로 수렴합니다.

3. **긴 plateau (epochs 10-30):** 이후 20 epoch 동안 PCC 증가는 사실상 +0.005 정도에 머뭅니다. 이는 `n=15`가 이미 4-layer D2NN의 범용 산란 보정 표현력 한계 근방에 도달했음을 시사합니다.

> [!important] 핵심 관찰
> `n=15`는 `n=10`과 거의 같은 학습 PCC를 유지하면서도, 후반부 plateau가 더 길고 안정적입니다. 즉, 이 구간의 추가 계산은 "더 높은 training PCC"를 위한 것이 아니라 "더 넓은 diffuser distribution에 대한 강건성"을 위해 소비됩니다.

### n=1, n=10과의 비교

| 모델 | 최종 PCC (train) | 소요 시간 | epoch당 시간 |
|---|---|---|---|
| n1_L4 | **0.907** | 7.6분 | ~15초 |
| n10_L4 | **0.884** | 53.8분 | ~108초 |
| n15_L4 | **0.879** | 79.5분 | ~160초 |

`n=10 -> 15`로 갈 때 학습 PCC 감소폭은 약 0.0043에 그치지만, 전체 학습 시간은 25.7분 늘어납니다. 이 비대칭성 자체가 `n=15` 구간의 의미를 보여 줍니다. 여기서부터는 성능 지표의 중심이 training PCC에서 blind/generalization PCC로 이동합니다.

---

## 3. 논문 기대 결과와의 비교

### 논문 맥락 (Luo et al. 2022, Fig. 3)

논문의 핵심 메시지는 다음과 같습니다.

1. **디퓨저 수가 늘수록 training PCC는 감소한다.** 더 많은 산란 조건을 동시에 만족시켜야 하므로 최적화가 어려워집니다.

2. **대신 blind-test PCC는 향상된다.** 네트워크가 특정 디퓨저 inverse를 암기하는 대신, diffuser-invariant imaging 전략을 학습하기 시작하기 때문입니다.

3. **`n >= 10`부터는 수확 체감이 시작된다.** training PCC의 추가 하락은 작아지지만, 일반화 향상은 점진적으로 포화에 접근합니다.

### 우리의 n=15 결과와 논문의 비교

- **Training PCC = 0.879**는 `n=1` 기준선(0.907)보다 낮고 `n=10`(0.884)보다도 약간 낮습니다. 이는 디퓨저 다양성이 regularizer처럼 작동한다는 논문 설명과 정확히 맞아떨어집니다.

- **감소폭이 작다**는 점이 중요합니다. `n=1 -> 10`에서의 하락은 -0.023이지만, `n=10 -> 15`는 -0.005 수준입니다. 즉 `n=15`는 더 이상 "급격한 regime 전환"이 아니라 "포화 직전 미세 조정"에 가깝습니다.

- **학습 시간 증가가 거의 선형적**이라는 점도 논문 프로토콜과 부합합니다. epoch당 디퓨저 수가 50% 늘어나면서 wall-clock time 역시 108초에서 160초 수준으로 증가합니다.

> [!tip] 해석 포인트
> `n=15`는 training 관점에서는 `n=10`보다 크게 낫지 않아 보일 수 있습니다. 그러나 실제 논문의 핵심 평가지표는 blind diffuser에 대한 성능이며, 이 지점부터는 "얼마나 잘 외웠는가"보다 "새 산란 매질에 얼마나 덜 민감한가"가 더 중요합니다.

---

## 4. n=15에서 드러나는 특별한 인사이트

### 4.1 실질적인 Pareto knee

`fig5_memory.npy`와 `fig6_conditions.npy`에 저장된 평가 결과를 보면, `n=15`는 성능과 비용의 균형점에 가장 가깝습니다.

| 모델 | Train PCC | Recent/Blind PCC | New diffuser PCC | 총 시간 |
|---|---|---|---|---|
| n10_L4 | 0.8836 | 0.8684 / 0.8684 | 0.8731 | 53.8분 |
| n15_L4 | 0.8793 | 0.8719 / 0.8728 | **0.8784** | 79.5분 |
| n20_L4 | 0.8785 | 0.8714 / 0.8729 | 0.8750 | 105.7분 |

여기서 읽을 수 있는 핵심은 단순합니다.

- `n=10 -> 15`에서는 총 시간이 25.7분 늘지만, blind/new diffuser 성능은 실제로 개선됩니다.
- `n=15 -> 20`에서는 시간이 다시 26.2분 늘지만, blind PCC는 사실상 그대로이고 `new diffuser` PCC는 오히려 약간 낮아집니다.

즉 `n=15`는 4-layer 조건에서 **일반화 포화 직전의 Pareto knee**로 해석하는 것이 가장 자연스럽습니다.

### 4.2 새 디퓨저가 오히려 더 잘 보인다

`fig6` 기준으로 `n=15`의 known diffuser 평균 PCC는 `0.8596 ± 0.0378`, new diffuser 평균 PCC는 `0.8784 ± 0.0344`입니다.  
즉, ==새 디퓨저가 known diffuser보다 약 0.0188 높습니다.==

이 패턴은 중요한 의미를 갖습니다.

- 네트워크가 마지막 epoch의 특정 diffuser 집합을 암기하고 있지 않다는 뜻입니다.
- 최적화 목표가 "각 diffuser를 완벽히 복원"하는 것이 아니라 "diffuser distribution 전체에서 평균적으로 잘 작동"하는 위상 보정으로 이동했다는 증거입니다.
- 다시 말해 `n=15`는 diffuser-specific inverse에서 scattering-invariant transform으로 넘어간 뒤의 구간입니다.

### 4.3 분산까지 함께 줄어드는 구간

blind evaluation의 표준편차도 `n=10`의 `±0.0111`에서 `n=15`의 `±0.0090`으로 줄어듭니다. 절대 차이는 크지 않지만, 이 감소는 네트워크가 평균 PCC뿐 아니라 diffuser 간 변동성까지 줄이기 시작했음을 시사합니다.

이 점이 중요합니다. 실제 사용에서는 평균 성능만큼이나 diffuser가 바뀔 때 결과가 흔들리지 않는 것이 중요하기 때문입니다. `n=15`는 정확도와 안정성 두 축이 동시에 포화 구간에 진입하는 첫 지점으로 보입니다.

### 4.4 n=15 이후에는 표현력 한계가 더 중요해진다

`n=20`의 training/generalization 성능이 `n=15`와 거의 같다는 사실은, 이후 병목이 diffuser 수 부족이 아니라 **4-layer D2NN 자체의 표현력 한계**로 이동했음을 암시합니다.

- `n < 10`: 더 많은 diffuser를 보면 실제로 새로운 불변 특징을 더 많이 학습함
- `n ≈ 15`: 이미 충분한 diffuser 분포를 봤기 때문에 추가 샘플의 정보 이득이 작아짐
- `n > 15`: 같은 분포를 더 많이 재샘플링하는 효과가 커지고, 성능은 레이어 수와 모델 용량에 더 강하게 제한됨

이 관점에서 보면 다음 실험의 우선순위는 `n=20`을 더 늘리는 것보다, `L2/L5` depth sweep처럼 **모델 용량을 바꾸는 실험**에 가까워집니다.

---

## 5. 물리적 해석

### 왜 디퓨저 다양성이 중요한가

$$
t_d(\mathbf{r}) = e^{j\phi_d(\mathbf{r})}
$$

- **n=1:** 네트워크는 특정 $\phi_d(\mathbf{r})$에 대한 보정 연산을 사실상 암기합니다.
- **n=15:** 네트워크는 서로 다른 $\phi_d$들에 공통적인 산란 통계에 적응합니다. 개별 디퓨저의 세부 위상 구조를 따라가기보다, 여러 디퓨저에 반복적으로 나타나는 구조를 보정하는 쪽으로 학습이 이동합니다.

### 손실함수 관점

각 epoch에서 동일한 위상 레이어 $\phi_m$은 15개의 디퓨저에 대해 동시에 최적화됩니다.

$$
\mathcal{L} =
-\frac{1}{15B}\sum_{i=1}^{15}\sum_{j=1}^{B}
\text{PCC}(I_{ij}^{\text{out}}, I_j^{\text{target}})
+ \alpha \cdot \mathcal{E}_{\text{penalty}}
$$

이 손실은 다음 두 효과를 만듭니다.

- 특정 diffuser realization에 대한 과적합 억제
- 서로 다른 산란 조건에서 공통적으로 유효한 위상 패턴 선호

결국 `n=15`는 "어떤 diffuser가 오더라도 무너지지 않는 평균 inverse"를 학습하는 구간으로 해석할 수 있습니다.

---

## 6. n-sweep 재현에서의 의미

### Sweep 안에서의 위치

| Sweep point | Status | Training PCC | 핵심 의미 |
|---|---|---|---|
| n=1 | Complete | 0.907 | 단일 diffuser 암기 기준선 |
| n=10 | Complete | 0.884 | 일반화 onset |
| **n=15** | **Complete** | **0.879** | **포화 직전의 효율적 sweet spot** |
| n=20 | Complete | 0.878 | 성능 포화 확인 |

### Fig. 3/5/6 관점의 해석

1. **Fig. 3:** training PCC는 `0.907 -> 0.884 -> 0.879 -> 0.878`로 감소하며, `n >= 15`에서 사실상 포화됩니다.
2. **Fig. 5:** recent/blind PCC는 `n=15`에서 이미 `0.872`대에 도달하며 `n=20`과 거의 구별되지 않습니다.
3. **Fig. 6:** `known < new` 패턴이 `n=15`에서 가장 뚜렷하게 나타나며, 이는 기억보다 일반화가 우세한 regime임을 보여 줍니다.

> [!success] n=15의 위치 정리
> `n=15`는 "더 늘려도 거의 안 좋아지는 지점"이면서 동시에 "아직 비용은 감당 가능한 지점"입니다. 따라서 이후 논문 재현이나 후속 실험의 기본 체크포인트로 사용하기에 가장 실용적입니다.

---

## 7. 결론

1. `n=15`는 논문이 예측한 대로 training PCC를 소폭 희생하면서 diffuser 일반화를 강화하는 구간입니다.
2. 실측 평가 기준으로 보면 `n=15`는 `n=20`과 거의 같은 blind 성능을 내면서도 계산 비용이 더 낮습니다.
3. `known < new` 패턴은 `n=15`가 더 이상 특정 diffuser를 외우는 모델이 아니라, diffuser distribution 자체를 학습한 모델임을 보여 줍니다.
4. 따라서 `n=15`는 4-layer 조건에서 가장 해석 가치가 높은 체크포인트이며, 이후 우선순위는 diffuser 수 추가보다 depth/architecture 확장에 가깝습니다.

---

## 부록: Run 산출물

| 파일 | 설명 |
|---|---|
| [`config.yaml`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n15_L4/config.yaml) | 전체 실험 설정 |
| [`model.pt`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n15_L4/model.pt) | 학습된 모델 체크포인트 |
| [`training_summary.json`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n15_L4/training_summary.json) | 최종 loss, PCC, 시간 요약: {loss: -0.9532, pcc: 0.8793, time: 4771.6s} |
| [`fig5_memory.npy`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/figures/fig5_memory.npy) | recent vs blind diffuser PCC 통계 |
| [`fig6_conditions.npy`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/figures/fig6_conditions.npy) | known / new / no-diffuser 조건별 PCC 통계 |
