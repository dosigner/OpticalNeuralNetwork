---
title: 실험 분석 - n20_L4 (최대 다양성, 포화 기준점)
aliases:
  - analysis_n20_L4 한국어
tags:
  - d2nn
  - luo2022
  - analysis
  - obsidian
  - korean
  - max-diversity
date: 2026-03-13
source_note: "[[analysis_n20_L4]]"
status: translated
---

# 실험 분석: n20_L4 (최대 다양성, 포화 기준점)

**Run ID:** `n20_L4`  
**Date:** `2026-03-13`  
**Reference:** Luo et al., "Computational Imaging Without a Computer: Diffractive Networks," *eLight* 2 (2022)

---

## 1. 실험 개요

> [!info] 실험 요약
> **모델**: 4-레이어 위상 전용 D2NN (240×240 그리드)  
> **디퓨저 수**: epoch당 20개 (최대 다양성)  
> **학습**: 30 epochs, B=64 (유효 배치 B×n=1280), A100 TF32  
> **최종 PCC**: ==0.878== | **최종 Loss**: -0.953 | **소요 시간**: 105.7분

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
| Diffusers per epoch (n) | **20** |
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
- epoch마다 20개의 고유한 diffuser realization을 새로 샘플링

> [!note] 설정 파일 메모
> 저장된 [`config.yaml`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n20_L4/config.yaml)에는 `batch_size_objects: 4`가 기록되어 있습니다. 이 노트의 요약 표기는 원본 분석 노트의 배치 표기를 유지합니다.

---

## 2. 학습 동역학

### 수렴 요약

| 구간 | Epoch | PCC | 관찰 |
|---|---|---|---|
| 급속 상승 | 1→5 | 0.62 → 0.85 | 기본 산란 보정 습득 |
| 안정화 | 5→15 | 0.85 → 0.87 | 20개 디퓨저 동시 최적화 |
| 수렴 | 15→30 | 0.87 → 0.878 | 미세 조정 완료 |

**최종 지표:** loss = -0.9526, PCC = 0.8785  
**총 학습 시간:** 6,339.3 s (105.7 min), 약 211.3 s/epoch

### 수렴 곡선 해석

1. **초기 공통 구조 학습 (epochs 1-5):** `n=20`에서도 초반 상승은 빠릅니다. 네트워크는 각 디퓨저의 세부 구조보다, 산란 과정 전체에 공통적인 저주파 보정 성분을 먼저 잡아냅니다.

2. **강한 앙상블 평균화 (epochs 5-15):** 이후에는 업데이트마다 20개 디퓨저의 gradient가 평균되므로, 단일 diffuser에 특화된 보정은 거의 허용되지 않습니다. 수렴 속도는 느려지지만 gradient는 더 "보수적"이고 더 넓은 분포를 대표하게 됩니다.

3. **포화 수렴 (epochs 15-30):** 후반부에는 PCC 향상이 거의 멈춥니다. 이 plateau는 단순한 학습 정체가 아니라, `n=20`이 이미 현재 4-layer 아키텍처가 표현할 수 있는 범용 산란 보정의 상한 근처에 도달했음을 뜻합니다.

> [!important] 핵심 관찰
> `n=20`의 핵심 가치는 더 높은 training PCC가 아니라, **최대 다양성 조건에서도 성능이 더 이상 거의 무너지지 않는다는 사실 자체**입니다. 이 실험은 "포화가 실제로 왔다"는 것을 확인하는 기준점입니다.

### 전체 n-sweep 비교

| 모델 | n | 최종 PCC (train) | 소요 시간 | epoch당 시간 | 유효 배치 |
|---|---|---|---|---|---|
| n1_L4 | 1 | **0.907** | 7.6분 | ~15초 | 64 |
| n10_L4 | 10 | **0.884** | 53.8분 | ~108초 | 640 |
| n15_L4 | 15 | **0.879** | 79.5분 | ~160초 | 960 |
| n20_L4 | 20 | **0.878** | 105.7분 | ~211초 | 1,280 |

`n=15 -> 20`에서 training PCC 차이는 약 -0.0008에 불과합니다. 계산 비용은 26.2분 늘지만, 평균 성능은 거의 그대로입니다. 이 결과는 `n=20`이 "추가 향상점"이라기보다 **포화 상한을 확인하는 실험**이라는 점을 분명하게 보여 줍니다.

---

## 3. 논문 기대 결과와의 비교

### 논문 맥락 (Luo et al. 2022, Fig. 3)

논문이 예측하는 핵심 패턴은 다음과 같습니다.

1. **n 증가 -> training PCC 감소**
2. **n 증가 -> blind generalization 향상**
3. **충분히 큰 n 이후 포화**

`n=20`은 이 세 번째 구간을 검증하는 실험입니다. 즉, "더 많은 diffuser를 보게 했더니 더 좋아졌다"가 아니라, "이제는 더 늘려도 거의 안 좋아진다"를 보여 주는 역할입니다.

### 우리의 n=20 결과와 논문의 비교

- **Training PCC = 0.8785**는 `n=15`의 0.8793과 사실상 같습니다. 논문이 제시한 saturation regime과 정성적으로 잘 맞습니다.

- **단조 감소 패턴이 완결됩니다.**
  - `n=1 -> 10`: -0.0231
  - `n=10 -> 15`: -0.0043
  - `n=15 -> 20`: -0.0008

  감소폭이 급격히 줄어들기 때문에, `n=20`은 sweep의 오른쪽 끝에서 곡선이 거의 평탄해졌음을 보여 주는 데이터포인트입니다.

- **일반화 측면에서도 ceiling 검증 역할**을 합니다. `n=20`은 `n=15`보다 평균 성능을 크게 밀어 올리지는 못하지만, blind diffuser 성능이 거의 유지된다는 점에서 이미 일반화가 saturation에 도달했음을 뒷받침합니다.

> [!tip] 해석 포인트
> `n=20`의 장점은 `n=15`보다 눈에 띄게 높은 PCC가 아니라, **"현재 4-layer 구조에서는 diffuser 수를 더 늘려도 개선 여지가 매우 작다"는 결론을 더 강하게 만들 수 있다는 점**입니다.

---

## 4. n=20에서 n=15보다 가지는 이점

### 4.1 최대 다양성 조건을 실제로 찍었다는 점

`n=15`는 포화 직전의 효율적 지점이지만, `n=20`은 그 포화가 진짜인지 확인하는 **상한 기준점**입니다.  
이 차이는 실험 해석에서 중요합니다.

- `n=15`만 있으면 "조금 더 늘리면 나아질 수도 있다"는 여지가 남습니다.
- `n=20`까지 보면 "더 늘려도 거의 안 변한다"는 결론을 훨씬 강하게 말할 수 있습니다.

즉 `n=20`의 첫 번째 장점은 평균 성능보다 **결론의 강도**에 있습니다.

### 4.2 blind diffuser 분산이 가장 작다

`fig5_memory.npy` 기준 blind evaluation 결과:

| 모델 | Blind mean PCC | Blind std |
|---|---|---|
| n15_L4 | 0.872835 | 0.008956 |
| n20_L4 | **0.872924** | **0.008833** |

평균 향상은 `+0.000089`로 사실상 미세하지만, 표준편차가 가장 낮습니다. 이 값은 크지 않더라도 해석상 의미가 있습니다.

- unseen diffuser가 바뀌어도 성능 분산이 더 작음
- 평균치보다 안정성에 더 무게를 둘 때 `n=20`이 가장 보수적인 선택지임

즉 `n=20`의 실측상 가장 명확한 이점은 **최고 평균치가 아니라 최소 분산**에 가깝습니다.

### 4.3 정규화 압력이 가장 강하다

`n=20`은 한 epoch에서 20개의 서로 다른 phase screen을 보게 하므로, 개별 diffuser에 맞춘 업데이트가 가장 강하게 억제됩니다. 이는 곧 **가장 강한 데이터-기반 regularization** 조건입니다.

이 점의 의미는 다음과 같습니다.

- 모델이 특정 seed나 특정 diffuser set에 특화될 여지가 최소화됨
- 학습된 위상 패턴이 diffuser-specific inverse보다 distribution-level transform에 더 가까워짐
- "이 정도까지 다양성을 늘려도 여전히 0.878 수준을 유지한다"는 점이 곧 구조의 robustness 증거가 됨

이 해석은 실측 blind variance 감소와도 방향성이 일치합니다.

### 4.4 논문 재현과 후속 실험의 기준점

`n=20`은 Luo et al. 논문에서 강조하는 "높은 diffuser diversity" regime을 가장 직접적으로 대응시키는 체크포인트입니다. 그래서 다음과 같은 실무적 이점이 있습니다.

- **Fig. 3 재현의 오른쪽 끝점**으로 사용 가능
- **Fig. 5/6 일반화 비교의 saturation reference**로 사용 가능
- 이후 depth sweep(`L2`, `L5`)에서 "데이터 다양성 부족"이 아니라 "모델 용량 한계"를 분리해 해석하는 기준점으로 사용 가능

즉 `n=20`은 배치 효율은 떨어지지만, **논문 대응성**과 **후속 실험 해석력** 면에서는 가장 강한 기준점입니다.

### 4.5 무엇이 장점이 아닌지도 분명하다

엄밀히 보면 `n=20`이 `n=15`보다 모든 지표에서 우세하지는 않습니다.

- `fig6`의 new diffuser mean PCC는 `n=15`가 더 높음 (`0.878434` vs `0.875003`)
- training PCC도 `n=15`가 약간 더 높음 (`0.879298` vs `0.878490`)

따라서 `n=20`의 우위는 "평균 정확도 우세"가 아니라 다음 두 가지로 요약해야 정확합니다.

1. **최대 다양성 조건 자체가 주는 해석적 가치**
2. **blind 조건에서의 아주 미세한 안정성 우위**

> [!success] n=20의 장점 정리
> `n=20`은 `n=15`보다 드라마틱하게 더 정확한 모델은 아닙니다. 대신, 최대 diffuser 다양성에서 이미 성능 포화가 왔음을 확인하고, blind 성능 분산을 가장 낮게 만드는 **가장 보수적이고 해석력이 높은 체크포인트**입니다.

---

## 5. 물리적 해석

### 디퓨저 다양성의 정규화 효과

각 epoch에서 손실 함수는 다음과 같습니다.

$$
\mathcal{L}
= -\frac{1}{20B}\sum_{i=1}^{20}\sum_{j=1}^{B}
\text{PCC}(I_{ij}^{\text{out}}, I_j^{\text{target}})
+ \alpha \cdot \mathcal{E}_{\text{penalty}}
$$

20개 디퓨저에 대한 평균 손실은 모델에 다음 제약을 강하게 겁니다.

- 한 diffuser에서만 좋은 해를 찾는 방향 억제
- 여러 diffuser에서 동시에 통하는 평균적 위상 보정 선호
- 개별 realization보다 산란 통계량 자체를 반영하는 해 선호

### 왜 n=15와 거의 같아지는가

`n=20`이 `n=15`보다 크게 나아지지 않는 이유는, 추가된 5개 diffuser가 더 이상 새로운 "학습 법칙"을 제공하지 못하기 때문입니다. 이미 `n=15`에서 diffuser distribution의 핵심 통계가 충분히 반영되었고, 이후부터는 모델의 자유도보다 더 많은 다양성을 넣어도 정보 이득이 작습니다.

즉 병목은 다음처럼 이동합니다.

- `n < 10`: diffuser 다양성 부족
- `n ≈ 15`: 다양성은 충분, 포화 직전
- `n = 20`: 다양성은 사실상 충분 이상, 이제 병목은 모델 용량

이 해석이 맞다면, 다음 개선은 `n`을 더 늘리는 것보다 레이어 수나 아키텍처를 바꾸는 쪽에서 나와야 합니다.

---

## 6. n-sweep 재현에서의 의미

### Sweep 안에서의 위치

| Sweep point | Status | Training PCC | 핵심 의미 |
|---|---|---|---|
| n=1 | Complete | 0.907 | 단일 diffuser 암기 기준선 |
| n=10 | Complete | 0.884 | 일반화 onset |
| n=15 | Complete | 0.879 | 효율적인 포화 직전 지점 |
| **n=20** | **Complete** | **0.878** | **포화 상한 확인용 기준점** |

### Fig. 3/5/6 관점의 해석

1. **Fig. 3:** training PCC 곡선의 오른쪽 끝이 실제로 평탄해졌음을 보여 줍니다.
2. **Fig. 5:** blind mean은 `n=15`와 사실상 동일하지만, std는 가장 낮습니다.
3. **Fig. 6:** 모든 지표에서 최고는 아니지만, 높은 다양성 조건에서도 known/new/no-diffuser 성능이 모두 안정적인 범위에 머무릅니다.

`n=20`이 있기 때문에, 이제 이 프로젝트는 "어디까지 diffuser를 늘려야 하는가?"보다 "같은 다양성 조건에서 더 깊은 모델이 무엇을 바꾸는가?"를 묻는 단계로 넘어갈 수 있습니다.

---

## 7. 결론

1. `n=20`은 최대 diffuser 다양성 조건에서 training PCC 0.8785를 유지하며, `n=15` 이후 포화가 실제임을 확인합니다.
2. `n=15` 대비 평균 성능 이득은 거의 없지만, blind diffuser 분산이 가장 낮아 가장 보수적인 일반화 체크포인트 역할을 합니다.
3. `n=20`의 진짜 장점은 더 높은 숫자보다, saturation reference와 최대 regularization 조건이라는 해석적 가치에 있습니다.
4. 따라서 `n=20`은 비용 효율 면에서는 `n=15`보다 불리하지만, 논문 재현과 후속 depth/architecture 비교의 상한 기준점으로는 가장 중요합니다.

---

## 부록: Run 산출물

| 파일 | 설명 |
|---|---|
| [`config.yaml`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n20_L4/config.yaml) | 전체 실험 설정 |
| [`model.pt`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n20_L4/model.pt) | 학습된 모델 체크포인트 |
| [`training_summary.json`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/runs/n20_L4/training_summary.json) | 최종 loss, PCC, 시간 요약: {loss: -0.9526, pcc: 0.8785, time: 6339.3s} |
| [`fig5_memory.npy`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/figures/fig5_memory.npy) | recent vs blind diffuser PCC 통계 |
| [`fig6_conditions.npy`](/root/dj/D2NN/luo2022_random_diffusers_d2nn/figures/fig6_conditions.npy) | known / new / no-diffuser 조건별 PCC 통계 |
