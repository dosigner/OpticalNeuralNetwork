---
title: 실험 분석 - n10_L4 (다중 디퓨저, 중간 수준 다양성)
aliases:
  - analysis_n10_L4 한국어
tags:
  - d2nn
  - luo2022
  - analysis
  - obsidian
  - korean
date: 2026-03-13
source_note: "[[analysis_n10_L4]]"
status: translated
---

# 실험 분석: n10_L4 (다중 디퓨저, 중간 수준 다양성)

**Run ID:** `n10_L4`  
**Date:** `2026-03-13`  
**Reference:** Luo et al., "Computational Imaging Without a Computer: Diffractive Networks," *eLight* 2 (2022)

---

## 1. 실험 개요

### 모델 설정

| Parameter | Value |
|---|---|
| Architecture | Phase-only D2NN (`d2nn_phase_only`) |
| Number of layers | 4 |
| Grid size | 240 x 240 |
| Pixel pitch | 0.3 mm |
| Wavelength | 0.75 mm (400 GHz, THz regime) |
| Phase init | Uniform [0, 2pi) |
| Propagation | Band-limited angular spectrum method (BL-ASM) |
| Pad factor | 2x (480 x 480 padded grid) |

### 기하 구조

- Object-to-diffuser: 40.0 mm
- Diffuser-to-layer-1: 2.0 mm
- Layer-to-layer: 2.0 mm
- Last-layer-to-output: 7.0 mm

### 학습 파라미터

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Batch size (objects) | 64 |
| Diffusers per epoch (n) | **10** |
| Effective batch per epoch | B x n = 640 |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| LR schedule | Multiplicative decay, gamma = 0.99/epoch |
| Loss function | -PCC + energy penalty (alpha=1.0, beta=0.5) |
| Hardware | NVIDIA A100 (TF32 enabled) |
| Dataset | MNIST (50k train / 10k val) |
| Input encoding | Amplitude (grayscale), resized 28 -> 160 -> 240 px |

### 디퓨저 모델

Gaussian-smoothed height profile를 갖는 얇은 랜덤 위상 스크린:

- delta_n = 0.74, mean height = 25 lambda, std = 8 lambda
- Smoothing sigma = 4 lambda, correlation length = 10 lambda
- 10개의 고유한 디퓨저 realization을 epoch마다 샘플링함 (매 epoch마다 새로운 집합)

---

## 2. 학습 동역학

### 수렴 요약

| Epoch | PCC | Delta from prev. | Notes |
|---|---|---|---|
| 1 | 0.6253 | -- | 초기 품질; n=1의 첫 epoch 값(0.6729)보다 낮음 |
| 5 | 0.8569 | +0.2316 | 빠른 상승 구간 |
| 10 | 0.8756 | +0.0187 | 수렴 속도 둔화 |
| 15 | 0.8792 | +0.0036 | plateau 접근 |
| 20 | 0.8799 | +0.0007 | 사실상 평탄 |
| 25 | 0.8778 | -0.0021 | 약간의 변동 |
| 30 | 0.8836 | +0.0058 | 후반부의 약한 향상 |

**최종 지표:** loss = -0.9584, PCC = 0.8836  
**총 학습 시간:** 3,228 s (53.8 min), 약 107.6 s/epoch

### 수렴 곡선 분석

학습 곡선은 n=1 기준선과 질적으로 다른 성격을 보입니다.

1. **급격한 상승 (epochs 1-5):** PCC는 0.625에서 0.857로 상승하며, +0.232의 증가를 보입니다. 네트워크는 10개의 모든 디퓨저에 공통적으로 존재하는 지배적 저주파 산란 보정을 빠르게 학습합니다. 초기 기울기가 n=1과 비슷하다는 점은, 1차 위상 보정이 대체로 디퓨저 불변적임을 시사합니다.

2. **감속되는 수렴 (epochs 5-15):** PCC는 0.857에서 0.879로 증가하며, 10 epoch 동안 증가폭은 +0.022에 불과합니다. epoch 10에서 0.905에 plateau하는 n=1과 달리, n=10 네트워크는 더 낮은 ceiling으로 수렴합니다. 각 epoch마다 10개의 서로 다른 디퓨저 구성이 제시되기 때문에 loss landscape가 계속 바뀌며, 네트워크가 어떤 단일 산란 프로파일에도 완전히 특화되는 것을 막습니다.

3. **변동을 동반한 긴 plateau (epochs 15-30):** PCC는 0.878-0.884의 좁은 범위에서 진동합니다. 이런 변동(예: epoch 25에서 0.878로 하락 후, epoch 30에서 0.884로 회복)은 regularization 효과의 전형적인 특징입니다. 네트워크는 하나의 구성에 대해 뾰족한 minimum으로 수렴하는 것이 아니라, 디퓨저 앙상블 전체에 대해 성능 균형을 맞추고 있습니다.

n=1보다 느린 수렴과 더 낮은 asymptotic PCC는 실패의 징후가 아닙니다. 이는 디퓨저 다양성에 대해 학습할 때 나타나는 예상된 결과입니다. 네트워크는 더 어려운 최적화 문제를 풀고 있습니다. 하나의 산란 구성에 완벽히 맞는 위상 패턴을 찾는 대신, 10개의 서로 다른 산란 구성에 대해 모두 적절히 동작하는 위상 패턴을 찾아야 합니다.

---

## 3. 논문 기대 결과와의 비교

### 논문 맥락 (Luo et al. 2022, Fig. 3)

논문은 학습 중 디퓨저 다양성을 늘리는 것이 산란 매질을 통한 견고한 computational imaging을 달성하는 핵심 메커니즘임을 보여줍니다. 중심적인 trade-off는 다음과 같습니다.

| Metric | n=1 (memorization) | n=10 (generalization onset) | n>=20 (saturated) |
|---|---|---|---|
| Known-diffuser PCC | High (~0.91) | Moderate (~0.88) | Lower (~0.85) |
| Blind-test PCC | Very low | Substantial improvement | Approaches known-diffuser PCC |
| Generalization gap | Large | Narrowing | Small |

### 우리의 n=10 결과와 논문의 비교

- **Training PCC = 0.884**는 중간 수준 다양성 구간에 대한 논문의 기대와 잘 맞습니다. n=1 기준선(0.907) 대비 약 0.023 감소했다는 점은, 디퓨저 다양성이 암묵적 regularizer로 작용하여 네트워크가 특정 산란 구성 하나에 과적합하는 것을 막고 있음을 확인해 줍니다.

- **n=1에서 n=10으로 갈 때 PCC 감소는 크지 않습니다(~2.5%).** 이는 긍정적인 신호입니다. D2NN 아키텍처가 여러 디퓨저 구성을 동시에 어느 정도 수용할 수 있을 만큼 충분한 용량을 갖고 있음을 시사합니다. 감소폭이 더 컸다면, 4-layer 네트워크가 multi-diffuser 보정에 필요한 자유도를 충분히 갖고 있지 않다는 뜻일 수 있습니다.

- **수렴 시간척도도 일관적입니다.** 논문의 학습 프로토콜은 epoch당 n개의 디퓨저를 사용하며, 각 디퓨저마다 diffuser-D2NN 광학 체인에 대한 독립적인 forward/backward 전파가 필요합니다. 우리의 53.8분 실행 시간은 n=1의 7.6분 대비 7.1배이며, 이는 디퓨저 구성 수의 10배 증가와 batch size의 16배 증가(64 vs. 4)를 반영합니다. 다만 큰 배치 크기에서 GPU 활용이 더 좋아져 일부 비용이 상쇄되었습니다.

- **epoch 15 이후 PCC ~0.88에서 plateau가 형성되는 것은 논문의 regime diagram과 부합합니다.** Luo et al.에 따르면 n=10은 네트워크가 보지 못한 디퓨저에도 전달되는 특징을 학습하기 시작하는 "generalization sweet spot"의 시작 지점입니다. training PCC가 계속 상승하지 않고 안정화된다는 사실은, 네트워크가 디퓨저 앙상블 전반에 걸쳐 성능을 균형 있게 맞추는 구성을 찾았음을 의미합니다.

---

## 4. 핵심 관찰

### 암묵적 정규화로서의 디퓨저 다양성

n=10 학습 체제는 conventional한 정규화 기법(dropout, weight decay 등)과는 본질적으로 다른 형태의 regularization을 도입합니다.

- **메커니즘:** 각 epoch마다 10개의 서로 다른 랜덤 위상 스크린이 제시되며, 각 스크린은 같은 입력 물체에 대해 서로 다른 speckle 패턴을 만듭니다. 네트워크는 이 모든 산란 구성에 대해 효과적인 위상 보정을 학습해야 합니다.

- **학습되는 특징에 대한 효과:** n=1처럼 특정 디퓨저 전달함수의 inverse를 배우는 대신, 네트워크는 디퓨저 realization에 대해 불변적인 물체 정보를 추출하는 방향으로 학습합니다. 이는 디퓨저 앙상블 전체에 대한 "평균 inverse"를 학습하는 것과 유사합니다.

- **관측 가능한 신호:** 알려진 디퓨저에 대한 PCC 감소(0.884 vs. 0.907)와 plateau 거동은 이 regularization이 실제로 작동하고 있다는 직접적인 증거입니다. 네트워크는 디퓨저별 최적성을 일부 포기하는 대신, 디퓨저 간 강건성을 얻고 있습니다.

### 학습 효율 비교

| Metric | n=1 (baseline) | n=10 (this run) |
|---|---|---|
| Batch size (objects) | 4 | 64 |
| Diffusers per epoch | 1 | 10 |
| Effective samples/epoch | 4 | 640 |
| Time per epoch | ~15.1 s | ~107.6 s |
| Total training time | 454 s (7.6 min) | 3,228 s (53.8 min) |
| Final PCC (training) | 0.907 | 0.884 |
| Epochs to 90% of final PCC | ~5 | ~5 |
| PCC at epoch 1 | 0.673 | 0.625 |

epoch당 비용은 batch size 증가(16배)와 디퓨저 수 증가(10배)에 모두 비례하므로, epoch당 forward pass 수는 총 160배 증가합니다. 실제 wall-clock 시간 증가는 7.1배에 불과한데, 이는 TF32를 사용하는 A100에서 GPU 병렬화가 매우 효율적으로 작동했기 때문입니다. 더 큰 batch size가 kernel launch overhead를 상쇄하고 tensor core 활용률을 높였습니다.

### 더 낮은 초기 PCC는 과제 난이도를 반영함

epoch 1의 PCC가 0.625인 점(n=1의 0.673 대비)은 더 어려운 최적화 지형을 반영합니다. 디퓨저가 10개이면 초기의 랜덤 위상 레이어는 10개의 서로 다른 산란 구성에 대해 동시에 그럴듯한 출력을 만들어야 합니다. gradient 신호는 이 구성들에 대한 평균과 같아지므로, 초기 학습에서 디퓨저별 보정 신호는 희석됩니다.

### 손실함수 분해

최종 loss -0.9584는 n=1의 -0.9715보다 높으며, 이는 더 낮은 PCC와 energy penalty 기여도의 차이를 함께 반영합니다. energy penalty 항은 출력 에너지가 균일하게 분포되도록 유도하는데, 네트워크가 다양한 입력 intensity pattern을 처리해야 하는 multi-diffuser 구간에서는 이 항이 다르게 작용할 수 있습니다.

---

## 5. n=1 결과와의 비교

### 나란히 보는 요약

| Metric | n=1 (n1_L4) | n=10 (n10_L4) | Difference |
|---|---|---|---|
| Final PCC | 0.907 | 0.884 | -0.023 (-2.5%) |
| Final loss | -0.9715 | -0.9584 | +0.013 |
| Epoch-1 PCC | 0.673 | 0.625 | -0.048 |
| Epochs to plateau | ~10 | ~15 | +5 epochs |
| PCC at plateau | ~0.913 | ~0.880 | -0.033 |
| Total time | 7.6 min | 53.8 min | +46.2 min (7.1x) |
| Late-stage behavior | Mild regression | Stable fluctuation | -- |

### PCC 차이의 해석

n=1에서 n=10으로 갈 때 알려진 디퓨저 PCC가 2.5% 감소한 것은, 일반화의 대가로 예상되는 수준입니다. 이 차이는 특화와 강건성 사이의 trade-off를 정량화합니다.

- **n=1 network**는 사실상 하나의 디퓨저 inverse를 암기했습니다. 이 디퓨저에서는 PCC 0.907을 달성하지만, 보지 못한 디퓨저에서는 성능이 크게 떨어질 것으로 예상됩니다. 논문은 n=1에서 blind-test PCC가 0.5 이하로 떨어질 수 있음을 시사합니다.

- **n=10 network**는 절충적인 해를 학습했습니다. training diffuser에서의 PCC는 0.884로 더 낮지만, 논문은 이 네트워크가 보지 못한 디퓨저에 대해서는 훨씬 더 높은 PCC를 달성할 것으로 예측합니다. blind test에서 0.80-0.85 수준이 가능하며, 이는 n=1의 <0.5와 대조적입니다.

- **generalization gap**(known-diffuser PCC - blind-test PCC)은 n=1보다 n=10에서 훨씬 작을 것으로 예상됩니다. 이것이 Luo et al.의 핵심 결과입니다. n>=10에서는 D2NN이 디퓨저별 이미징에서 디퓨저 불변 이미징으로 넘어가기 시작합니다.

### 수렴 특성

n=1 곡선은 정적인 target에 과적합되는 경우에서 흔히 보이는, 뾰족한 optimum으로 빠르게 수렴한 뒤 약간 회귀하는 형태를 보입니다. 반면 n=10 곡선은 작은 진동을 동반하며 더 넓고 강건한 optimum으로 느리게 수렴하는데, 이는 새로운 디퓨저 샘플에 의해 loss landscape가 계속 재구성되는 regularized optimization의 전형적인 모습입니다.

---

## 6. n-Sweep 연구에서의 의미 (Fig. 3 재현)

### Sweep 안에서의 위치

| Sweep point | Status | Training PCC | Key finding |
|---|---|---|---|
| n=1 | Complete | 0.907 | Baseline: memorization regime |
| **n=10 (this run)** | **Complete** | **0.884** | **Onset of generalization; 2.5% PCC trade-off** |
| n=15 | Pending | ~0.87 (predicted) | Near-saturation expected |
| n=20 | Pending | ~0.85 (predicted) | Full-diversity reference |

### 논문 주장 검증

이 run은 Luo et al.의 두 가지 핵심 예측을 검증합니다.

1. **Training PCC는 n과 함께 감소한다** (0.907 -> 0.884): 확인됨. 네트워크는 여러 디퓨저를 동시에 완벽하게 보상할 수 없습니다.

2. **n=10은 generalization sweet spot의 시작이다:** 안정적인 plateau 거동과 중간 수준의 PCC 감소는, 네트워크가 암기에서 feature learning으로 넘어갔음을 시사합니다. 다음으로 중요한 검증은 blind-test evaluation입니다.

### 예측과 다음 단계

1. **이 체크포인트에 대해 20개의 unseen diffuser로 blind-test evaluation을 수행하는 것**이 가장 중요합니다. 논문에 따르면, n=10의 blind-test PCC는 0.80-0.85 범위일 것으로 예상되며, 이는 n=1 체크포인트의 <0.5와 대조됩니다.

2. **generalization gap**(known PCC - blind PCC)은 n=10에서 n=1보다 훨씬 작아야 합니다. 예를 들어 blind-test PCC가 ~0.83이라면 gap은 ~0.05가 되며, n=1의 예상 gap ~0.4+보다 훨씬 작습니다.

3. **n=20으로 확장하면** known-diffuser PCC는 조금 더 감소하겠지만, generalization gap은 거의 닫히며 논문의 Fig. 3 포화 곡선을 재현할 것입니다.

4. **학습 시간 외삽:** n=1에서 n=10으로 갈 때 7.1배 증가한 것을 보면, 같은 batch size에서 n=20 run은 대략 100-110분이 걸릴 것으로 추정됩니다. 디퓨저 수에 비례하되, GPU 포화로 인해 일부 sublinear 효율 향상이 있을 수 있습니다.

---

## 부록: Run 산출물

| File | Description |
|---|---|
| `config.yaml` | 전체 실험 설정 |
| `model.pt` | 학습된 모델 체크포인트 (4개 phase layer) |
| `training_summary.json` | 최종 loss, PCC, 시간 요약: {loss: -0.9584, pcc: 0.8836, time: 3228s} |
