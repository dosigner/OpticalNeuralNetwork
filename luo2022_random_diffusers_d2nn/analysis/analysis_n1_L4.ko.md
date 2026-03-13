---
title: 실험 분석 - n1_L4 (단일 디퓨저 기준선)
aliases:
  - analysis_n1_L4 한국어
tags:
  - d2nn
  - luo2022
  - analysis
  - obsidian
  - korean
date: 2026-03-13
source_note: "[[analysis_n1_L4]]"
status: translated
---

# 실험 분석: n1_L4 (단일 디퓨저 기준선)

**Run ID:** `n1_L4`  
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
| Batch size (objects) | 4 |
| Diffusers per epoch (n) | **1** |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| LR schedule | Multiplicative decay, gamma = 0.99/epoch |
| Loss function | -PCC + energy penalty (alpha=1.0, beta=0.5) |
| Hardware | NVIDIA A100 (TF32 enabled) |
| Dataset | MNIST (50k train / 10k val) |
| Input encoding | Amplitude (grayscale), resized 28 -> 160 -> 240 px |

### 디퓨저 모델

Gaussian-smoothed height profile을 갖는 얇은 랜덤 위상 스크린:

- delta_n = 0.74, mean height = 25 lambda, std = 8 lambda
- Smoothing sigma = 4 lambda, correlation length = 10 lambda

---

## 2. 학습 동역학

### 수렴 요약

| Epoch | PCC | Notes |
|---|---|---|
| 1 | 0.6729 | 한 번의 pass 이후 초기 복원 품질 |
| 10 | 0.9050 | 빠른 수렴, 대부분의 학습이 이 구간에서 일어남 |
| 20 | 0.9125 | plateau에 근접, 증가폭이 작음 |
| 30 | 0.9066 | peak 대비 약간의 회귀 |

**최종 지표:** loss = -0.9715, PCC = 0.9066  
**총 학습 시간:** 454 s (7.6 min), 약 15.1 s/epoch

### 수렴 곡선 분석

학습 곡선은 세 개의 뚜렷한 구간을 보입니다.

1. **급격한 상승 (epochs 1-10):** PCC는 0.67에서 0.91로 상승하며, +0.24의 증가를 보입니다. 네트워크는 단일 고정 디퓨저를 역으로 보정하는 데 필요한 위상 보정을 빠르게 학습합니다. 이 구간은 디퓨저 전달함수의 지배적인 저주파 구조가 보상되는 영역입니다.

2. **정체 구간 (epochs 10-20):** PCC는 0.905에서 0.913으로 조금씩 증가하며, 10 epoch 동안 증가폭은 +0.008에 불과합니다. 네트워크는 사실상 수렴했으며, 남은 개선은 고공간주파수 위상 보정의 미세 조정에 해당합니다.

3. **약한 진동 (epochs 20-30):** PCC는 약간 감소하여 0.907이 됩니다. 이는 training-set 통계에 대한 과적합 또는 learning-rate 동역학의 작은 영향에서 흔히 나타나는 현상입니다. `n=1`에서는 일반화를 유도할 디퓨저 다양성이 없으므로, 네트워크는 일반화 가능한 특징을 학습할 유인이 없습니다. 학습 배치 구성의 작은 변화만으로도 loss landscape가 약간 이동할 수 있습니다.

거의 단조로운 수렴과 이른 plateau는 `n=1`에서 예상되는 현상입니다. 네트워크는 하나의 고정된 산란 구성을 역으로 푸는 방법만 학습하면 되므로, 최적화 문제는 비교적 단순합니다.

---

## 3. 논문에서 기대되는 결과와의 비교

### 논문 맥락 (Luo et al. 2022, Fig. 3)

논문의 핵심 결과는, 한 epoch당 여러 개의 랜덤 디퓨저(`n >> 1`)로 학습하면 D2NN이 특정 디퓨저를 암기하는 것이 아니라 일반화 가능한 이미징 변환을 학습하게 된다는 점입니다. 논문에서의 핵심 기준점은 다음과 같습니다.

| n (diffusers/epoch) | Expected behavior |
|---|---|
| n = 1 | Baseline. High PCC on the training diffuser, poor generalization to unseen diffusers. |
| n = 5 | Moderate improvement in blind-test PCC. |
| n = 10-15 | Significant jump; network begins learning diffuser-invariant features. |
| n = 20 | Near-saturated generalization performance. |

### 우리의 n=1 결과와 논문의 비교

- **Training PCC = 0.907**은 논문의 n=1 baseline 영역과 일관됩니다. 논문은 training diffuser에서의 정확한 n=1 PCC 값을 직접 보고하지는 않지만, 알려진 디퓨저에 대해 PCC가 약 0.91 수준이라는 것은 물리적으로 타당합니다. 이는 4-layer D2NN이 단일 phase screen으로부터 발생하는 산란의 대부분을 보상할 수 있음을 뜻합니다.

- 손실함수의 형태가 `-PCC + energy penalty`이므로, final_loss = -0.9715는 대략 0.97 수준의 raw PCC 기여에서 energy penalty 항(~0.06)을 뺀 것으로 해석할 수 있습니다. 이는 네트워크가 높은 구조적 상관성을 달성했지만, 출력 에너지를 목표 support 내부에만 모두 집중시키지 못하도록 energy penalty가 작동하고 있음을 의미합니다.

- **수렴 속도**가 epoch 10 무렵 plateau에 도달하는 것도 `n=1`에서 예상되는 결과입니다. 디퓨저가 하나뿐이면 유효 데이터셋이 multi-diffuser 경우보다 훨씬 단순합니다. 반대로 multi-diffuser의 경우에는 각 epoch마다 새로운 산란 구성이 들어오므로 loss landscape가 지속적으로 흔들립니다.

---

## 4. 핵심 관찰

### 디퓨저 암기 vs 일반화 가능한 학습

`diffusers_per_epoch = 1`에서는 네트워크가 매 epoch마다 정확히 같은 산란 구성을 봅니다. 따라서 학습된 위상 레이어는 해당 디퓨저의 point-spread function에 대한 근사적 inverse로 수렴합니다. 이는 multi-diffuser 학습과 본질적으로 다릅니다.

- **n=1 learns:** "이 특정 디퓨저의 왜곡을 어떻게 되돌릴 것인가."
- **n>>1 learns:** "디퓨저가 유발하는 수차와 관계없이 물체 정보를 어떻게 추출할 것인가."

이 구분은 일반적인 딥러닝에서의 overfitting과 generalization의 차이와 유사합니다. 하나의 디퓨저로 학습된 네트워크는 그 디퓨저에서는 높은 PCC를 보일 수 있지만, 테스트 시 새로운 디퓨저를 만나면 성능이 크게 무너질 가능성이 높습니다.

### 학습 효율

7.6분의 학습 시간은 `n=1` 경우의 단순함을 반영합니다.

- epoch당 1개의 디퓨저 패턴만 생성하고 적용하면 됨
- batch마다 여러 디퓨저 구성을 통과시킬 필요가 없음
- target이 고정되어 있으므로 최적화가 빠르게 수렴함

이는 유용한 계산 기준선이 됩니다. multi-diffuser run(`n=10`, `n=20`)은 각 디퓨저 구성마다 별도의 forward/backward pass가 필요하므로, 학습 시간은 대체로 `n`에 비례하여 증가합니다.

### 미세한 PCC 회귀 (Epochs 20-30)

0.913에서 0.907로의 감소는 작지만(~0.7%) 주목할 만합니다. 가능한 원인은 다음과 같습니다.

- LR schedule (`gamma=0.99`)은 epoch 30에서도 여전히 무시할 수 없는 크기의 업데이트를 허용합니다.  
  effective LR = `1e-3 * 0.99^29 = 7.5e-4`  
  따라서 optimizer가 minimum 근처에서 약간 떠돌 수 있습니다.

- `n=1`에서는 디퓨저 다양성으로부터 오는 regularization 효과가 없으므로, 네트워크는 학습 데이터의 batch-level noise에 과적합할 수 있습니다.

---

## 5. n-Sweep 연구에서의 의미 (Fig. 3 재현)

이 run은 디퓨저 다양성 sweep에서 **하한(lower bound)** 을 설정합니다.

### Sweep에서의 역할

| Sweep point | Purpose |
|---|---|
| **n=1 (this run)** | Baseline: best-case for known diffuser, worst-case for blind test |
| n=5 | Transition regime |
| n=10 | Onset of robust generalization |
| n=15 | Near-saturation |
| n=20 | Full-diversity reference (paper's primary result) |

### 이후 run에 대한 예측

1. **Known-diffuser PCC는 n이 증가할수록 감소할 것입니다.** 더 많은 디퓨저로 학습할수록 네트워크는 어느 하나의 디퓨저도 완벽하게 보상할 수 없습니다. known-diffuser PCC는 약 `0.91 (n=1)`에서 `0.85-0.88 (n=20)` 정도로 내려갈 것으로 예상됩니다.

2. **Blind-test PCC는 n이 증가할수록 상승할 것입니다.** 이것이 논문의 핵심 결과입니다. `n=1` 네트워크는 blind-test PCC가 매우 낮을 가능성이 높고(0.5 이하일 수도 있음), 반면 `n=20`은 blind-test PCC가 known-diffuser PCC와 비슷한 수준에 이를 수 있습니다.

3. **Known-diffuser PCC와 blind-test PCC의 차이는 generalization gap입니다.** `n=1`에서는 이 gap이 최대일 것이고, n-sweep이 진행될수록 이 gap이 줄어들며 논문의 Fig. 3을 재현하게 될 것입니다.

4. **학습 시간은 n에 비례하여 증가할 것입니다.** `n=1`에서 7.6분이라는 결과를 바탕으로 단순 추정하면, `n=20`의 30 epoch run은 약 150분이 걸릴 수 있습니다. 논문처럼 100 epoch를 사용한다면 그보다 더 길어질 것입니다.

### 다음 단계

- 이 체크포인트를 대상으로 20개의 unseen diffuser에 대한 blind-test evaluation 수행
- 동일한 hyperparameter를 유지한 채 `n=5, 10, 15, 20` 학습 run 실행 (`diffusers_per_epoch`만 변경)
- known-diffuser PCC와 blind-test PCC 곡선을 비교하여 Fig. 3 재현

---

## 부록: Run 산출물

| File | Description |
|---|---|
| `config.yaml` | 전체 실험 설정 |
| `model.pt` | 학습된 모델 체크포인트 (4개의 phase layer) |
| `training_summary.json` | 최종 loss, PCC, 시간 요약 |
