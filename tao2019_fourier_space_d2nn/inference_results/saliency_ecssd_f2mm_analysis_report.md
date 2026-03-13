# F-D2NN ECSSD Saliency Detection: Diagnosis & Fix Report

## 1. Experiment Overview

**Objective**: ECSSD saliency detection의 blob-only 출력 문제 해결.

3가지 실험 진행:
- **Baseline**: 원본 config (SBN enabled, init_scale 무시 버그 존재)
- **Attempt 1**: SBN 파라미터 튜닝 (per_layer, per_sample_minmax, I_sat=0.5 등)
- **Attempt 2**: SBN 완전 비활성화 + init_scale 버그 수정

## 2. Quantitative Results

| | Baseline | Attempt 1 (SBN 튜닝) | Attempt 2 (SBN off + init_scale fix) |
|---|----------|----------------------|--------------------------------------|
| Run dir | `260304_085630` | `260304_160712` | `260304_162156` |
| SBN | enabled (rear) | enabled (per_layer) | **disabled** |
| init_scale | 0.1 (무시됨) | 0.1 (무시됨) | **0.1 (적용됨)** |
| Best epoch | 100 | 145 | **20** |
| Train loss (final) | 0.1563 | 0.1662 | **0.1544** |
| Val loss (best) | 0.1799 | 0.1869 | **0.1780** |
| **Val F_max** | **0.5461** | **0.5290** | **0.5551** |
| Delta vs baseline | - | -0.0171 | **+0.0090** |

### Key Observations
- **Attempt 1 (SBN 튜닝)**: F_max 0.5290으로 **오히려 악화**. SBN이 방해 요인이었음.
- **Attempt 2 (SBN off + init_scale)**: F_max 0.5551로 **1.6% 개선**. Best epoch이 20으로 수렴 속도도 빨라짐.
- SBN 비활성화가 SBN 튜닝보다 0.026 더 높은 F_max.

## 3. Code Fixes Applied

### Fix 1: SBN 비활성화
```yaml
# saliency_ecssd_f2mm.yaml
nonlinearity:
  enabled: false  # was: true
```

**근거**: SBN 파라미터들은 논문에 없는 추가 구현. 실험적으로도 성능을 저하시킴.

### Fix 2: init_scale 버그 수정

**문제**: Config의 `init_scale: 0.1`이 PhaseMask 코드에서 무시됨.

**수정 파일 3개**:

1. `src/tao2019_fd2nn/models/phase_mask.py`: `init_scale` 파라미터 추가
   ```python
   # Before: nn.init.uniform_(self.raw, -1.0, 1.0)
   # After:  nn.init.uniform_(self.raw, -init_scale, init_scale)
   ```

2. `src/tao2019_fd2nn/models/fd2nn.py`: `Fd2nnConfig`에 `phase_init_scale` 필드 추가, `PhaseMask` 생성 시 전달

3. `src/tao2019_fd2nn/cli/common.py`: `build_model()`에서 config의 `init_scale`을 `phase_init_scale`로 매핑

**효과**: Phase mask 초기값이 uniform(-1,1) → uniform(-0.1,0.1)로 축소.
- sigmoid(-0.1)~sigmoid(0.1) = 0.475~0.525 → phase ≈ [0.475, 0.525] * 2π
- 좁은 초기 phase 범위에서 시작하여 점진적으로 학습

## 4. Visual Quality Assessment

### Saliency Predictions
- **Baseline/Attempt1/Attempt2 모두**: 여전히 center-biased blob 패턴. GT saliency map과의 공간적 대응 부족.
- Attempt 2에서 일부 이미지에 약간 더 선명한 경계가 보이나 근본적 차이 없음.

### Phase Masks
- **Baseline (init_scale 무시)**: 넓은 phase 분포 (noise-like)
- **Attempt 2 (init_scale 적용)**: 좁은 phase 분포 (더 uniform에 가까움, 작은 변화만 학습)
- 구조화된 패턴은 여전히 미발견

### PR Curve
- Baseline: Fmax 0.546, max precision ~0.65
- Attempt 2: Fmax 0.555, max precision ~0.68
- 전체 곡선에서 약간의 개선

## 5. 남은 문제

SBN 제거 + init_scale 수정으로 소폭 개선(+0.009)했지만, **blob 출력 문제는 해결되지 않음**.

이는 F-D2NN 아키텍처의 근본적 특성:
- **Phase-only Fourier filtering**: 진폭 변조 없이 위상만 조절 → |u|² readout에서 low-frequency 지배
- **비선형성 부재**: SBN 제거 후 유일한 비선형성은 detector의 |u|² → linear system + square-law detection
- **Center bias 활용**: F_max ~0.55는 ECSSD의 center bias와 대략 일치하는 수준

## 6. 향후 방향

1. Loss function 변경 (MSE → BCE + IoU)
2. Amplitude modulation 허용
3. Real-space 또는 Hybrid D2NN으로 전환
4. Layer 수 증가 / 다른 propagation distance 탐색
