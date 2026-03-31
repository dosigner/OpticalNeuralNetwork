---
title: FD2NN Loss Sweep Review
aliases:
  - fd2nn-loss-sweep-comprehensive-review
tags:
  - kim2026
  - fd2nn
  - review
  - optics
date: 2026-03-25
status: review-note-updated
---

# FD2NN Loss Sweep Review

대상 문서: [[fd2nn-loss-sweep-comprehensive]]

> [!summary]
> 2026-03-25 병렬 검토와 코드 대조 결과, 가장 큰 문제는 네 가지였다.
> 1. export 노트의 그림 임베드 상대경로가 잘못되어 Obsidian에서 보이지 않았다.
> 2. Section 8.3의 `Complex` 라벨이 Section 2.1의 complex loss와 다른 objective를 가리키고 있었다.
> 3. `Strehl`, `z/z_R`, `Fresnel lens` 같은 용어가 과하게 강하거나 부정확하게 쓰였다.
> 4. 현재 sweep에서 본 현상을 phase-only optics의 보편 법칙처럼 서술한 문장이 있었다.

## Confirmed Root Causes

- 그림 미표시는 원본 노트의 상대경로를 export 노트가 그대로 복사한 탓이다.
  원본은 `kim2026/docs/04-report/features` 아래에 있고, export는 `tmp/pdfs` 아래에 있으므로 `../../../runs/...`는 더 이상 유효하지 않다.
  실제 이미지 자산은 `kim2026/runs/...`와 `kim2026/figures/...`에 있다.
- Section 8.3의 `Complex`는 문서 초반의 `complex_overlap + 0.5 amplitude_mse`와 같은 loss가 아니다.
  `kim2026/scripts/sweep_loss_f100mm.py` 기준으로 2차 loss sweep의 `complex` 라벨은 `complex_overlap + intensity_overlap + beam_radius + encircled_energy` 조합이다.
- 2차 loss sweep에서 `Complex`와 `Irradiance`의 요약 metric이 동일하다고 해서 `L_CO`가 무의미했다고 결론내릴 수는 없다.
  서로 다른 구현, 같은 basin, 중복 config, 또는 요약 metric의 분해능 부족이 모두 가능하다.

## Physics

- `Parseval 제약`과 `점wise intensity 불변`은 분리해야 한다.
  점wise intensity 불변은 `|e^{i\phi}|=1` 때문이고, Parseval은 이상적 유니터리 변환의 총 에너지 보존에 관한 성질이다.
- 현재 구현은 NA mask와 band-limited propagation을 포함하므로, 총 power가 자동 보존된다고 쓰면 과하다.
- 본문은 관찰된 `co↔io` 충돌을 phase-only physics의 보편 법칙처럼 확장한 부분이 있었다.
  올바른 표현은 "현재 architecture / objective / metric 조합에서 empirical trade-off가 관찰되었다"이다.
- `io` 하나로 direct-detection usefulness를, 또는 low `co` 하나로 coherent usefulness 부재를 단정하면 과하다.
  receiver-level usefulness는 coupling, BER, power, SNR, misalignment까지 봐야 한다.

## Optics

- `Strehl`은 고전적 Strehl ratio가 아니라 unit-energy normalized peak ratio다.
  1보다 큰 값이 나오므로 bare `Strehl` 대신 `Peak Proxy` 또는 `peak-concentration proxy`가 맞다.
- irradiance / amplitude / intensity 표기가 섞인 부분이 있었다.
  `amplitude = |E|`, `intensity(or irradiance) = |E|^2`로 분리해 써야 한다.
- 동심원형 wrapped phase를 바로 `Fresnel lens`로 읽는 것은 과하다.
  phase unwrapping과 curvature fitting 전에는 "quadratic-like / defocus-like spectral phase candidate" 정도가 안전하다.
- direct-detection-like objective와 coherent-field objective의 차이는 sensor 종류 그 자체라기보다 phase 민감도 차이로 설명하는 편이 정확하다.

## Diffraction Optics

- `spacing=0이면 amplitude 재분배 불가`는 구현과 맞지 않는다.
  zero-spacing이어도 전체 스택은 하나의 Fourier-plane phase filter로 작동하므로 출력면 intensity reshaping 자체는 가능하다.
- `z_R = π(10 dx_f)^2 / λ`는 실제 beam Rayleigh range가 아니라 10-pixel feature에 대한 heuristic diffraction-length scale이다.
  따라서 `Near-field`, `Rayleigh range`, `Far-field` 같은 전체 시스템 레짐 이름을 붙이면 과하다.
- multilayer phase-only 시스템에서 output-plane amplitude/intensity는 불가능이 아니라 간접 제어 문제다.
  propagation/interference를 통해 reachable set이 제한적으로 형성된다고 쓰는 편이 맞다.
- illuminated pupil 기준 Fresnel number까지 함께 보지 않으면 layer spacing을 far-field로 부를 수 없다.

## Engineering

- `가능`, `유망`, `제작 가능` 같은 문구는 현재 증거보다 강하다.
  지금은 `co/io/pr/Peak Proxy` 같은 시뮬레이션 지표만 있고 BER, coupling efficiency, tolerance, polarization, bandwidth, alignment 분석이 없다.
- 1차 vs 2차 비교는 confounded comparison이다.
  `f`, spacing, ROI, phase range, epochs가 함께 바뀌었으므로 ranking이 아니라 hypothesis 수준으로만 써야 한다.
- 재현성 메타데이터도 부족했다.
  seed, split size, batch size, optimizer/scheduler, checkpoint selection, hardware/runtime, git commit가 문서에 바로 보이지 않는다.
- 장기 제작 로드맵은 후보 공정안 수준으로 낮추고, sampling proxy와 fabrication feasibility를 분리해서 적는 편이 낫다.

## Mathematics

- phasor loss 설명이 코드보다 단순했다.
  plain phasor는 amplitude threshold mask 위에서 계산되고, f=10mm loss sweep의 phasor run은 `soft_weighted_phasor_loss + leakage`다.
- Section 8.3의 `Complex` 정의는 수학적으로 Section 2.1 complex loss와 충돌했다.
  후반부 objective는 별도 라벨로 분리해야 한다.
- `[-π, π] = [0, 2π]` 같은 표기는 집합으로서 거짓이다.
  `mod 2π` wrap 이후 physical equivalence로 써야 한다.
- `|e^{i\phi_1}-e^{i\phi_2}|^2 ≈ 2(1-\cos\Delta\phi)`는 근사가 아니라 exact identity다.
- `phase correction에 최소 2π 필요`도 현재 table이 보여주는 것보다 강한 일반화다.
  올바른 표현은 "이 sweep에서는 tested 2π-range parameterization이 π-range보다 좋았다"이다.

## Priority Fixes

- export 스크립트에서 이미지 경로를 export 위치 기준 `../../kim2026/...`로 재기록
- `Strehl`를 bare label로 쓰지 말고 `Peak Proxy`로 통일
- `z/z_R`를 `z/z_diff,10px` 같은 heuristic scale 이름으로 바꾸고 near/far-field 명명 제거
- Section 8.3의 `Complex*`를 Section 2.1 complex loss와 명시적으로 분리
- `co↔io trade-off`와 `f=10mm 후보`는 empirical observation / hypothesis로만 서술
