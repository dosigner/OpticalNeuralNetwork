# D2NN Physics Hardening Troubleshooting

## 목적

`kim2026` D2NN 경로에서 발견된 물리 위반성 수치 문제를 기록한다.
이번 노트는 다음 3가지를 재발 방지용으로 정리한다.

- same-window FFT 전파의 `wrap-around` / aliasing
- passive D2NN 해석과 에너지 처리 원칙
- legacy Strehl 계산의 물리적 한계와 corrected Strehl 기준

## 증상

### 1. 경계 반대편으로 에너지가 재등장함

same-window FFT 전파는 수치적으로는 편하지만, 패딩이 없으면 원형 합성곱이 된다.
그 결과 우측 경계 근처의 field를 전파했을 때 좌측 경계에 비물리적인 에너지가 나타난다.

이 현상은 다음 착시를 만든다.

- throughput은 거의 1.0으로 보일 수 있음
- complex overlap 보존이 과도하게 좋아 보일 수 있음
- 하지만 실제 자유공간 전파와는 다른 periodic torus 모델이 됨

### 2. Strehl이 reference 정의에 따라 비정상적으로 커짐

기존 legacy Strehl은 focal intensity끼리 바로 비교했다.
이 방식은 다음 문제가 있었다.

- flat-phase diffraction-limited reference가 아님
- PSF undersampling 영향을 직접 받음
- passive system인데도 `S > 1`이 쉽게 나올 수 있음

### 3. zero-phase D2NN를 identity로 해석하는 착오

zero-phase D2NN는 "아무것도 하지 않는 identity"가 아니라,
위상판은 0이지만 층간 자유공간 전파는 그대로 존재하는 광학계다.

즉 zero-phase model은:

- phase modulation 없음
- free-space propagation 있음
- 따라서 단순 pass-through와 동일하지 않음

## 반영한 물리 원리

### 1. 자유공간 전파는 선형 합성곱으로 다뤄야 함

입력 field와 propagation kernel의 관계는 선형 합성곱이다.
FFT를 직접 쓰면 기본적으로 원형 합성곱이므로,
선형 합성곱을 근사하려면 `zero-padding -> propagate -> center crop`가 필요하다.

이번 수정에서 `kim2026` D2NN 경로는 이 원칙으로 바꿨다.

- 내부 helper: `kim2026/src/kim2026/optics/padded_angular_spectrum.py`
- D2NN callsite: `kim2026/src/kim2026/models/d2nn.py`

기본값은 `pad_factor = 2`다.
이건 "minimum practical anti-wrap padding"으로 채택했다.

### 2. phase-only D2NN는 amplitude modulation을 만들면 안 됨

각 층의 전달함수는 다음이어야 한다.

`t(x, y) = exp(j phi(x, y))`

즉 magnitude는 항상 1이다.
이 원칙은 여전히 유지한다.

의미:

- 위상만 조절
- 증폭/흡수 없음
- passive optical element

따라서 출력 에너지는 propagation/crop/aperture 때문에 바뀔 수는 있어도,
층 자체가 gain을 만들면 안 된다.

### 3. finite window 밖으로 나간 에너지는 물리적으로 손실로 취급

padding 후 center crop한 출력은 "관측 창 내부에 남아 있는 field"다.
crop 뒤에 renormalize를 하면,
창 밖으로 빠져나간 에너지를 인위적으로 다시 집어넣는 셈이 된다.

그래서 이번 구현은 crop 후 재정규화를 하지 않는다.

이 해석은 다음 물리 모델에 맞는다.

- finite receiver window
- finite detector window
- finite computational observation window

### 4. Strehl은 flat-phase diffraction-limited reference로 정의

corrected Strehl은 다음 원칙으로 계산한다.

- reference amplitude는 `|U_vac|` 사용
- reference phase는 flat phase
- zero-padding FFT로 focal spot를 더 촘촘히 샘플링
- total energy를 1로 정규화한 뒤 peak ratio 비교

이 정의를 쓰면 passive system에서 Strehl은 물리적으로 `<= 1`이어야 한다.

이번 수정에서 active path는 이 정의로 바꿨다.

- sweep loss/eval: `kim2026/autoresearch/d2nn_focal_pib_sweep.py`
- sanity checks: `kim2026/src/kim2026/eval/sanity_check.py`

### 5. unitary complex overlap 검증은 같은 연산자 H를 양쪽에 적용해야 함

복소 중첩 불변성은 같은 선형 연산자 `H`를 두 입력에 모두 적용했을 때만 의미가 있다.

올바른 비교:

- `CO(u_turb, u_vac)`
- `CO(H u_turb, H u_vac)`

틀린 비교:

- `CO(H u_turb, D0 u_vac)` 처럼 서로 다른 연산자를 섞는 경우

이번 수정으로 sanity check는 같은 trained model을 양쪽에 적용한다.

## 구현 메모

### alias-safe propagation

- 새 helper는 `propagate_padded_same_window(...)`
- 기본 `pad_factor = 2`
- D2NN 내부 segment에는 보수적 distance guard를 둔다
- 현재 guard 기본값: `z <= 0.05 m`

주의:

- 이 guard는 full sampling theorem solver가 아니다
- `kim2026` D2NN 기본 설정을 보호하는 practical safety belt다
- 누적 target propagation처럼 다른 용도의 경로는 guard를 끄고 같은 padded propagation만 사용한다

### corrected Strehl

active path에서는 legacy `strehl_ratio()` 대신 `strehl_ratio_correct()`를 사용한다.
메모리 사용량 때문에 batch chunking helper를 같이 사용한다.

### pre-training vacuum Strehl 기대값

corrected Strehl 기준에서는 vacuum beam 자체의 curvature 때문에
zero-phase D2NN의 Strehl이 반드시 1.0 근처일 필요가 없다.

따라서 pre-check의 의미는:

- "ideal lens-compensated system인지"가 아니라
- "passive bound 안에서 finite하고 비폭주인지"를 보는 것

## Troubleshooting 가이드

### throughput은 1인데 결과가 수상할 때

먼저 의심할 것:

- periodic wrap-around
- crop 없는 same-window FFT 경로
- unconstrained phase가 Nyquist를 넘는 고주파를 만들었는지

확인:

- edge-localized Gaussian을 전파해서 opposite edge leakage를 측정
- padded 경로와 periodic 경로를 비교

### Strehl이 1보다 크게 나올 때

먼저 확인:

- legacy Strehl 경로를 아직 쓰는지
- flat-phase reference인지
- focal spot가 undersampled인지

원칙:

- 논문/보고서에서 bare `Strehl`라고 쓰기 전에 정의를 명시
- corrected Strehl인지 peak proxy인지 구분

### zero-phase D2NN를 identity baseline으로 해석하고 싶을 때

그렇게 쓰면 안 된다.
zero-phase D2NN는 propagation-only optical stack이지 identity map이 아니다.

identity baseline이 필요하면:

- propagation이 전혀 없는 reference
- 혹은 동일 optical stack의 특정 analytic baseline

을 별도로 정의해야 한다.

## 이번 수정 이후 기본 체크리스트

- `wrap-around` 회귀 테스트가 있어야 함
- D2NN 내부 free-space propagation은 padded 경로를 써야 함
- target generation도 같은 물리 전파를 써야 함
- active Strehl path는 corrected Strehl이어야 함
- sanity check의 CO 검증은 동일 연산자 기준이어야 함

## 한계

이번 수정은 `kim2026` D2NN 경로만 hardening한 것이다.

아직 포함하지 않은 것:

- repo 전체 shared optics API 통합
- 모든 legacy script/report에서 용어 정리
- general alias theorem 기반의 자동 grid solver
- FD2NN 전체 경로에 대한 동일 수준 hardening

따라서 앞으로 물리 audit를 할 때는
`kim2026` D2NN active path와 다른 project/shared path를 분리해서 봐야 한다.
