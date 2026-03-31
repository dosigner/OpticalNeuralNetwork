# Physics Reassessment — 2026-03-31

## 결론

이 run은 **보조 참고자료로는 유지 가능**하지만, **현재 kim2026 물리 기준의 최종 결과로는 인용하면 안 된다.**
현재 기준으로는 `focal_pib_only`를 다시 학습해야 한다.

## 왜 재학습이 필요한가

이유는 단순히 메트릭 이름만 고친 것이 아니라, **학습 당시의 forward physics 자체가 바뀌었기 때문**이다.

### 1. 학습 당시 전파 연산자가 현재와 다름

이 checkpoint는 same-window FFT 기반의 **periodic propagation**으로 학습됐다.
현재 kim2026 D2NN은 `zero-padding -> propagation -> center crop` 기반의
**alias-safe padded propagation**을 사용한다.

즉, 학습 당시 최적화한 목적함수는:

- 옛 연산자 `H_old` 아래의 PIB/Strehl

현재 우리가 원하는 목적함수는:

- 새 연산자 `H_new` 아래의 PIB/Strehl

이다. 일반적으로 `argmin_theta L(H_old(theta))` 와
`argmin_theta L(H_new(theta))` 는 다르다.

따라서 후처리로 메트릭만 다시 계산해도,
이 checkpoint가 현재 물리 모델의 최적해가 되지는 않는다.

### 2. legacy focal Strehl 값은 최종 수치로 사용할 수 없음

`04_training_results_full.json`의 legacy 값:

- `focal_strehl = 28.780078887939453`

이 값은 legacy Strehl 정의의 산출물이며,
현재 corrected Strehl 정의와 직접 비교할 수 없다.

### 3. corrected Strehl은 후처리 참고값일 뿐, 재학습 대체물이 아님

동일 checkpoint를 corrected Strehl로 다시 평가한 값은 다음과 같다.

- `vacuum strehl_correct = 0.016384214162826538`
- `turbulent strehl_correct = 0.04744464773980398`
- `d2nn strehl_correct = 0.5463925112807562`

이 값은 "옛 checkpoint를 새 기준으로 재채점한 결과"다.
하지만 checkpoint 자체는 여전히 옛 forward model에서 학습된 것이다.

## 현재 해석 기준

- `PIB@10um = 0.8128` 수준의 강한 중심 집중 경향은 참고 가능
- `CO` 붕괴와 `PIB` 집중의 trade-off도 참고 가능
- 그러나 최종 표/논문/공식 비교에는 **재학습 run**을 사용해야 함

## 기록

이 run 폴더의 메인 요약 JSON에 다음 항목을 보강했다.

- `focal_strehl_correct_recomputed`
- `physics_hardening_status`
- `retrain_required_current_physics`
- `retrain_reason`

즉, 이 폴더는 이제 다음처럼 읽어야 한다.

> 구 run artifact + 2026-03-31 물리 기준 재평가 기록
