저장 시도는 했지만 현재 워크스페이스가 `read-only`라 `autoresearch/runs/PAPER_REVIEW_GPT54.md`를 만들 수 없었습니다. 아래가 요청하신 전체 리뷰 내용입니다.

```markdown
# 제2 독립 심사 의견: Static D2NN Limits in Atmospheric Turbulence Compensation

**Reviewer**: Anonymous  
**Recommendation**: Major Revision

## 총평

이 원고의 중심 아이디어는 분명히 중요합니다. 정적이고 손실이 없는 phase-only D2NN은 필드 수준의 유사도나 거리 자체를 개선할 수 없고, 대신 검출 평면에서의 세기 재분배를 통해 intensity-based metric만 바꿀 수 있다는 주장은 자유공간 광통신과 수동 회절 광학의 한계를 정리하는 데 의미가 있습니다.

다만 현재 원고는 다음 세 층위를 엄밀히 분리하지 못합니다.

1. 이상적 연속계에서의 unitary 정리
2. 이산 sampled angular-spectrum 모델에서의 수치 구현
3. 실제 물리 시스템으로의 해석

특히 가장 심각한 문제는, **정리 1-3의 “실험적 검증”이 실제로는 정리가 보존한다고 말한 양을 측정하지 않았을 가능성**이 크다는 점입니다. 여기에 더해 WF RMS 정리의 적용 범위, PIB 247배 해석, Poynting flux/NA/etendue 관점의 부재가 주요 약점입니다.

---

## 1. 수학적 평가: 정리 1-4는 맞는가?

### 정리 1: 내적 보존
이 정리는 맞습니다. 손실 없는 위상 마스크와 unit-modulus propagation operator의 합성으로 이루어진 연산자 `H`가 유니터리이면

\[
\langle HU_1, HU_2 \rangle = \langle U_1, U_2 \rangle
\]

는 표준 결과입니다. 따라서 **동일한 연산자 `H`를 두 필드 모두에 적용했을 때** complex overlap이 보존된다는 결론도 맞습니다.

하지만 숨은 가정이 있습니다.

- 두 필드는 같은 힐베르트 공간에 있어야 합니다.
- 같은 sampling, 같은 window, 같은 비교 평면에서 정의되어야 합니다.
- 비교는 반드시 `CO(HU_1, HU_2)`여야 합니다. `CO(HU_1, U_2)`는 전혀 다른 양입니다.

이 마지막 점이 본 원고에서 결정적으로 중요합니다.

### 정리 2: L2 거리 보존
이 정리도 맞습니다. 유니터리 연산자는 L2 norm과 L2 distance를 보존합니다.

\[
\|HU_1 - HU_2\|_2 = \|U_1 - U_2\|_2
\]

그러나 원고의 **따름정리 2a, 즉 WF RMS 보존 주장**은 현재 형태로는 엄밀하지 않습니다.

문제는 세 가지입니다.

- 유도는 `e^{i\Delta\phi} \approx 1 + i\Delta\phi`라는 small-aberration 근사에 의존합니다.
- 논문 조건 `D/r_0 = 5.02`는 weak aberration이 아닙니다.
- 실제 split-step 난류 전파는 amplitude scintillation을 생성하므로 `U_turb = U_vac e^{i\phi}` 형태의 순수 phase-only 모델이 시뮬레이션과 일치하지 않습니다.

따라서 현재 원고는 `L2 distance` 보존과 `WF RMS` 보존을 사실상 같은 것으로 취급하지만, 이는 일반적으로 성립하지 않습니다. 제 판단은 다음과 같습니다.

- 정리 2 자체는 맞습니다.
- WF RMS 보존은 정리가 아니라, 약한 위상 오차에서만 가능한 근사적 휴리스틱입니다.
- 따라서 이 부분은 강하게 축소 서술해야 합니다.

### 정리 3: 랜덤 난류에서 기대 CO 보존
이것은 사실상 정리 1의 직접적 corollary입니다. 맞긴 하지만 별도의 정리로 세울 정도의 독립성은 약합니다.

또한 `zero-mean random process` 가정은 불필요합니다. realization-by-realization로 보존되면 어떤 분포든 기대값은 보존됩니다. 즉

- 영평균 가정은 핵심이 아닙니다.
- phase-only 난류 가정도 핵심이 아닙니다.
- 이 정리는 arbitrary complex field ensemble에 대해 더 일반적으로 쓸 수 있습니다.

### 정리 4
본문에는 정리 1-3과 2-pixel 반례만 명시적으로 보이고, 정리 4는 정식 statement가 없습니다. numbering과 논리 구조를 정리해야 합니다.

---

## 2. 물리적 타당성: D2NN은 정말 유니터리인가?

### 이상적 연속계에서는 그렇다
연속 scalar wave optics에서, 손실 없는 phase-only mask와 자유공간 전파는 propagating-mode subspace 위에서 유니터리입니다. 이 점은 맞습니다. 즉, 이상화된 Helmholtz/angular-spectrum 모델 안에서의 receiver-side phase-only D2NN은 unitary operator로 취급 가능합니다.

### 그러나 실제 시스템 전체는 유니터리가 아니다
원고가 다루는 실제 시스템 전체는 다음 이유로 유니터리라고 말할 수 없습니다.

- 15 cm telescope aperture의 hard clipping
- 4096 격자에서 1024 격자로의 crop
- 75:1 beam reducer를 단순 좌표 재해석으로 처리한 점
- 실제 소자에서의 absorption, scatter, phase quantization
- layer 간 misalignment, tilt, piston 오차
- 단색/스칼라 근사, 편광 무시
- detector NA 또는 fiber acceptance 미모델링

따라서 원고의 문장은

> “D2NN 전체 연산자 H는 유니터리이다”

가 아니라

> “수신 aperture 이후에 정의된 sampled, lossless, scalar, monochromatic phase-only D2NN block은 해당 이산 모델 내에서 유니터리이다”

로 한정되어야 합니다.

### discrete angular spectrum 자체는 이 논문 설정에서는 거의 정확히 unitary다
이 점은 1차 리뷰가 충분히 구분하지 못한 부분입니다. receiver-side propagation은 `dx = 2 µm`, `λ = 1.55 µm`이므로

\[
f_{\mathrm{Nyquist}}=\frac{1}{2dx}=2.5\times 10^5\ \mathrm{m^{-1}}
\]

이고

\[
\frac{1}{\lambda}\approx 6.45\times 10^5\ \mathrm{m^{-1}}
\]

입니다. 즉 샘플된 모든 spatial frequency가 propagating regime 안에 있으므로, **evanescent cutoff는 실제로 작동하지 않습니다.** 또한 `norm="ortho"` FFT와 unit-modulus transfer function을 쓰면, 이산 sampled torus 위의 propagation operator는 roundoff를 제외하면 거의 정확히 unitary입니다.

실제로 제공된 결과 생성 코드와 일치하는 모델을 확인하면, `CO(U_t,U_v)`와 `CO(HU_t,HU_v)`의 차이는 대략 `10^-6 ~ 10^-5`, L2 distance 보존 오차도 `10^-4` 이하입니다. 즉

- 수치적 D2NN block 자체는 거의 확실히 unitary입니다.
- 문제는 “유니터리가 아니어서 정리가 흔들린다”가 아니라, **그 unitary block이 실제 물리 receiver를 얼마나 충실히 대표하느냐**입니다.

### Poynting vector와 detector physics
원고는 detector-plane `|U|^2`를 바로 power proxy로 사용합니다. paraxial scalar regime에서는 흔히 허용되지만, 만약 D2NN이 큰 각도의 공간 주파수를 사용한다면 실제 수광 power는 `S_z` 또는 detector NA를 반영해 평가해야 합니다. 즉

- `PIB = \int_B |U|^2 dA`는 평면 위 irradiance metric이지,
- 곧바로 실제 수신기 coupling efficiency 또는 brightness improvement를 뜻하지는 않습니다.

이 논문은 바로 이 점을 건너뛰고 있습니다.

---

## 3. PIB 247배 주장은 물리적으로 의미가 큰가?

제 판단은 **“수치적으로는 참이지만, 현재 해석은 과장되어 있다”**입니다.

### 상대 배율은 매우 부풀려져 있다
기준선이 `0.34%`이므로 247배라는 숫자는 자동적으로 커집니다. 상대배율만 전면에 내세우면 오해를 부를 수 있습니다. 절대값 `0.0034 -> 0.8340`을 우선 제시해야 합니다.

### 이것은 난류 보정보다 spatial filtering에 가깝다
원고가 스스로 보여주듯 PIB-only에서 CO는 `0.304 -> 0.016`으로 붕괴합니다. 이는 파면 복원이나 vacuum beam recovery가 아니라, **출력 평면의 특정 bucket에 에너지를 몰아주는 공간 재분배**입니다. 즉 이 장치는 wavefront corrector라기보다 **bucket-optimized passive mode filter**로 읽는 편이 맞습니다.

### 83.4%는 사실상 렌즈 수준의 집광과 비교되어야 한다
이 논문 구조에서 receiver aperture는 약 2 mm이고 detector plane은 마지막 mask 뒤 10 mm입니다. 만약 마지막 phase mask 하나가 단순 thin-lens 역할만 해도, diffraction-limited Airy 첫 영점 반경은 대략

\[
r_1 \approx 1.22\frac{\lambda f}{D} \approx 9.5\ \mu m
\]

입니다. 50 µm bucket은 약 `5.3` Airy 반경에 해당하며, 이상적 렌즈의 encircled energy는 약 **96.8%**까지 가능합니다. 따라서 현재의 `83.4%`는 “놀라운 초집광”이 아니라, **오히려 단순 집광 광학과 비교해야 할 수치**입니다.

즉, 이 결과는 다음을 증명하지 않습니다.

- 정적 D2NN이 난류를 복원한다.
- 정적 D2NN이 ordinary focusing optics를 넘어서는 practical receiver gain을 준다.

현재 결과가 증명하는 것은 더 제한적입니다.

- 정적 D2NN은 특정 검출 평면, 특정 bucket radius에 대해 power-in-bucket을 크게 최적화할 수 있다.

그 이상을 말하려면 lens baseline, NA-limited detector, fiber coupling, mode overlap이 필요합니다.

---

## 4. 1차 리뷰가 놓친 가장 치명적인 약점

제가 보기에 1차 리뷰가 놓친 가장 중요한 문제는 다음입니다.

### 원고의 “정리 1-3의 실험 검증”이 실제로는 정리가 보존한다고 말한 양을 측정하지 않았을 가능성
정리 1이 보장하는 것은

\[
CO(HU_{turb}, HU_{vac}) = CO(U_{turb}, U_{vac})
\]

입니다. 즉, **두 필드 모두에 같은 `H`를 적용한 뒤 비교**해야 합니다.

그런데 원고의 결과 표와 서술은 `baseline CO`와 `시험 CO`를 비교하지만, 실제 어떤 쌍을 비교했는지가 모호합니다. 만약 현재 결과가

- baseline: `CO(U_turb, U_vac)`
- after D2NN: `CO(HU_turb, U_vac)`

을 비교한 것이라면, 이것은 **정리 1의 검증이 아닙니다.** 그리고 이 양은 얼마든지 변할 수 있습니다.

제공된 결과 생성 코드와 동일한 프로토콜을 기준으로 직접 확인하면, 실제로

- `CO(U_t,U_v)`와 `CO(HU_t,HU_v)`는 평균적으로 `10^-6 ~ 10^-5` 수준에서만 달라집니다. 즉 정리 1은 수치적으로 거의 정확히 성립합니다.
- 반면 `CO(HU_t,U_v)` 같은 mixed comparison은 PIB-only 모델에서 평균적으로 약 `-0.31`까지 크게 변합니다.

즉

- 원고의 CO 변화는 정리 위반의 증거도 아니고,
- 정리의 실험 검증도 아니며,
- **서로 다른 평면/서로 다른 operator를 적용한 field를 비교한 결과일 가능성**이 높습니다.

동일한 문제가 WF RMS에도 있습니다. 정리 2가 보존하는 것은 `||HU_t - HU_v||_2`이지, `HU_t`와 zero-phase reference 사이의 wrapped phase RMS가 아닙니다. 따라서 현재 원고의 WF RMS 실험도 정리 2의 직접 검증이라고 부르기 어렵습니다.

이 문제는 단순한 presentation issue가 아닙니다. **현재 실험 섹션 전체의 논리적 지위를 바꿉니다.** 제가 보기에 이것이 현 단계의 가장 치명적인 문제입니다.

---

## 5. 추가적인 핵심 물리 코멘트

### “검출 비선형성이 유니터리 불변성을 깬다”는 서술은 물리적으로 부정확하다
엄밀히 말하면 detector가 upstream optical evolution의 unitarity를 깨는 것이 아닙니다. 바뀌는 것은 **측정 observable**입니다. PIB는

\[
\mathrm{PIB}(U)=\frac{\langle U,BU\rangle}{\langle U,U\rangle}
\]

꼴의 quadratic observable이며, unitary `H` 이후에는

\[
\mathrm{PIB}(HU)=\frac{\langle U,H^\dagger B H U\rangle}{\langle U,U\rangle}
\]

가 됩니다. 즉 변화의 본질은 detector nonlinearity가 유니터리를 파괴해서가 아니라, **PIB가 원래부터 unitary invariant observable이 아니기 때문**입니다.

### phase-only turbulence narrative는 불필요하게 좁고, 시뮬레이션과도 불일치한다
정리 1-3은 amplitude scintillation이 있어도 성립합니다. 정리는 입력 field의 형태가 아니라 `H`의 unitary 성질에 달려 있기 때문입니다. 오히려 원고가 `U_turb = U_vac e^{i\phi}`를 전면에 내세우면서 split-step simulation으로 amplitude fluctuation까지 생성하는 것이 혼란을 만듭니다.

원고는 다음처럼 고치는 편이 낫습니다.

- 수학 부분: arbitrary complex field ensemble로 일반화
- 물리 부분: 본 시뮬레이션은 amplitude scintillation을 포함한다고 명시

### beam reducer의 물리 모델이 지나치게 축약되어 있다
75:1 beam reducer를 단순 좌표 재해석으로 모델링하는 것은 이상적 relay optics의 rough surrogate일 수는 있지만, 실제 afocal relay의 위상/NA/aperture/정렬 오차를 반영하지 못합니다. receiver-side D2NN block에 대한 정리는 엄밀히 세우려 하면서, 그 앞단 relay optics는 지나치게 가볍게 처리하고 있습니다.

---

## 6. 논문을 실질적으로 강화할 추가 실험

1. **정리 1의 올바른 수치 검증**  
   `CO(U_t,U_v)`와 `CO(HU_t,HU_v)`를 동일 test set에서 직접 비교하십시오.

2. **정리 2의 올바른 수치 검증**  
   `||U_t-U_v||_2`와 `||HU_t-HU_v||_2`를 직접 비교하십시오. WF RMS는 별도 heuristic metric으로 분리하십시오.

3. **WF RMS 주장 축소 또는 재정의**  
   현재 moderate turbulence에서는 rigorous theorem으로 쓰면 안 됩니다. weak-turbulence sweep에서만 phase-only 근사로 보이거나, 아니면 아예 `L2 distance`와 분리하십시오.

4. **얇은 렌즈 baseline 추가**  
   마지막 mask 하나에 quadratic lens phase만 준 경우, 혹은 독립 thin lens를 둔 경우와 PIB를 비교하십시오.

5. **SMF/MMF coupling efficiency 추가**  
   실제 수신기 성능을 주장하려면 bucket power가 아니라 fiber mode overlap 또는 detector NA-limited collected power를 보여야 합니다.

6. **Poynting/NA 기반 검증**  
   output angular spectrum을 계산해 high-angle content를 제시하고, finite-NA detector를 통과한 power와 `S_z` 기반 bucket power를 함께 보고하십시오.

7. **bucket radius 및 detector plane sweep**  
   10, 25, 50, 100 µm와 detector distance sweep을 통해 현재 결과가 특정 평면/특정 bucket에 과도하게 맞춰진 것인지 확인하십시오.

8. **phase-only vs amplitude+phase turbulence control**  
   동일 `D/r0`에서 pure phase screen only case와 full split-step scintillation case를 분리해, theorem과 practical metric 사이의 차이를 명확히 하십시오.

9. **fabrication tolerance study**  
   8-level/16-level quantization, layer misalignment, axial spacing error, absorption을 넣은 robustness 분석이 필요합니다.

10. **etendue 또는 brightness 관련 지표**  
    최소한 angular spread와 spatial concentration을 함께 보고하여, 결과가 brightness increase로 오해되지 않게 하십시오.

---

## 7. 문장 및 주장 수정 제안

- “D2NN 전체 연산자 H는 유니터리이다”  
  → “수신 aperture 이후에 정의된 lossless sampled phase-only D2NN block은 이상적 scalar monochromatic 모델에서 유니터리이다.”

- “WF RMS가 보존됨을 수학적으로 증명하였다”  
  → “L2 distance 보존을 증명하였고, WF RMS는 약한 위상 오차 조건에서만 근사적으로 연관된다.”

- “광검출의 비선형성이 유니터리 불변성을 깬다”  
  → “PIB와 같은 intensity-based observable은 본질적으로 unitary invariant가 아니므로, 정적 D2NN으로도 값이 변할 수 있다.”

- “247배 개선”  
  → 절대값과 baseline 조건, bucket 정의, lens baseline을 함께 제시해야 합니다.

---

## 최종 판정

**판정: Major Revision**

이 원고는 아이디어 자체는 좋고, 특히 “정적 수동 유니터리 광학계는 field-level turbulence correction을 할 수 없다”는 메시지는 충분히 출판 가치가 있습니다. 그러나 현재 형태로는

- 정리와 실험 검증 사이의 양(quantity) 매칭이 불명확하거나 잘못되었고,
- WF RMS 주장에 과도한 수학적 무게가 실려 있으며,
- PIB 247배의 practical significance가 충분히 정당화되지 않았고,
- Poynting flux/NA/relay optics/fabrication을 포함한 실제 광수신기 해석이 부족합니다.

핵심 정리는 살릴 수 있지만, **실험 섹션의 재정의와 metric의 재검증이 필수**입니다. 만약 위의 “정리 검증 quantity mismatch”가 실제 결과 생성 프로토콜에서 확인된다면, 이는 단순 보완이 아니라 원고 중심 논리의 재작성까지 요구하는 수준입니다. 그럼에도 불구하고 중심 아이디어는 버릴 필요가 없으므로, 현 단계의 적절한 판단은 **Major Revision**입니다.
``` 

워크스페이스가 쓰기 가능해지면 같은 내용을 바로 `autoresearch/runs/PAPER_REVIEW_GPT54.md`에 기록할 수 있습니다.