# Supplementary Figure S7 결과 분석: 광학/물리학적 해석

## 재현 결과 요약

### S7(a) - Performance vs Layer Number (MNIST, 30 epochs, bs=10)

| Layers | Linear Fourier | Single SBN | Multi-SBN |
|--------|---------------|------------|-----------|
| 1      | 54.7%         | 66.1%      | 66.1%     |
| 2      | 87.7%         | 89.2%      | 88.9%     |
| 3      | 89.8%         | 92.4%      | 92.9%     |
| 4      | 90.4%         | 93.9%      | 94.6%     |
| 5      | 90.4%         | 94.5%      | 94.8%     |

### S7(b) - SBN Position (10-layer, per-layer SBN, 30 epochs, bs=500)

| Configuration                       | Test Acc |
|-------------------------------------|----------|
| Nonlinear Fourier, SBN Front        | 96.6%    |
| Nonlinear Fourier, SBN Rear         | 96.6%    |
| Nonlinear Fourier & Real, SBN Front | 97.8%    |
| Nonlinear Fourier & Real, SBN Rear  | 97.8%    |

---

## S7(a): 레이어 수와 성능의 관계

### Linear Fourier가 빠르게 포화되는 이유 (90.4%에서 정체)

Fourier-space D2NN에서 각 diffractive layer는 공간 주파수 영역에서 complex amplitude modulation을 수행한다.
핵심은 **선형 변환의 합성(cascade)은 여전히 선형 변환**이라는 점이다.

```
Layer_1 . Layer_2 . ... . Layer_n  =  하나의 등가 선형 변환 T
```

2f 렌즈 시스템에서 각 레이어는 Fourier plane에서의 amplitude/phase mask이다.
비선형성 없이 이들을 cascade하면, 수학적으로는 **단일 transfer function**과 동등하다.
그래서 2-layer에서 이미 87.7%를 달성한 후 3->4->5에서 거의 개선이 없다.
추가 레이어는 optimization landscape를 더 smooth하게 만들어 학습을 약간 돕지만,
표현력(expressivity) 자체는 늘지 않는다.

### SBN 비선형성이 성능을 올리는 이유

SBN (Strontium Barium Niobate)은 photorefractive 결정으로,
입사광의 intensity에 의존하는 굴절률 변화를 일으킨다:

```
Delta_n  proportional to  |E(x,y)|^2
```

이것이 activation function 역할을 한다.
비선형성이 끼어들면 cascade한 레이어들이 더 이상 단일 선형 변환으로 축소되지 않는다.
각 레이어가 독립적인 feature extraction 단계가 되어,
레이어를 추가할수록 표현력이 실질적으로 증가한다.

### Multi-SBN >= Single SBN인 이유

- **Single SBN**: 비선형성이 맨 마지막에 하나만 있음
  -> 중간 레이어들은 여전히 선형 cascade
  -> 실질적으로 "하나의 선형 변환 + 하나의 비선형 변환"
- **Multi-SBN**: 매 레이어 뒤에 비선형성
  -> 진정한 "deep nonlinear network"
  -> 각 레이어가 독립적으로 비선형 feature를 추출

Neural network에서 activation function을 매 layer에 넣는 것과
마지막에만 넣는 것의 차이와 정확히 같은 원리이다.

---

## S7(b): SBN 위치에 따른 성능 차이

**S7(b)의 4개 구성 모두 10-layer, per-layer SBN 구조이다.**
"Front"과 "Rear"는 네트워크 전체의 앞/뒤가 아니라,
**각 unit cell 내에서 SBN이 D2NN modulation layer의 앞에 오는지, 뒤에 오는지**를 의미한다.

### 4가지 구성의 실제 광학 구조

**1. Nonlinear Fourier, SBN Front (96.6%)**:
```
[Lens → SBN → D2NN → Lens] x10  (Fourier-space only)
```

**2. Nonlinear Fourier, SBN Rear (96.6%)**:
```
[Lens → D2NN → SBN → Lens] x10  (Fourier-space only)
```

**3. Nonlinear Fourier & Real, SBN Front (97.8%)**:
```
[Lens → SBN → D2NN → Lens] x10  (Fourier/Real 교대)
```

**4. Nonlinear Fourier & Real, SBN Rear (97.8%)**:
```
[Lens → D2NN → SBN → Lens] x10  (Fourier/Real 교대)
```

### SBN Front ≈ SBN Rear: Per-layer SBN에서 순서가 거의 무관한 이유

10-layer per-layer SBN 구조에서 흥미로운 결과가 나타났다:
**Front와 Rear의 성능 차이가 거의 없다** (Fourier: 96.6% vs 96.6%, Hybrid: 97.8% vs 97.8%).

이는 매 layer에 SBN이 있는 깊은 비선형 네트워크에서,
unit cell 내 `SBN → D2NN`와 `D2NN → SBN` 순서의 차이가 상쇄되기 때문이다.

- **Per-layer SBN Front** (`SBN → D2NN`): 첫 번째 layer의 SBN은 raw input에 작용하지만,
  두 번째 layer부터는 이전 layer의 D2NN 출력에 SBN이 작용한다.
  즉, layer 경계에서 보면 `... D2NN_prev → SBN_curr → D2NN_curr → ...`로,
  사실상 이전 layer의 D2NN과 현재 layer의 SBN이 연속적으로 `D2NN → SBN` 구조를 형성한다.

- **Per-layer SBN Rear** (`D2NN → SBN`): 각 layer에서 `D2NN → SBN`이 명시적이다.
  Layer 경계에서 `... SBN_prev → D2NN_curr → SBN_curr → ...`이다.

결국 10개 layer가 깊게 쌓이면, 첫 번째와 마지막 layer의 경계 효과를 제외하면
두 구조의 비선형 변환 깊이와 표현력이 사실상 동등해진다.
이는 neural network에서 `BN-ReLU-Conv`(pre-activation ResNet)와
`Conv-BN-ReLU`(post-activation)의 성능이 깊은 네트워크에서 수렴하는 현상과 유사하다.

### Hybrid(97.8%)가 Fourier-only(96.6%)보다 좋은 이유

두 구조 간의 진짜 차이는 **Front/Rear 순서가 아니라 domain 다양성**에서 온다.

Hybrid D2NN은 Fourier-space와 real-space diffractive layer를 교대 배치한다:

- **Fourier-space layer**: 공간 주파수(spatial frequency) 영역에서 modulation
- **Real-space layer**: 공간 위치(spatial position) 영역에서 modulation

이 교대 구조는 빛의 주파수 도메인과 공간 도메인 모두에서 정보를 처리하여,
Fourier optics 관점에서 더 일반적인 unitary transform을 구현할 수 있다.
Fourier-only 구조는 동일 도메인에서만 modulation하므로 표현력이 제한적이다.

Per-layer SBN과 결합하면:
- **Fourier-only + per-layer SBN (96.6%)**: 주파수 영역 필터링 + 비선형성
- **Hybrid + per-layer SBN (97.8%)**: 주파수/공간 교대 필터링 + 비선형성 → 더 풍부한 표현력

---

## 핵심 원리 정리

| 원리 | 설명 |
|------|------|
| 선형 cascade = 단일 선형 변환 | 비선형성 없이는 레이어를 쌓아도 표현력이 늘지 않음 |
| Per-layer SBN에서 순서 무관 | 깊은 네트워크에서 Front/Rear 차이가 layer 경계에서 상쇄됨 |
| Multi-domain 처리가 핵심 차이 | Fourier + Real space 교대(97.8%)가 Fourier-only(96.6%)보다 우수 |
| SBN의 물리적 역할 | Intensity-dependent refractive index change → optical activation function |
| Per-layer 비선형성 | S7(b)의 4개 구성 모두 10-layer, 매 layer마다 SBN 포함 |

광학 신경망 설계의 핵심 원칙:
1. **Per-layer 비선형성**: 매 layer에 SBN을 배치하여 깊은 비선형 네트워크를 구성하는 것이 필수
2. **Multi-domain 처리**: Fourier/Real space를 교대하여 더 일반적인 광학 변환을 구현
3. **SBN 순서보다 구조 다양성**: 깊은 per-layer SBN에서는 Front/Rear 순서보다 domain 교대가 성능에 더 큰 영향
