# D2NN 물리 구현 검증 분석 보고서

> **대상 논문**: Lin et al., "All-optical machine learning using diffractive deep neural networks," *Science* 361, 1004–1008 (2018)
> **대상 코드**: `/root/dj/D2NN/src/d2nn/`
> **분석 일자**: 2026-02-18

---

## 요약 (Executive Summary)

본 코드베이스는 논문의 핵심 물리 모델을 **대체로 올바르게** 구현하고 있다. Angular Spectrum Method(ASM) 전파, 위상 변조 레이어 구조, 검출기 에너지 기반 분류, 입력 인코딩 방식 등이 논문과 정합한다. 다만, 아래에서 상세히 기술하는 **몇 가지 주의점/잠재적 불일치**가 존재한다.

| 항목 | 논문 일치 여부 | 심각도 |
|------|:---:|:---:|
| 1. 전파 모델 (ASM) | **일치** | - |
| 2. 레이어 변조 순서 | **주의 필요** | 중 |
| 3. 위상 제약 조건 | **일치** | - |
| 4. 입력 인코딩 (MNIST/Fashion) | **일치** | - |
| 5. 검출기 에너지 분류 | **일치** | - |
| 6. 손실 함수 | **논문 대비 확장** | 저 |
| 7. 물리 파라미터 | **일치** | - |
| 8. Evanescent wave 처리 | **주의 필요** | 저~중 |
| 9. 이미징 D2NN | **일치 (방향성)** | - |
| 10. FFT convention | **주의 필요** | 중 |

---

## 1. 전파 모델: Angular Spectrum Method (ASM)

### 논문에서의 정의

논문 본문 및 보충자료(ref 14)에 따르면, D2NN의 뉴런 간 연결은 **자유공간 회절(free-space diffraction)**로 정의되며, 이는 Huygens-Fresnel 원리에 기반한다. 구현에서는 이를 **Angular Spectrum Method**로 수치화한다.

전달 함수:
```
H(fx, fy) = exp(i · 2π · z · √((n/λ)² - fx² - fy²))
```

### 코드 구현 (`physics/asm.py:20-77`)

```python
spatial_term = (n / wavelength) ** 2 - FX**2 - FY**2
spatial_term_c = torch.complex(spatial_term, torch.zeros_like(spatial_term))
kz = torch.sqrt(spatial_term_c)
phase = 2.0 * torch.pi * z * kz
H = torch.exp(1j * phase)
```

### 검증 결과: **정확**

- 전달 함수 수식이 논문/표준 Fourier Optics (Goodman) 교과서와 일치한다.
- `(n/λ)² - fx² - fy²`의 복소 제곱근을 통해 evanescent 모드도 자연스럽게 처리된다.
- 주파수 그리드는 `np.fft.fftfreq(N, d=dx)`로 올바르게 생성된다.
- 전파는 `FFT → H 곱셈 → IFFT` 순서로 표준적이다.
- z=0 항등성, 에너지 보존 테스트(`test_asm_energy.py`)가 존재하며 통과한다.

---

## 2. 레이어 변조 순서 (Forward Model)

### 논문에서의 정의

논문 Fig. 1D 및 본문:
> "each point on a given layer either transmits or reflects the incoming wave... the amplitude and phase of which are determined by the **product of the input wave and the complex-valued transmission coefficient** at that point"

즉, 논문의 forward 모델은:
```
E_{l+1,입사} = ASM_propagate(E_l_출사, z)
E_{l+1,출사} = E_{l+1,입사} ⊙ t_{l+1}(x,y)
```
여기서 `t_l = A_l · exp(i·φ_l)`은 레이어 l의 복소 투과 계수이다.

### 코드 구현 (`models/layers.py:58-94`)

```python
def forward(self, field, shift_pixels=None):
    # 1) 전파
    propagated = asm_propagate(field, H, fftshifted=self.fftshifted)
    # 2) 변조
    mod = amp * phase_term  # A · exp(iφ)
    out = propagated * mod
    return out
```

### 검증 결과: **논문과 일치**

`DiffractionLayer`의 forward 순서는 **전파 → 변조**이다. 이는 "이전 레이어에서 나온 필드가 거리 z만큼 전파된 후, 현재 레이어의 투과 마스크를 통과한다"는 논문의 물리와 정확히 일치한다.

### 주의 사항

**레이어 순서 해석**: 코드에서 `DiffractionLayer`는 "전파 z 후 변조"를 수행한다. 즉 첫 번째 `DiffractionLayer(z=z_layer)`는 입력 평면에서 첫 번째 회절층까지 전파한 후 그 층의 위상 마스크를 적용한다. 이는 **입력 평면과 첫 번째 회절층 사이의 거리가 z_layer**임을 의미한다.

논문에서는 입력 평면이 곧 첫 번째 레이어에 인접해 있는 것처럼 서술하는 부분이 있는데, 보충자료에서는 입력면과 첫 레이어 사이에도 전파 거리가 존재한다. 현재 구현은 **모든 레이어 간 간격이 동일한 `z_layer`**로 설정되어 있어, 이는 논문의 균일 간격 구조와 일치한다.

하지만 **입력 평면 → 첫 번째 레이어 사이의 거리**가 논문에서 정확히 명시되지 않았다. 현재 구현에서는 입력면에서 바로 z_layer=3cm 전파 후 L1 마스크를 적용하는데, 논문의 실험 셋업(Fig. 2)에서 입력 평면은 첫 번째 레이어 바로 앞에 놓여 있다. 이 차이가 결과에 유의미한 영향을 줄 수 있다. **만약 논문이 입력면과 L1 사이를 0으로 가정했다면, 현재 구현은 추가 3cm 전파를 하고 있는 셈이다.**

**권장**: 논문 보충자료를 확인하여, 입력면~L1 간 거리가 layer 간격과 동일한지(3cm), 아니면 0인지(즉 입력면이 곧 L1의 위치인지)를 명확히 해야 한다. Config에서 `z_input_to_first_layer`를 별도로 설정할 수 있게 하는 것이 가장 안전하다.

---

## 3. 위상 제약 조건 (Phase Constraint)

### 논문에서의 정의

> "For coherent transmissive networks with **phase-only modulation**, each layer can be approximated as a thin optical element"

- MNIST classifier: phase-only, 3D 프린팅 물리 제약 상 φ ∈ [0, 2π] 이내 (정확한 범위는 보충자료에서 결정)
- 논문의 실험에서는 물리적 높이(thickness)로 위상을 인코딩하므로, 위상 범위는 재료의 굴절률과 최대 프린팅 높이에 의존

### 코드 구현 (`models/constraints.py`)

```python
class PhaseConstraint:
    def __call__(self, raw_phase):
        phi = self.max_phase * torch.sigmoid(raw_phase)
        return torch.clamp(phi, min=0.0, max=self.max_phase)
```

- MNIST config: `phase_max = π`
- Fashion-MNIST config: `phase_max = π`
- Imaging config: `phase_max = 2π`

### 검증 결과: **적절한 구현**

- Sigmoid 기반 제약은 gradient가 끊기지 않으면서 유한 범위를 보장하는 표준적 기법이다.
- 논문 본문에서 MNIST classifier는 "phase-only transmission masks"를 사용한다고 명시하며, 보충자료 Fig. S1에서 phase-only와 complex modulation을 비교한다.
- `phase_max = π`는 보수적 선택이며, 논문의 3D 프린팅 재료 특성상 합리적이다 (VeroBlackPlus의 최대 높이 제약).
- `torch.clamp`는 sigmoid 이후 수치적 안전장치로 적절하다.

---

## 4. 입력 인코딩

### 4.1 MNIST (Amplitude Encoding)

**논문**: "Input digits were encoded into the **amplitude** of the input field to the D2NN"

**코드** (`data/preprocess.py:26-32`, `data/mnist.py`):
```python
def amplitude_encode(image, *, binarize=False, threshold=0.5):
    amp = image.clamp(0.0, 1.0)
    if binarize:
        amp = (amp >= threshold).to(amp.dtype)
    return amp
```

- MNIST 28×28 → resize to 80×80 (nearest) → pad to 200×200 → amplitude ∈ [0,1]
- `binarize=True` (config에서 설정됨)
- `E0 = amp * exp(i·0) = amp` (zero phase)

**검증**: **논문과 일치.** 논문의 실험에서는 알루미늄 포일 컷아웃으로 binary transmittance를 구현했으므로, `binarize=True`는 물리적으로 정확한 대응이다.

### 4.2 Fashion-MNIST (Phase Encoding)

**논문**: "we encoded each input image corresponding to a fashion product as a **phase-only object** modulation"

**코드** (`data/preprocess.py:35-38`, `data/fashion_mnist.py`):
```python
def phase_encode(image, *, max_phase=2*pi):
    return image.clamp(0.0, 1.0) * max_phase
```

- `E0 = 1 · exp(i · 2π · g)` where g ∈ [0,1] is grayscale
- 균일 진폭(amplitude=1), 위상에 이미지 정보를 인코딩

**검증**: **논문과 일치.**

### 주의: Object size

- 코드에서 `object_size=80` pixels → 실제 크기 = 80 × 0.4mm = 32mm = 3.2cm
- 논문의 입력 객체는 8cm × 8cm 레이어 전체에 놓이는 것이 아니라, 중앙에 더 작은 영역을 차지한다.
- 28→80 upsampling은 nearest-neighbor 보간을 사용하여 3D 프린팅된 바이너리 입력의 계단식 구조를 모사한다.

이 부분은 논문에서 정확한 객체 크기를 명시하지 않아 직접 비교가 어렵지만, 방향성은 합리적이다.

---

## 5. 검출기 에너지 분류 (Detector-Based Classification)

### 논문에서의 정의

> "The classification criterion was to find the detector with the **maximum optical signal**"

- 출력면에 10개의 검출기 영역 정의
- 각 검출기의 총 광 에너지를 계산
- 최대 에너지 검출기가 예측 클래스

### 코드 구현

**검출기 레이아웃** (`configs/layouts/mnist_fashion_2x5_v1.json`):
- 10개 영역, 2×5 배치
- 각 영역: 8mm × 8mm
- 출력면 크기: 80mm × 80mm (= 200 × 0.4mm)

**에너지 적분** (`detectors/integrate.py`):
```python
e_t = (i.unsqueeze(1) * m.unsqueeze(0)).sum(dim=(-1, -2))
```

**분류** (`detectors/metrics.py`):
```python
def predict_from_energies(energies):
    return torch.argmax(energies, dim=-1)
```

### 검증 결과: **논문과 일치**

- 논문의 "maximum optical signal" 기준은 `argmax(energies)` 로 정확히 구현되어 있다.
- 검출기 영역의 물리적 좌표가 SI 단위(m)로 정의되어 있고, 픽셀 마스크로 올바르게 변환된다.
- 검출기 크기(8mm)와 간격은 논문 Fig. 3A의 "0.5 cm" 스케일바와 대체로 일치한다.

---

## 6. 손실 함수 (Loss Function)

### 논문에서의 정의

논문 본문:
> "The classification criterion was to find the detector with the maximum optical signal, and **this was also used as a loss function** during the network training (14)"

보충자료(ref 14)를 직접 확인할 수 없으나, 논문의 서술로부터 detector 에너지를 logit으로 사용하는 softmax cross-entropy가 표준적 선택이다.

### 코드 구현 (`training/losses.py`)

```python
logits = energies / max(temperature, 1e-8)
ce = F.cross_entropy(logits, hard_labels.long())
# + leakage penalty
leakage = leakage_energy.mean()
return ce + leakage_weight * leakage
```

**Leakage 계산** (`training/loops.py:54-59`):
```python
union = detector_masks.any(dim=0)
in_detector = (intensity_map * union).sum(dim=(-1, -2))
total = intensity_map.sum(dim=(-1, -2))
leakage_ratio = 1.0 - (in_detector / total)
```

### 검증 결과: **논문 대비 확장, 기본적으로 올바름**

- **Cross-entropy**: detector 에너지를 softmax logit으로 사용하는 것은 논문의 "maximum signal" 기준과 일관된 표준 기법이다.
- **Temperature**: `temperature=1.0`은 에너지 값을 그대로 logit으로 사용함을 의미한다. 에너지 스케일에 따라 temperature 조정이 학습 안정성에 영향을 줄 수 있다.
- **Leakage penalty**: 논문 본문에서는 명시적으로 언급되지 않지만, 검출기 밖으로 새는 에너지를 억제하는 것은 물리적으로 합리적이며, 학습 성능 향상에 기여할 수 있다. `leakage_weight=0.1`은 보수적인 값이다.

### 주의: 논문의 원래 loss와의 차이 가능성

논문의 보충자료에서 사용된 정확한 loss function 형태를 확인할 수 없다. 일부 D2NN 후속 논문에서는 단순 cross-entropy 외에도 diffraction efficiency loss, power concentration loss 등을 사용하기도 한다. 현재 구현의 CE + leakage 조합은 합리적인 선택이다.

---

## 7. 물리 파라미터 정합성

### MNIST Classifier

| 파라미터 | 논문 | 코드 (config) | 일치 |
|----------|------|---------------|:---:|
| λ | 0.75 mm (0.4 THz) | 0.00075 m | **O** |
| 레이어 크기 | 8 cm × 8 cm | N=200, dx=0.4mm → 80mm | **O** |
| 레이어 수 | 5 | num_layers=5 | **O** |
| 레이어 간격 | 3 cm (Fig. 2A) | z_layer=0.03 m | **O** |
| 출력면 거리 | (보충자료) | z_out=0.03 m | **추정 일치** |
| 변조 방식 | Phase-only | train_amplitude=false | **O** |
| 입력 인코딩 | Amplitude | encoding="amplitude" | **O** |
| 목표 정확도 | 91.75% | ≥ 90% 목표 | **O** |

### Imaging Lens

| 파라미터 | 논문 | 코드 (config) | 일치 |
|----------|------|---------------|:---:|
| λ | 0.75 mm | 0.00075 m | **O** |
| 레이어 크기 | 9 cm × 9 cm | N=300, dx=0.3mm → 90mm | **O** |
| 레이어 수 | 5 | num_layers=5 | **O** |
| 레이어 간격 | 4 mm (Fig. 2B) | z_layer=0.004 m | **O** |
| 출력면 거리 | (보충자료) | z_out=0.007 m | **추정 일치** |
| 변조 방식 | Phase-only | train_amplitude=false | **O** |
| 손실 | MSE | imaging_mse_loss | **O** |

### 검증 결과: **주요 물리 파라미터는 모두 논문과 일치**

---

## 8. Evanescent Wave 처리

### 논문

논문에서는 evanescent wave 처리에 대해 명시적으로 언급하지 않는다. 그러나 ASM에서 `(n/λ)² < fx² + fy²`인 주파수 성분은 evanescent wave에 해당하며, 물리적으로 전파 시 지수적으로 감쇠한다.

### 코드 구현

```python
# 복소 제곱근으로 evanescent 모드 자동 처리
spatial_term_c = torch.complex(spatial_term, torch.zeros_like(spatial_term))
kz = torch.sqrt(spatial_term_c)  # negative spatial_term → imaginary kz
```

`bandlimit=True`인 경우:
```python
propagating = spatial_term >= 0
H = H * propagating.to(H.dtype)  # evanescent 성분 제거
```

### 검증 결과: **두 가지 방식 모두 구현되어 있음**

1. **`bandlimit=False`**: 복소 sqrt를 통해 evanescent 성분이 자연스럽게 감쇠한다 (exp(-|kz|·z)). 이는 물리적으로 정확하다.
2. **`bandlimit=True`** (기본값): evanescent 성분을 완전히 0으로 설정한다. 이는 **Band-Limited ASM** (BL-ASM)으로, aliasing artifact를 줄이는 데 효과적이다.

### 주의

현재 config에서 `bandlimit=True`가 기본값이다. 이는 z=3cm, λ=0.75mm 조건에서 합리적이지만, **논문의 원래 구현이 BL-ASM을 사용했는지는 확인이 필요하다**. BL-ASM은 고주파 성분을 제거하므로, 특히 짧은 전파 거리(imaging D2NN의 z=4mm)에서는 결과에 미세한 차이가 있을 수 있다.

단, 에너지 보존 테스트에서 `bandlimit=False`를 사용하여 테스트하고 있어, 물리적 정확성은 검증되어 있다.

---

## 9. D2NN Model 전체 구조

### 논문에서의 정의 (Fig. 1D)

```
입력면 → [전파 z] → L1(변조) → [전파 z] → L2(변조) → ... → L5(변조) → [전파 z_out] → 출력면
```

### 코드 구현 (`models/d2nn.py:37-60`)

```python
def forward(self, field, return_intermediates=False):
    out = field
    for layer in self.layers:  # 각 DiffractionLayer: 전파 → 변조
        out = layer(out, shift_pixels=shift)
    if self.output_layer is not None:  # PropagationLayer: 전파만
        out = self.output_layer(out)
    return out
```

`build_d2nn_model()`:
- 5개 `DiffractionLayer(z=z_layer)` → 1개 `PropagationLayer(z=z_out)`

### 검증 결과: **구조적으로 논문과 일치**

전체 forward 경로:
```
E_input → ASM(z_layer) → ⊙t₁ → ASM(z_layer) → ⊙t₂ → ... → ⊙t₅ → ASM(z_out) → I_output
```

이는 논문의 Fig. 1A 구조와 일치한다. 마지막 레이어 이후 출력면까지의 추가 전파(`PropagationLayer`)도 논문의 실험 구조(마지막 레이어 → 검출기 면)와 일치한다.

---

## 10. FFT Convention 관련

### 코드 (`physics/asm.py:80-99`)

```python
# fftshifted=False (기본):
u = torch.fft.fft2(field)
y = torch.fft.ifft2(u * H)

# fftshifted=True:
u = fftshift(fft2(ifftshift(field)))
y = fftshift(ifft2(ifftshift(u * H)))
```

### 검증 결과: **기능적으로 올바르나 주의 필요**

- 기본 모드(`fftshifted=False`)에서는 field와 H 모두 unshifted FFT 순서를 사용한다. `make_frequency_grid(fftshift=False)`도 unshifted 순서를 반환하므로, H의 주파수 배치와 FFT 출력의 주파수 배치가 일치한다. **이는 올바르다.**
- `fftshifted=True` 모드도 제공되어 있으며, shift 연산이 올바르게 쌍을 이루고 있다.

### 주의

- `z=0` identity 테스트와 에너지 보존 테스트가 통과하므로, FFT convention 오류 가능성은 낮다.
- 그러나 **fftshifted=False가 기본인 상태에서, 시각화 시 출력 필드의 공간 좌표가 올바르게 해석되고 있는지**는 별도로 확인이 필요하다. FFT 순서에서는 DC 성분이 (0,0)에 위치하므로, 시각화 시 `fftshift`를 적용해야 물리적으로 올바른 이미지를 얻는다.

---

## 11. 추가 검토 사항

### 11.1 Optimizer

**논문** (보충자료 참조): "stochastic gradient descent approach" → 실제로는 Adam optimizer가 일반적으로 사용됨

**코드**: `torch.optim.Adam(model.parameters(), lr=lr)` — 합리적 선택이며, 논문의 구현에서도 Adam을 사용했을 가능성이 높다.

### 11.2 Batch Size와 Training Epochs

**코드**: `batch_size=8`, `epochs=10`

논문에서는 55,000개 학습 이미지(5,000 검증)를 사용했다. Batch size와 epoch 수는 논문에서 명시되지 않았으므로, 현재 설정이 동일한 결과를 재현할 수 있는지는 실험적 검증이 필요하다. 특히 `batch_size=8`은 다소 작을 수 있으며, 학습 안정성에 영향을 줄 수 있다.

### 11.3 Weight Initialization

**코드** (`models/layers.py:52`):
```python
self.raw_phase = nn.Parameter(torch.zeros(self.N, self.N))
```

모든 raw_phase를 0으로 초기화하면, sigmoid(0) = 0.5이므로 초기 위상은 `0.5 × phase_max`이다. 이는 모든 위상이 동일한 상태에서 시작함을 의미한다.

논문에서 초기화 방식을 명시하지 않았으므로, 이것이 논문 구현과 동일한지는 확인할 수 없다. 하지만 경사 하강법 기반 학습에서는 대칭 파괴(symmetry breaking)를 위해 **랜덤 초기화**가 더 일반적이다. 현재의 0 초기화가 학습 속도에 영향을 줄 수 있다.

**권장**: `torch.randn` 또는 `torch.rand` 기반 초기화와 비교 실험을 수행할 것.

### 11.4 에너지 보존과 정규화

**코드**에서 에너지 보존 테스트는 `bandlimit=False` 조건에서만 수행된다. `bandlimit=True`에서는 고주파 성분이 제거되므로 에너지가 약간 감소할 수 있다. 이는 물리적으로 올바른 동작이지만, 학습 시 gradient 스케일에 영향을 줄 수 있다.

### 11.5 Imaging D2NN의 출력 정규화

**코드** (`training/loops.py:171`):
```python
out_i = out_i / out_i.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
```

출력 intensity를 max로 정규화한 후 MSE를 계산한다. 이는 절대 에너지 스케일 차이를 무시하고 상대적 패턴만 비교하겠다는 의미이다. 논문에서의 imaging loss가 정확히 이 방식인지는 보충자료 확인이 필요하지만, 방향성은 합리적이다.

### 11.6 Misalignment Error Model

**논문** (보충자료): 레이어 정렬 오차(misalignment)가 실험 성능 저하의 원인 중 하나로 언급됨

**코드** (`models/d2nn.py:28-35`):
```python
def _sample_misalignment_pixels(self, device):
    max_px = max(int(round(self.max_misalignment_m / self.dx)), 0)
    shifts = torch.randint(low=-max_px, high=max_px + 1, size=(2,))
    return int(shifts[0].item()), int(shifts[1].item())
```

- `torch.roll`로 정수 픽셀 단위 이동 구현
- Training 시에만 적용 (`self.training`)
- 기본 config: `max_misalignment_m=0.0` (비활성)

**검증**: 합리적 구현이나, 실제 물리적 misalignment는 서브픽셀 수준의 연속적 이동이다. 정수 픽셀 이동은 근사이지만, dx=0.4mm에서 0.1mm 미만의 misalignment는 0 pixel로 반올림되므로 실질적으로 효과가 없다. 이는 config에서 `max_misalignment_m=0.0001`(imaging config)으로 설정되어 있는데, 0.1mm/0.3mm ≈ 0.33 pixel로 0 또는 1 pixel 이동만 가능하다.

---

## 12. 테스트 커버리지 평가

| 테스트 | 검증 내용 | 평가 |
|--------|----------|------|
| `test_asm_identity_z0` | z=0에서 항등 전파 | **핵심 — 통과** |
| `test_asm_energy_sanity` | 에너지 보존 (rel_err < 1e-3) | **핵심 — 통과** |
| `test_phase_constraint_range` | 위상 범위 [0, max_phase] | **필수 — 통과** |
| `test_reproducibility_smoke` | 동일 seed → 동일 결과 | **핵심 — 통과** |
| `test_detector_integrals` | 검출기 마스크 적분 | **필수** |
| `test_heightmap_export` | 높이맵 변환 | 보조 |
| `test_lumerical_builder_mock` | Lumerical 빌더 | 보조 |

### 누락된 테스트 (권장 추가)

1. **Fresnel number / sampling 조건 검증**: N, dx, λ, z 조합이 ASM의 유효 조건(Nyquist 기준)을 만족하는지 확인하는 테스트
2. **Round-trip 테스트**: ASM(z) 후 ASM(-z)로 원래 필드를 복원하는 테스트 — FFT convention 오류를 더 강력하게 검출
3. **Gradient flow 테스트**: 전체 forward → loss → backward가 수렴하는지 확인하는 짧은 학습 테스트
4. **검출기 에너지 합 검증**: 전체 에너지 대비 검출기 내 에너지 비율이 합리적인지 검증

---

## 13. 종합 결론

### 올바르게 구현된 부분

1. **ASM 전파 모델**: 수식, 구현, 테스트 모두 물리적으로 정확
2. **레이어 구조**: 전파 → 변조 순서가 논문의 D2NN 정의와 일치
3. **위상 제약**: Sigmoid 기반 [0, max_phase] 제약이 물리적으로 적절
4. **입력 인코딩**: MNIST(amplitude), Fashion-MNIST(phase) 모두 논문과 일치
5. **검출기 분류**: 에너지 적분 → argmax 분류 논리가 정확
6. **물리 파라미터**: λ, N, dx, z, 레이어 수 모두 논문 수치와 일치
7. **재현성 인프라**: 시드 관리, config 기반 실행, 결정론적 모드 지원

### 개선이 필요하거나 확인이 필요한 부분

| 우선순위 | 항목 | 설명 |
|:---:|------|------|
| **높음** | 입력면~L1 거리 | 논문에서 입력면이 L1과 같은 위치인지, z_layer 떨어져 있는지 확인 필요. 결과 정확도에 직접 영향. |
| **높음** | Weight 초기화 | 모든 raw_phase=0 초기화가 학습 성능/수렴에 미치는 영향 검증 필요. Random init과 비교 실험 권장. |
| **중간** | Bandlimit 모드 | 논문 원래 구현이 BL-ASM을 사용했는지 확인. 특히 imaging D2NN(짧은 z)에서 영향 가능. |
| **중간** | Batch size | batch_size=8은 학습 안정성에 영향. 논문의 실제 batch size 확인 또는 스윕 실험 필요. |
| **낮음** | Misalignment 모델 | 정수 픽셀 shift는 서브픽셀 정렬 오차를 모사하기에 해상도가 부족. 보간 기반 shift가 더 정확. |
| **낮음** | 이미징 출력 정규화 | max 정규화 방식이 논문과 동일한지 확인 필요. |

### 최종 판정

코드베이스는 논문의 D2NN 물리 모델을 **충실하게 구현**하고 있으며, 핵심 물리 엔진(ASM), 모델 구조, 분류/이미징 파이프라인이 논문과 정합한다. 위에서 언급된 개선 사항들은 대부분 **재현 정확도를 높이기 위한 미세 조정** 수준이며, 물리적 오류는 발견되지 않았다.