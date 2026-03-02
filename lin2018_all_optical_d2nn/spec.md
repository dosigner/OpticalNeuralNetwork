**Lin et al., “All‑optical machine learning using diffractive deep neural networks” (Science 2018)**의 본문/보충자료에 기재된 **전파 모델(Angular Spectrum), 학습(오차역전파+Adam), 파라미터(λ=0.75 mm@0.4 THz, 픽셀 0.4 mm/0.3 mm, 레이어 간격 3 cm/4 mm 등), 결과(예: MNIST 91.75%)**를 “재현 가능한 시뮬레이션 코드”로 다시 만들기 위한 PRD/아키텍처/시각화/에이전트 지침을 포함합니다.  
---

## 내가 이렇게 설계한 절차 (step‑by‑step)

1. **논문에서 “재현해야 하는 산출물”을 먼저 고정**:
   (a) MNIST 5‑layer phase‑only 분류 정확도(≈91.75%), (b) Fashion‑MNIST 분류(phase encoding), (c) ImageNet 기반 amplitude imaging lens(SSIM/해상도), 그리고 (d) 보충자료의 wave‑propagation 시각화/architecture sweep(SSIM vs layer 수/간격).
2. **현재 ipynb 코드의 핵심 컴포넌트 분해**:
   Angular spectrum 전파(FFT), phase mask 학습 변수, detector region 집계, height map 변환, Lumerical geometry 생성.
3. **재현성 중심으로 패키지화(모듈/CLI/config/seed/test)**:
   연구 코드가 아니라 “논문 그림/숫자 재현”에 최적화된 구조로 PRD/코드트리/테스트/시각화 API를 정의.
4. **시각화는 ‘그림을 그리는 함수’ + ‘해석 가이드’로 분리**:
   동일한 figure를 반복 생성할 수 있어야 해서, plotting 함수에 **표준 입력/출력, 스케일, 색상범위, annotation 규칙**을 강제.
5. **Codex(또는 에이전트)가 그대로 구현할 수 있도록 AGENT.md에 규칙/체크리스트/DoD 정의**.

---

## 문서 파일 목록 (이대로 repo에 생성)

* `docs/PRD.md`
* `docs/CODETREE.md`
* `docs/CORE_API.md`
* `docs/VISUALIZATION.md`
* `docs/COMMENTING_STYLE.md`
* `docs/REPRODUCTION_PROTOCOL.md`
* `docs/LUMERICAL_INTEGRATION.md`
* `AGENT.md`

아래부터 파일별 내용입니다.

---

## File: `docs/PRD.md`

```md
# D2NN Reproducible Simulation PRD

## 1. 목적 (Why)
Science 2018 "All‑optical machine learning using diffractive deep neural networks(D2NN)"의
핵심 원리/수치 결과를 재현 가능한 형태로 다시 구현한다.

본 프로젝트의 "재현"은 다음을 의미한다:
- 동일(또는 논문에 근접한) 물리 파라미터 + 동일한 forward 모델(각 레이어: 전파 + 복소 변조) 사용
- 동일한 평가 파이프라인(Detector energy 기반 classification, SSIM/MSE 기반 imaging) 사용
- 고정 seed/config로 여러 번 실행해도 동일한 결과 파일/figure가 생성됨

## 2. 범위 (Scope)

### 2.1 반드시 포함 (Must-have)
A) 공통 물리 엔진
- Coherent scalar wave 모델 (complex field E(x,y))
- Free-space propagation: Angular Spectrum Method(ASM)
- Layer modulation: phase-only + (옵션) complex modulation(amplitude+phase)
- Output plane intensity: I = |E|^2

B) 논문 재현 목표 실험 (Simulation)
1) MNIST digit classifier
- N=200 grid, dx=0.4 mm, layer count=5
- layer spacing=3 cm, output distance=3 cm (configurable)
- 목표: blind test accuracy ~91.75% (phase-only 5-layer 기준)

2) Fashion-MNIST classifier
- 입력을 phase channel로 encoding (0~2π)
- 목표: 논문에서 보고된 수준(예: 5-layer phase-only ~81% 근처) 재현

3) Imaging lens D2NN (amplitude imaging)
- N=300 grid, dx=0.3 mm, layer count=5
- layer spacing=4 mm, output distance=7 mm
- Loss: output intensity vs target image MSE
- 평가: SSIM 및 resolution chart/ pinhole PSF 유사 지표

C) 재현성/운영성
- YAML(또는 JSON) config 기반 실행
- seed 고정 + deterministic 옵션
- 결과 폴더 구조(artifact) 표준화
  - weights, height maps, metrics.json, figures/*.png
- 최소한의 unit test(전파 보존/에너지 보존/shape 체크)
- 표준 시각화 모듈(논문 figure 대응)

### 2.2 선택(Option)
- Error sources 모델링(보충자료: misalignment 0.1mm, absorption, Poisson recon error 등)
- “Lego-like patching”(기존 5-layer 고정 + 추가 레이어 학습)
- Lumerical FDTD용 height map -> geometry 생성 자동화

## 3. 비범위 (Non-goals)
- THz 실험 하드웨어(AMC/lock-in/XY stage) 재현
- 다파장/편광/벡터 맥스웰 전자기 전체 해석(기본은 scalar ASM)
- 논문과 완전히 동일한 training time/환경(Pytorch, Python3.10)까지 동일화

## 4. 사용자 스토리 (User Stories)
- 연구자: "논문 파라미터 그대로 config를 주고 `train_classifier` 실행하면 동일한 accuracy/혼동행렬/에너지분포 figure가 나온다."
- 연구자: "레이어 수/레이어 간격 sweep 돌려 SSIM vs layer count plot을 자동 생성한다."
- 엔지니어: "학습된 phase mask를 height map으로 변환하고, (옵션) Lumerical .fsp 모델을 생성한다."
- 리뷰어: "forward 모델/손실/검증 지표가 문서화되어 있고 코드에서 추적 가능하다."

## 5. 성공 기준 (Success Metrics)
- MNIST 5-layer phase-only:
  - accuracy >= 0.90 (목표 0.9175 ± small tolerance)
  - confusion matrix/energy distribution figure 생성
- Fashion-MNIST 5-layer:
  - accuracy >= 0.78 (목표 ~0.81 근처)
- Imaging lens:
  - SSIM이 free-space 대비 유의미하게 높음(예: 0.69 vs 0.34 수준의 방향성)
  - unit magnification 이미지가 시각적으로 복원됨
- 재현성:
  - 동일 config/seed로 실행 시 metrics.json의 hash가 동일
  - figure 픽셀 차이(또는 수치 차이)가 허용 오차 이하

## 6. 시스템 아키텍처 (High-level)
- physics/
  - grid 생성, ASM transfer function(H) 생성/캐싱, propagate()
- models/
  - DiffractionLayer(phase-only/complex), D2NNModel
- data/
  - MNIST/FashionMNIST/Image folder dataset, preprocessing(upsample/pad/encode)
- training/
  - loss, optimizer, loop, checkpoint, metrics logging
- viz/
  - figure generator(논문 그림 대응), 해석 가이드
- export/
  - phase->height map, (옵션) Lumerical builder

## 7. 품질/성능 요구
- 큰 N(200~300)에서 GPU 사용 가능해야 함
- ASM H는 (N,dx,λ,z) 조합별로 캐시하여 재계산 방지
- numpy와 PyTorch 혼용 시: gradient 끊김/성능 저하 방지 규칙 필요

## 8. 리스크 및 완화
- Detector layout(영역 위치/크기)이 accuracy에 민감할 수 있음
  -> detector layout을 파일로 외부화(layout JSON) + 실험별 명시
- ASM 구현에서 FFT shift/sign convention 오류
  -> propagation unit tests + known analytic diffraction case 비교
- ImageNet 데이터 접근/라이선스
  -> "이미지 폴더" 인터페이스 제공 + 샘플 데이터셋 대체 옵션 제공
```

---

## File: `docs/CODETREE.md`

```md
# Code Tree (Refactor Target)

아래 트리는 "논문 재현"과 "Codex 구현"에 최적화된 구조다.
pytorch 기준으로 작성하되, backend 교체가 가능하도록 분리한다.

repo/
├─ AGENT.md
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ mnist_phase_only_5l.yaml
│  ├─ fashion_phase_only_5l.yaml
│  ├─ imaging_lens_5l.yaml
│  ├─ sweeps/
│  │  ├─ imaging_layers_vs_ssim.yaml
│  │  └─ imaging_z_vs_ssim.yaml
├─ src/
│  └─ d2nn/
│     ├─ __init__.py
│     ├─ types.py                  # dataclass/typing: Shapes, Regions, Config
│     ├─ utils/
│     │  ├─ seed.py                # seed 고정, deterministic 옵션
│     │  ├─ io.py                  # npy/json/yaml, 결과 저장 구조
│     │  └─ math.py                # 안전한 complex ops, normalization
│     ├─ physics/
│     │  ├─ grid.py                # x,y, fx,fy grid 생성
│     │  ├─ asm.py                 # Angular Spectrum transfer function + propagate
│     │  ├─ materials.py           # n,k, absorption 모델, phase->height
│     │  └─ apertures.py           # padding/zero mask, input plane 정의
│     ├─ detectors/
│     │  ├─ layout.py              # detector region 정의(물리좌표/픽셀좌표)
│     │  ├─ integrate.py           # region energy 계산
│     │  └─ metrics.py             # accuracy/confusion/energy distribution
│     ├─ models/
│     │  ├─ layers.py              # DiffractionLayer, PropagationLayer
│     │  ├─ d2nn.py                # build_d2nn_model()
│     │  └─ constraints.py         # phase range sigmoid/clip 전략
│     ├─ data/
│     │  ├─ mnist.py
│     │  ├─ fashion_mnist.py
│     │  ├─ imagenet_folder.py
│     │  └─ preprocess.py          # upsample/pad/encode (amp/phase)
│     ├─ training/
│     │  ├─ losses.py              # classification + leakage, imaging MSE
│     │  ├─ loops.py               # train/eval 공통 루프
│     │  ├─ callbacks.py           # checkpoint, early stop, logging
│     │  └─ sweeps.py              # layer count / z sweep 실행
│     ├─ viz/
│     │  ├─ fields.py              # amp/phase/intensity plotting
│     │  ├─ classifier.py          # detector overlay, energy bars, confusion matrix
│     │  ├─ imaging.py             # input/output/free-space, SSIM, difference
│     │  ├─ propagation.py         # z-stack / layer-stack 시각화
│     │  └─ style.py               # colormap/range/labels 규칙
│     ├─ export/
│     │  ├─ heightmap.py           # phase->height map, 저장
│     │  └─ lumerical/
│     │     ├─ builder.py          # lumapi 기반 .fsp 생성
│     │     └─ merge.py            # layer fsp merge
│     └─ cli/
│        ├─ train_classifier.py
│        ├─ train_imager.py
│        ├─ eval.py
│        ├─ export_heightmap.py
│        └─ make_figures.py
├─ notebooks/
│  ├─ 00_sanity_asm.ipynb
│  ├─ 01_train_mnist.ipynb
│  ├─ 02_train_fashion.ipynb
│  ├─ 03_train_imaging.ipynb
│  └─ 90_debug_lumerical.ipynb
├─ tests/
│  ├─ test_asm_energy.py
│  ├─ test_phase_constraint.py
│  ├─ test_detector_integrals.py
│  └─ test_reproducibility_smoke.py
└─ docs/
   ├─ PRD.md
   ├─ CODETREE.md
   ├─ CORE_API.md
   ├─ VISUALIZATION.md
   ├─ COMMENTING_STYLE.md
   ├─ REPRODUCTION_PROTOCOL.md
   └─ LUMERICAL_INTEGRATION.md

핵심 원칙:
- 논문 수치 재현은 config로 고정(파라미터/레이아웃/손실/seed)
- figure는 viz 모듈로 "함수화"하여 언제나 동일한 결과를 생성
```

---

## File: `docs/CORE_API.md`

````md
# Core API Specification (D2NN)

아래 API는 Codex가 "함수 시그니처/입출력/단위"를 정확히 구현하도록 하는 계약서다.
모든 함수/클래스는 docstring에 (1) 단위, (2) shape, (3) 물리 의미를 반드시 포함한다.

---

## 1) physics.grid

### make_spatial_grid
```python
def make_spatial_grid(N: int, dx: float, centered: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        x, y: shape (N,), unit [m]
    """
````

### make_frequency_grid

```python
def make_frequency_grid(N: int, dx: float, fftshift: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        fx, fy: shape (N,), unit [cycles/m]
    """
```

---

## 2) physics.asm

### asm_transfer_function

```python
def asm_transfer_function(
    N: int,
    dx: float,
    wavelength: float,
    z: float,
    n: float = 1.0,
    bandlimit: bool = True,
    fftshifted: bool = False,
    dtype: str = "complex64",
) -> "TensorLike":
    """
    Angular Spectrum Method transfer function H(fx,fy).
    H = exp( i * 2π * z * sqrt((n/λ)^2 - fx^2 - fy^2) )
    - evanescent mode: sqrt(negative) => i*sqrt(|...|)
    """
```

### asm_propagate

```python
def asm_propagate(
    field: "TensorLike",
    H: "TensorLike",
    *,
    fftshifted: bool = False,
) -> "TensorLike":
    """
    field: complex tensor, shape (..., N, N)
    returns: propagated complex field, same shape
    """
```

권장 구현:

* H를 미리 계산/캐시하고 forward에서 재사용
* shift를 최소화(가능하면 unshift FFT 순서로 통일)

---

## 3) models.constraints

### PhaseConstraint (권장: sigmoid 기반)

```python
class PhaseConstraint:
    def __init__(self, max_phase: float):  # max_phase = π or 2π
        ...

    def __call__(self, raw_phase: "TensorLike") -> "TensorLike":
        """
        phi = max_phase * sigmoid(raw_phase)
        range: [0, max_phase]
        """
```

---

## 4) models.layers

### DiffractionLayer

```python
class DiffractionLayer(torch.layer):
    def __init__(
        self,
        N: int,
        dx: float,
        wavelength: float,
        z: float,
        *,
        phase_max: float = 2*np.pi,
        train_amplitude: bool = False,
        amplitude_range: tuple[float, float] = (0.0, 1.0),
        use_absorption: bool = False,
        absorption_alpha: float | None = None,  # optional per-layer attenuation
        name: str | None = None,
    ):
        """
        Forward:
            E_in (prev plane) -> propagate z -> multiply modulator -> E_out (this plane)
        """
```

### PropagationLayer (마지막 레이어→출력면)

```python
class PropagationLayer(torch.layer):
    def __init__(self, N: int, dx: float, wavelength: float, z: float, name: str | None = None):
        ...
```

---

## 5) detectors.layout

### DetectorRegion (dataclass)

```python
@dataclass(frozen=True)
class DetectorRegion:
    name: str
    center_xy: tuple[float, float]   # [m]
    size_xy: tuple[float, float]     # [m] (width, height)
```

### DetectorLayout

```python
@dataclass(frozen=True)
class DetectorLayout:
    regions: list[DetectorRegion]
    plane_size_xy: tuple[float, float]  # [m] typically (N*dx, N*dx)
```

### build_region_masks

```python
def build_region_masks(layout: DetectorLayout, N: int, dx: float) -> np.ndarray:
    """
    Returns:
        masks: shape (K, N, N), dtype=bool
    """
```

---

## 6) detectors.integrate

### integrate_regions

```python
def integrate_regions(intensity: "TensorLike", masks: "TensorLike", reduction: str = "sum") -> "TensorLike":
    """
    intensity: shape (B, N, N) or (N, N)
    masks: shape (K, N, N)
    returns:
        energies: shape (B, K) or (K,)
    """
```

---

## 7) training.losses

### classification_loss (CE + leakage)

```python
def classification_loss(
    energies: "TensorLike",          # (B,K)
    labels: "TensorLike",            # int (B,) or onehot (B,K)
    *,
    leakage_energy: "TensorLike" | None = None,  # (B,) optional
    leakage_weight: float = 0.1,
    temperature: float = 1.0,
) -> "TensorLike":
    """
    Option A (recommended):
        logits = energies / temperature
        L = CE(softmax(logits), labels) + leakage_weight * mean(leakage_ratio)
    """
```

### imaging_mse_loss

```python
def imaging_mse_loss(output_intensity: "TensorLike", target_intensity: "TensorLike") -> "TensorLike":
    """
    both shape: (B,N,N) scaled to [0,1]
    """
```

---

## 8) export.heightmap

### phase_to_height

```python
def phase_to_height(phase: np.ndarray, wavelength: float, delta_n: float) -> np.ndarray:
    """
    Δz = (λ / (2π)) * (Δφ / Δn)
    phase: [rad], range [0, max_phase]
    Returns: height map [m]
    """
```

---

## 9) viz (필수 figure API)

### plot_phase_mask

* 입력: phase[N,N] rad
* 출력: fig, ax, 저장 경로

### plot_output_with_detectors

* intensity + detector overlay

### plot_confusion_matrix

* normalize 옵션

### plot_energy_distribution_heatmap

* matrix (KxK) 또는 (samplesxK)

### plot_propagation_stack

* layer별 amplitude/phase 스택 또는 z-scan cross-section

각 plotting 함수는 반드시 다음을 지원:

* `save_path: Path | None`
* `show: bool`
* `title: str`
* `physical_units`: axis label(m/mm), colorbar label 포함

````

---

## File: `docs/VISUALIZATION.md`
```md
# Visualization Design & Interpretation Guide (D2NN)

이 문서는 "어떻게 그릴 것인가(구현)" + "어떻게 읽을 것인가(해석)"를 같이 정의한다.
목표: 논문 figure와 1:1로 대응되는 시각화 산출물을 자동 생성.

---

## 0. 공통 규칙 (모든 figure에 적용)

### 0.1 좌표/단위 표기
- 기본: x,y는 [mm]로 표시(사람이 보기 쉬움)
- 내부 계산은 [m]로 통일
- x축/ y축 눈금은 `extent=[-L/2, L/2, -L/2, L/2]` 형태로 실제 길이를 반영

### 0.2 intensity 스케일
- intensity = |E|^2
- classifier 출력 분포는 dynamic range가 크므로:
  - 기본은 `log10(intensity + eps)` 또는
  - `Normalize(vmin, vmax)`로 clip
- 동일 실험에서 figure 비교를 위해 vmin/vmax는 고정 권장

### 0.3 phase 시각화
- phase = angle(E) 는 [-π, π] 래핑됨
- phase mask(학습 변수)는 [0, π] 또는 [0, 2π]로 정의되므로:
  - mask는 반드시 [0,max] 범위로 표시
  - colormap은 cyclic(예: hsv/twilight) 사용 권장
- phase unwrapping은 기본적으로 하지 않는다(간섭 패턴에서 wrap 자체가 정보)

### 0.4 Reproducibility
- figure 생성 함수는 입력 배열이 같으면 항상 같은 png를 만든다
- matplotlib 스타일/폰트 고정(가능하면 repo에서 style.py로 통제)

---

## 1) Classifier Figures

### Fig-A: Trained phase masks per layer
**무엇을 그리나**
- layer l의 phase mask φ_l(x,y) 이미지 (N×N)
- 5 layers면 5장의 패널(또는 grid)

**왜 중요한가(해석)**
- 학습이 진행되면 phase mask가 무작위 잡음이 아니라
  “에너지 라우팅(constructive interference)”를 위한 구조를 형성한다.
- layer 간 패턴이 서로 다른 이유는 각 레이어가 서로 다른 거리 z에서의
  diffraction coupling을 조절하기 때문.

**구현 체크리스트**
- 동일한 색상범위: [0, π] 또는 [0,2π]
- colorbar에 rad 표기
- 레이어 index(L1..L5) 표기

---

### Fig-B: Output intensity + detector overlay
**무엇을 그리나**
- 샘플 입력 1개에 대한 출력면 intensity I(x,y)
- detector region 사각형(또는 원) overlay

**해석**
- 올바른 분류라면 target detector region에 peak가 생긴다.
- 잘못된 분류는 energy가 다른 detector region으로 분산되거나
  detector 밖(“leakage”)으로 새는 패턴을 보인다.

**구현**
- `plot_output_with_detectors(intensity, layout)`
- detector index(0..9)를 overlay로 같이 찍기

---

### Fig-C: Confusion matrix (test set)
**무엇을**
- y_true vs y_pred matrix (K×K)

**해석**
- 대각선이 진할수록 정확도 높음
- 특정 클래스 간 혼동(예: 4 vs 9)은 detector layout/입력 전처리/네트워크 용량 문제의 신호

**구현 포인트**
- raw count + normalized(%) 둘 다 저장
- `plot_confusion_matrix(cm, normalize=True/False)`

---

### Fig-D: Energy distribution heatmap
**정의**
- 입력 클래스 c에 대해 detector energy vector P_k 를 집계
- heatmap: row=input class, col=detector index (또는 samples×K)

**해석**
- 이상적이면 row마다 1개의 col에 에너지가 집중
- leakage penalty를 넣으면 detector 밖 에너지가 줄어드는 경향

**구현**
- `energies = integrate_regions(I, masks)` (sum 또는 mean)
- `energies = energies / (energies.sum(axis=-1, keepdims=True)+eps)`로 정규화 후 heatmap

---

## 2) Imaging Lens Figures

### Fig-E: Input vs D2NN output vs Free-space
**무엇을**
- input amplitude(or intensity) target
- D2NN output intensity
- 동일 거리의 free-space propagation intensity

**해석**
- D2NN이 학습한 것은 "단순한 회절 패턴"이 아니라
  입력의 amplitude 정보를 다시 공간적으로 재배치해 영상(autocoding)으로 만든다.
- free-space는 speckle/blur가 심해 SSIM이 낮다.

**구현**
- `simulate_free_space(input_field, z_total)`
- `simulate_d2nn(input_field)`
- SSIM 계산을 figure에 annotate

---

### Fig-F: Propagation inside D2NN (layer stack / z-stack)
보충자료 Fig S6 유형을 재현하는 핵심 figure.

**무엇을**
- (A) 레이어 평면들에서 amplitude/phase를 순서대로 배치
- (C) 특정 x-slice에 대해 z에 따른 cross-sectional amplitude/phase map

**해석**
- D2NN은 중간 레이어에서 복잡한 간섭 패턴을 만들고,
  마지막에 target 이미지를 형성하도록 위상을 조정한다.
- free-space는 에너지가 퍼지며 target과 다른 분포가 된다.

**구현 전략**
- forward 중간 출력 `E_l` 를 모두 저장하는 옵션 제공:
  - `return_intermediates=True`
- `plot_layer_stack([E0,E1,...,E_out])`
- cross-section은:
  - x index 하나 고정 → z별로 y 또는 x line을 stack
  - 또는 z를 세분(예: layer 사이 20 steps)해서 propagate 반복

---

## 3) Architecture Sweep Figures (SSIM vs layers / z)

**무엇을**
- layer count sweep: N_layers=1..10에서 SSIM 평균
- z sweep: layer spacing을 바꾸며 SSIM 평균

**해석**
- 너무 적은 레이어는 DoF 부족으로 SSIM 낮음
- z가 너무 작거나 크면 receptive field/aliasing 때문에 성능 저하 가능
- 논문에서는 5 layers, z=4mm가 최적 근처로 보고됨

**구현**
- sweep runner:
  - 고정된 작은 validation set(예: 100 images)
  - 각 조건에서 짧은 epoch로 pretrain 후 SSIM 측정(또는 충분히 학습)
- 결과는 반드시 csv/json으로 저장하고 plot은 재생성 가능해야 함

---

## 4) 흔한 실수/디버깅 (Plot이 이상할 때)
1) FFT shift 순서 오류 → output이 좌우/상하 뒤집힘
2) dx/λ/z 단위 mismatch → 전혀 다른 diffraction scale
3) padding 미흡 → wrap-around artifact(가장자리 링/복제)
4) phase range constraint가 잘못됨(clip vs sigmoid) → 학습 불안정/모드 collapse
````

---

## File: `docs/COMMENTING_STYLE.md`

````md
# Commenting & Docstring Style Guide (Physics + ML)

목표: "코드만 봐도 물리 모델/단위/shape/근거(논문 eq)"가 추적 가능해야 한다.
특히 D2NN은 단위/FFT convention 오류가 치명적이므로 문서화를 강제한다.

---

## 1) 모든 함수/클래스 docstring에 반드시 포함할 것

### (A) Units
- 모든 물리 파라미터(λ, dx, z, n, Δn, alpha 등)는 단위를 명시
- 예: `dx: float [m]`

### (B) Shapes
- 텐서/배열의 shape을 명시
- 예: `field: complex tensor, shape (B, N, N)`

### (C) Physical meaning / equation reference
- 가능하면 논문/보충자료 식 번호를 주석으로 연결
- 예: "ASM propagation (Goodman Fourier optics), D2NN forward model eq.(3)-(5) conceptually"

### (D) Determinism note
- random init/seed 관련 동작을 하는 곳이면 seed/PRNG 의존성을 명시

---

## 2) Inline comment는 "왜(why)"만 남기기
Bad:
```python
E = torch.fft.fft2d(E)  # do fft
````

Good:

```python
# FFT-based propagation assumes periodic boundary; padding is handled upstream to reduce wrap-around artifacts.
E_f = torch.fft.fft2d(E)
```

---

## 3) Phase/angle 관련 주석 규칙

* angle()은 [-π,π]로 wrap됨을 주석으로 명시
* 학습 변수 φ는 [0,π] 또는 [0,2π]로 정의됨을 명시
* height map 변환식(Δz = λ/(2π) Δφ/Δn)은 반드시 export 모듈에 주석으로 넣기

---

## 4) Naming conventions

* `N`: grid size (pixels)
* `dx`: pixel pitch [m]
* `L`: aperture size [m] = N*dx
* `wavelength`: λ [m]
* `z`: propagation distance [m]
* `E`: complex field
* `I`: intensity = |E|^2
* `phi`: phase mask [rad]
* `H`: ASM transfer function in Fourier domain

---

## 5) Example Docstring (권장 템플릿)

```python
def asm_transfer_function(...):
    """
    Build Angular Spectrum transfer function.

    Physics:
        H(fx,fy) = exp(i*2π*z*sqrt((n/λ)^2 - fx^2 - fy^2))
        Evanescent region uses imaginary kz.

    Args:
        N: int, grid size [pixels]
        dx: float, pixel pitch [m]
        wavelength: float, vacuum wavelength [m]
        z: float, propagation distance [m]
        n: float, refractive index of propagation medium
        bandlimit: bool, apply band-limited ASM to reduce aliasing

    Returns:
        H: complex tensor, shape (N,N), dtype complex64/complex128

    Notes:
        - Must match FFT ordering convention used in asm_propagate().
        - Cached by (N,dx,wavelength,z,n,bandlimit).
    """
```

````

---

## File: `docs/REPRODUCTION_PROTOCOL.md`
```md
# Reproduction Protocol (Simulation)

이 문서는 "논문 결과 재현을 위한 실행 레시피"다.
모든 값은 config로 외부화하고, 이 문서에는 '기본값'과 '기대 결과'를 기록한다.

---

## 0) 공통 준비

### 0.1 설치/환경
- Python >= 3.10 권장(논문은 3.5였으나 재현 목적이면 최신 OK)
- pytorch + numpy + matplotlib + (옵션) scikit-image(SSIM)
- GPU 권장: N=200~300에서 batch 학습은 CPU로도 가능하지만 느릴 수 있음

### 0.2 재현성 설정
- seed: config에 `seed: 1234` 고정
- deterministic 옵션:
  - torch: `torch_DETERMINISTIC_OPS=1` 등 환경변수 지원(가능한 범위 내)

### 0.3 결과 폴더 규칙
runs/{exp_name}/{timestamp_or_hash}/
- config_resolved.yaml
- checkpoints/
- metrics.json
- figures/
- exports/ (height maps 등)

---

## 1) MNIST Digit Classifier (Phase-only, 5 layers)

### 1.1 기본 물리 파라미터(논문 기반)
- wavelength λ = 0.75e-3 m (0.4 THz in air)
- N = 200
- dx = 0.4e-3 m  (layer size 8 cm)
- num_layers = 5
- z_layer = 3e-2 m
- z_out = 3e-2 m  (논문/보충자료를 기준으로 config에서 조정 가능)

### 1.2 입력 전처리 (Amplitude encoding)
- MNIST 28x28 grayscale
- (권장) binarize (실험에서는 알루미늄 foil로 0 transmission 구현)
- upsample -> target object size(예: 80x80) (nearest)
- pad with zeros to N=200 (8cm aperture)

field 정의:
- plane wave illumination amplitude=1
- object transmission amplitude T(x,y) ∈ [0,1]
- input field: E0 = T (complex64, zero phase)

### 1.3 Detector layout
- detector regions(10개)을 **물리 좌표**로 정의
- region 크기/간격은 config로 고정
- leakage penalty 계산을 위해 "detector union mask"도 유지

### 1.4 Loss (논문 서술에 맞춘 권장형)
- energies: P_k = sum_{region k} I
- logits = P_k / temperature
- CE loss + leakage penalty:
  - leakage = (sum outside detector union) / (total energy)
  - L = CE(softmax(logits), y) + w_leak * mean(leakage)

### 1.5 학습
- optimizer: Adam
- epochs: 10 (논문에서 digit classifier는 ~10 epochs)
- batch size: 8
- phase constraint: φ ∈ [0,π] 또는 [0,2π] (논문은 classifier에 0~π 제한 언급)

### 1.6 기대 결과(Acceptance)
- test accuracy >= 0.90
- 목표는 0.9175 근처(논문 수치)
- figures 생성:
  - phase masks (L1..L5)
  - training curve(loss/acc)
  - confusion matrix
  - sample output intensity + detector overlay
  - energy distribution heatmap

---

## 2) Fashion-MNIST Classifier (Phase encoding)

차이점:
- 입력은 amplitude가 아니라 phase channel로 encoding
- grayscale g∈[0,1] → φ_in = 2π*g
- input field: E0 = exp(i*φ_in) * aperture_mask

기대:
- 5-layer phase-only에서 ~0.8대 accuracy 방향성
- complex modulation을 켜면 더 좋아지는 경향(옵션)

---

## 3) Imaging Lens D2NN (Amplitude imaging)

### 3.1 파라미터(논문 기반)
- N = 300
- dx = 0.3e-3 m (9 cm aperture)
- num_layers = 5
- z_layer = 4e-3 m
- z_out = 7e-3 m
- dataset: ImageNet grayscale subset (논문은 2000장 랜덤 subset)

### 3.2 데이터
- 법적/실무상 ImageNet이 없으면:
  - 사용자 제공 폴더(images/)를 읽도록 구현
  - 동일 인터페이스로 대체 데이터(OpenImages subset 등)도 가능

### 3.3 Loss
- output intensity I_out
- target intensity I_target (입력 amplitude 또는 amplitude^2에 맞춰 통일)
- MSE(I_out, I_target)
- metric: SSIM(I_out, I_target)

### 3.4 기대 결과
- free-space 대비 SSIM 향상
- input vs output 비교 figure 자동 생성
- propagation inside D2NN figure (layer stack, cross section)

---

## 4) Error source simulation (Optional)
보충자료에 언급된 요소들을 toggle로 켠다:
- layer misalignment: each layer random shift <= 0.1 mm
- absorption: per-layer attenuation exp(-alpha * thickness)
- Poisson recon error: height map smoothing/quantization

---

## 5) Export: phase -> height map
- Δz = (λ/(2π)) * (φ / Δn)
- Δn = n_material - n_air
- material 예시: VeroBlackPlus RGD875 (n=1.7227 @0.4THz)
- 결과는 exports/height_map.npy 로 저장
````

---

## File: `docs/LUMERICAL_INTEGRATION.md`

````md
# Lumerical (lumapi) Integration Plan

목표: 학습된 height map(Δz)을 Lumerical FDTD 모델로 자동 변환해
(옵션으로) full-wave 시뮬레이션 검증을 가능하게 한다.

---

## 1) 현재 코드(레거시) 특징
- height_map.npy / filter_height_map.npy 로딩
- structure group + rect를 (i,j) loop로 생성
- layer마다 수만 개 rect(200x200=40k) → 매우 느림
- 병렬 프로세스로 layer별 .fsp 생성 후 merge

---

## 2) 리팩토링 목표
- Windows 경로 하드코딩 제거(lumapi path는 env/config)
- 전역변수 제거 → `LumericalBuilder(config)` 형태
- (가능하면) rect per pixel 대신:
  - height map 기반 surface import(가능한 API가 있다면) 또는
  - 더 coarse한 구조로 근사(학습 grid와 FDTD mesh의 trade-off)
- 실행/에러 핸들링/로그 표준화

---

## 3) API 스펙

### LumericalConfig (dataclass)
- size(N), dx, z_layer, z_out
- wavelength
- material name, refractive index, (optional) extinction k
- mesh override(dx/dy/dz)
- file paths: base_fsp, out_fsp, temp_dir
- hide GUI

### LumericalBuilder
```python
class LumericalBuilder:
    def __init__(self, cfg: LumericalConfig):
        ...

    def build_base_simulation(self) -> Path:
        """
        Create source, monitors, simulation region, optional filter.
        Save base .fsp.
        """

    def build_layer(self, layer_index: int, height_map: np.ndarray) -> Path:
        """
        Create a single layer group based on height map.
        Save temp_layer_{i}.fsp.
        """

    def merge_layers(self, base_fsp: Path, layer_fsps: list[Path]) -> Path:
        """
        Paste each layer group into base simulation.
        Save final .fsp.
        """
````

---

## 4) 주의 사항/해석

* D2NN 학습은 scalar ASM 기반이므로,
  FDTD는 boundary/mesh/손실/다중반사까지 포함해 결과가 달라질 수 있음.
* 논문에서도 다중반사는 약하다고 보고되어 기본 모델에서는 무시.
* FDTD 검증은 "정성적 일치(energy focusing/이미징 경향)"를 우선 목표로 둔다.

---

## 5) 최소 기능(DoD)

* height_map.npy를 읽어 1-layer/5-layer .fsp 생성 가능
* detector plane monitor에서 intensity map export 가능
* 생성된 .fsp 파일이 GUI에서 열리고 구조가 올바르게 배치됨

````

---

## File: `AGENT.md`
```md
# D2NN Domain Agent Instructions (for Codex)

You are an expert domain agent tasked with implementing a reproducible D2NN simulation/training codebase.
Your primary goal is to reproduce the key numerical results and figures of the 2018 Science D2NN paper
using a modular, testable, and deterministic implementation.

---

## 0) Non-negotiables
1) Units: ALL internal physics must use SI units (meters, radians).
2) Shapes: Every tensor op must explicitly document expected shapes.
3) Determinism: Given the same config + seed, outputs must be identical (within deterministic limits).
4) Separation of concerns:
   - physics (ASM) is independent from ML training loop
   - visualization is independent from training loop
   - detector layout is externalized (config/layout file)
5) Figures must be reproducible artifacts generated by code (no manual plotting in notebooks).

---

## 1) Physics model you must implement
- Complex scalar field E(x,y) per plane.
- Each diffractive layer l is a thin modulator:
    mod_l(x,y) = A_l(x,y) * exp(i * phi_l(x,y))
  where:
    - phase-only: A_l = 1
    - complex modulation (optional): A_l is trainable in [0,1] via sigmoid
- Forward propagation between planes uses Angular Spectrum Method:
    E_{l,plane} = ASM(E_{l-1,after_mask}, z_l)
    E_{l,after_mask} = E_{l,plane} * exp(i*phi_l)
- Output plane intensity:
    I = |E_out|^2

Important: FFT ordering conventions must be consistent across transfer function H and asm_propagate().
Write unit tests to detect sign/shift mistakes.

---

## 2) Phase constraints (match paper intent)
Implement sigmoid-based constraints:
- classifier networks: phi_max = π  (or expose config to use 2π)
- imaging networks: phi_max = 2π
Use: phi = phi_max * sigmoid(raw_phi)

Do NOT silently clip without documenting it. If you implement clip, make it an explicit option.

---

## 3) Classification head
- Define K detector regions on output plane.
- Compute energies P_k by integrating intensity over each region.
- Prediction: argmax_k P_k

Loss (recommended):
- cross entropy over detector energies + leakage penalty:
    leakage = energy outside union(detectors) / total energy
    L = CE(softmax(P/temperature), y) + w_leak * mean(leakage)

Detector layout must be specified in physical coordinates and converted to pixel masks.

---

## 4) Imaging head
- Output intensity should match target intensity (unit magnification).
- Loss: MSE(output_intensity, target_intensity)
- Metric: SSIM (optional dependency) + MSE

---

## 5) Performance rules
- Cache ASM transfer functions H per unique (N,dx,λ,z,n,bandlimit) key.
- Avoid recomputing frequency grids inside forward() every call.
- Avoid mixing numpy ops inside differentiable forward pass, except for precomputed constants.
- Use complex64 for speed; allow complex128 for debugging.

---

## 6) Testing requirements (minimum)
Create tests that run fast on CPU with small N (e.g., N=32/64):
1) ASM identity test: z=0 -> output == input
2) Energy sanity: for phase-only and no cropping, total energy should be approximately conserved
3) Detector integration: masks sum and indexing correctness
4) Deterministic smoke: same seed/config -> identical metrics.json and identical first-batch outputs
5) Phase constraint range test: min/max phi within [0,phi_max]

---

## 7) Visualization requirements
Implement figure functions that:
- accept arrays + metadata
- produce deterministic png files
- include axis units (mm), colorbars, and consistent scaling
Required figures:
- Layer phase masks (L1..Ln)
- Training curves (loss/accuracy)
- Output intensity + detector overlay
- Confusion matrix
- Energy distribution heatmap
- Imaging comparison: input vs D2NN vs free-space (+SSIM)
- Propagation inside network: layer stack and/or x-z cross-section

Also write a short interpretation note in docs/VISUALIZATION.md.

---

## 8) Implementation checklist (Definition of Done)
A module is "done" only when:
- it has type hints + docstrings (units/shapes)
- it has at least one unit test
- it is referenced by at least one CLI command
- it has a minimal example config that runs end-to-end and generates figures

---

## 9) Deliverables
- The repo tree defined in docs/CODETREE.md
- CLI commands:
  - train_classifier, train_imager, eval, export_heightmap, make_figures
- configs that reproduce MNIST/Fashion/Imaging experiments
- docs as specified
````

---

## 다른 관점/대안 (Alternative solutions you might consider)

1. **Backend를 PyTorch로 전환**

   * 장점: complex autograd/성능/생태계(예: torchmetrics, lightning)가 편함
   * 단점: 현재 노트북이 torch 기반이라 migration 비용이 있음
   * 타협: `physics/asm.py`는 numpy/torch 3가지 backend를 지원하는 “thin adapter”로 설계

2. **ASM 대신 Fresnel/RS(레이리-좀머펠트) 근사**

   * 논문은 ASM을 사용했다고 명시(보충자료)하지만, 특정 거리/조건에서는 Fresnel이 더 빠를 수 있음. 
   * 다만 “논문 재현”이 목적이면 ASM을 기본으로 두고 Fresnel은 옵션으로.

3. **Detector layout을 학습/최적화 대상으로 포함**

   * detector region 위치/크기 자체를 최적화(또는 여러 레이아웃 비교)하면 성능이 흔들리는 요인을 줄일 수 있음
   * 하지만 논문 재현에서는 레이아웃을 고정해야 하므로, 우선은 “외부 파일로 고정”이 맞음.

---

## 바로 실행 가능한 액션 플랜 (Practical plan)

1. **repo skeleton 생성**: `src/d2nn/...` 및 `configs/` `docs/` `tests/` 폴더부터 만들기
2. **physics.asm 먼저 구현 + 테스트 2개**: z=0 identity / 간단 aperture diffraction sanity
3. **D2NN 모델(5-layer) + detector integration 구현**: 작은 N=64로 smoke test
4. **MNIST 전처리(amp encoding) + loss(CE+leakage) 구현** → accuracy가 올라가는지 확인
5. **viz 모듈로 “논문형 figure”를 먼저 고정**: 결과가 조금 달라도 figure가 항상 생성되게
6. **height map export 구현**: Δz 식/Δn 포함
7. (옵션) **Lumerical builder 리팩토링**: 최소 기능(1-layer)부터

원하면, 다음 단계로는 **configs 예시(YAML)까지 완성된 형태**(detector layout의 실제 좌표 포함)로 더 “바로 구현 가능한 수준”으로 내려서 작성해줄게요.
