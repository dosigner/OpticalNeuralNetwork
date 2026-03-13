**Supplementary Methods(NA, f, neuron size, layer count, spacing, 학습 설정 등)**와 **Main paper Fig.4 도식(특히 MNIST에서 linear Fourier는 f=1mm, nonlinear Fourier는 f=4mm, hybrid는 f=1mm×N)**을 그대로 반영했어.  

---

## 내가 이렇게 만든 방법(요청하신 step-by-step)

1. **Supplementary Methods에서 “고정 사양”을 config로 내림**

   * Saliency(Cell): NA=0.112, f1=f2=10mm, neuron=2µm, 800×800, spacing=100µm, λ=532nm, epochs=100, bs=10, lr=0.01, phase range 0~2π
   * Saliency(CIFAR): f=2mm, 160×160, neuron=2µm, resize 100×100 + pad 160×160
   * MNIST classification: NA=0.16, neuron=1µm, 200×200, upsample×3 + pad, epochs=30, phase range 0~π, detector width=12µm


2. **Main paper Fig.4 도식에서 ‘구성별 focal length/구조’까지 config로 분해**

   * Linear Fourier (5L): 2f f1=f2=1mm
   * Nonlinear Fourier (5L): 2f f1=f2=4mm + single SBN rear
   * Real-space(5L/10L): layer spacing 3mm
   * Hybrid: 2f system ×6 (5L), ×11 (10L), f=1mm


3. **Fig.2/3/4 이미지를 직접 보고**(패널 개수/행/열/컬러바 위치)
   “**픽셀 고정 bounding-box 레이아웃 엔진**”으로 저장하면 항상 동일한 subplot 배치가 나오도록 스펙을 만들었어. 

---

# 1) `docs/CONFIG_TEMPLATES.md` (실제 YAML 템플릿 포함)

````markdown
# Config Templates (paper-parameter filled)

아래 YAML은 `src/fd2nn/config/` 아래에 그대로 저장하는 것을 전제로 한다.
- 모든 길이 단위는 meters(m)
- phase는 radians
- dataset 경로(root)는 로컬 환경에 맞게 수정

근거:
- Supplementary Methods: NA, f, neuron size, grid size, layer spacing, training hyperparams, phase range, detector width :contentReference[oaicite:5]{index=5}
- Main paper Fig.4 schematic: MNIST 구성별 f=1mm vs 4mm 및 hybrid 2f count :contentReference[oaicite:6]{index=6}

---

## File: src/fd2nn/config/saliency_cell.yaml  (Main Fig.2: unit magnification)

```yaml
experiment:
  name: "fig2_saliency_cell_unitmag"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7      # 532 nm
  n: 1.0
  grid:
    nx: 800
    ny: 800
    dx_m: 2.0e-6             # neuron size 2 µm
    dy_m: 2.0e-6
  dual_2f:
    enabled: true
    f1_m: 1.0e-2             # 10 mm
    f2_m: 1.0e-2             # 10 mm
    na1: 0.112
    na2: 0.112
    apply_scaling: false     # intensity-normalized training에서는 보통 불필요
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4  # 100 µm
    bandlimit: true
    evanescent: "mask"       # mask|decay

model:
  type: "fd2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 6.283185307179586  # 2π (saliency)
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "rear"                  # single SBN at end (paper default for F-D2NN)
    phi_max_rad: 3.141592653589793    # π
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "saliency"
  gamma_flip: true

data:
  dataset: "cell_gdc"
  root: "data/cell_gdc/"
  cell_types:
    train: ["type1"]
    val: ["type1"]
    test: ["type1", "type2", "type3"]
  split:
    train: 2750
    val: 250
    test_type1: 500
    test_type2: 250
    test_type3: 250
  patch:
    raw_patch_px: [200, 200]
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 2               # 200->400
    resize_mode: "bilinear"
    pad_to: [800, 800]               # 400->800 (boundary padding)
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 100
  loss: "mse"

eval:
  pr_thresholds: 256
  f_beta2: 0.3                       # saliency 관례(논문은 F-measure만 표기)
  report_f1_also: true

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  layout_id: "fig2"
  intensity_viz_norm: "percentile"
  intensity_percentile: [1, 99]
  phase_cmap: "twilight"
  scalebar:
    enabled: true
    row_um: {0: 200, 1: 200, 2: 200, 3: 400}
  fig2_samples:
    # Fig.2 패널 구성(첫 3행)은 예시 5개: type1 1개, type2 2개, type3 2개
    type1_indices: [0]
    type2_indices: [0, 1]
    type3_indices: [0, 1]
````

---

## File: src/fd2nn/config/saliency_cell_mag2x.yaml  (Supp Fig. S1: 2× magnification)

```yaml
experiment:
  name: "supp_s1_saliency_cell_mag2x"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 800
    ny: 800
    dx_m: 2.0e-6
    dy_m: 2.0e-6
  dual_2f:
    enabled: true
    f1_m: 1.0e-2             # 10 mm
    f2_m: 2.0e-2             # 20 mm (2× magnification)
    na1: 0.112
    na2: 0.112
    apply_scaling: false
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4
    bandlimit: true
    evanescent: "mask"

model:
  type: "fd2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 6.283185307179586
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "rear"
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "saliency"
  gamma_flip: true

data:
  dataset: "cell_gdc"
  root: "data/cell_gdc/"
  cell_types:
    train: ["type1"]
    val: ["type1"]
    test: ["type1", "type2", "type3"]
  split:
    train: 2750
    val: 250
    test_type1: 500
    test_type2: 250
    test_type3: 250
  patch:
    raw_patch_px: [200, 200]
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 2
    resize_mode: "bilinear"
    pad_to: [800, 800]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 100
  loss: "mse"

eval:
  pr_thresholds: 256
  f_beta2: 0.3
  report_f1_also: true

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  layout_id: "fig2"
  intensity_viz_norm: "percentile"
  intensity_percentile: [1, 99]
  phase_cmap: "twilight"
  scalebar:
    enabled: true
    row_um: {0: 200, 1: 200, 2: 200, 3: 400}
  fig2_samples:
    type1_indices: [0]
    type2_indices: [0, 1]
    type3_indices: [0, 1]
```

---

## File: src/fd2nn/config/saliency_cifar_video.yaml  (Main Fig.3)

```yaml
experiment:
  name: "fig3_saliency_cifar_train_video_test"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 160
    ny: 160
    dx_m: 2.0e-6             # neuron size 2 µm
    dy_m: 2.0e-6
  dual_2f:
    enabled: true
    f1_m: 2.0e-3             # 2 mm
    f2_m: 2.0e-3             # 2 mm
    na1: 0.112               # "other settings the same" 가정
    na2: 0.112
    apply_scaling: false
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4
    bandlimit: true
    evanescent: "mask"

model:
  type: "fd2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 6.283185307179586  # 2π
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "rear"
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "saliency"
  gamma_flip: true

data:
  dataset: "cifar10"
  root: "data/cifar10/"
  train_class: "cat"
  test_sets:
    - {type: "cifar10", class: "horse"}
    - {type: "video_frames", root: "data/online_horse_video_frames/"}  # 사용자 환경에 맞게 준비
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    resize_to: [100, 100]
    resize_mode: "bilinear"
    pad_to: [160, 160]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 100
  loss: "mse"

eval:
  pr_thresholds: 256
  f_beta2: 0.3
  report_f1_also: true

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  layout_id: "fig3"
  intensity_viz_norm: "percentile"
  intensity_percentile: [1, 99]
  phase_cmap: "twilight"
  scalebar:
    enabled: true
    row_um: {0: 80, 1: 80, 2: 80, 3: 80}     # Fig.3 caption: 80 µm
  fig3_frames:
    # Fig.3는 5프레임 예시
    frame_indices: [0, 60, 120, 180, 239]
```

---

# MNIST classification — Fig.4 configs

공통 근거:

* NA=0.16, neuron=1µm, grid=200×200, MNIST upsample×3+pad, epochs=30, lr=0.01, detector width=12µm, phase range 0~π 
* Fig.4 schematic: Linear Fourier f=1mm, Nonlinear Fourier f=4mm, Hybrid f=1mm 및 2f count(x6, x11) 

## File: src/fd2nn/config/cls_mnist_linear_real_5l.yaml  (Fig.4a: Linear Real 5L, 3mm spacing)

```yaml
experiment:
  name: "fig4a_cls_mnist_linear_real_5l"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  propagation:
    method: "ASM"
    layer_spacing_m: 3.0e-3      # 3 mm (Fig.4 diagram)
    bandlimit: true
    evanescent: "mask"
  aperture:
    enabled: false               # 필요 시 true로 두고 NA 마스크 적용 가능
    na: 0.16

model:
  type: "real_d2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793    # π
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    enabled: false

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"          # 구현에서 고정(예: 2×5 grid)
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3           # 28->84
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  layout_id: null
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_nonlinear_real_5l.yaml  (Fig.4a: Nonlinear Real 5L, multi-SBN)

```yaml
experiment:
  name: "fig4a_cls_mnist_nonlinear_real_5l"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  propagation:
    method: "ASM"
    layer_spacing_m: 3.0e-3
    bandlimit: true
    evanescent: "mask"

model:
  type: "real_d2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "per_layer"            # multi-SBN (each layer)
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_linear_fourier_5l_f1mm.yaml  (Fig.4a: Linear Fourier 5L, f=1mm)

```yaml
experiment:
  name: "fig4a_cls_mnist_linear_fourier_5l_f1mm"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  dual_2f:
    enabled: true
    f1_m: 1.0e-3          # 1 mm (Fig.4 schematic)
    f2_m: 1.0e-3
    na1: 0.16
    na2: 0.16
    apply_scaling: false
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4      # 100 µm (compact)
    bandlimit: true
    evanescent: "mask"

model:
  type: "fd2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793    # π (classification)
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    enabled: false

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_nonlinear_fourier_5l_f4mm.yaml  (Fig.4a: Nonlinear Fourier 5L, f=4mm + single SBN rear)

```yaml
experiment:
  name: "fig4a_cls_mnist_nonlinear_fourier_5l_f4mm"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  dual_2f:
    enabled: true
    f1_m: 4.0e-3          # 4 mm (Fig.4 schematic)
    f2_m: 4.0e-3
    na1: 0.16
    na2: 0.16
    apply_scaling: false
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4
    bandlimit: true
    evanescent: "mask"

model:
  type: "fd2nn"
  num_layers: 5
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "rear"                 # single SBN at end
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_linear_real_10l.yaml  (Fig.4b: Linear Real 10L)

```yaml
experiment:
  name: "fig4b_cls_mnist_linear_real_10l"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  propagation:
    method: "ASM"
    layer_spacing_m: 3.0e-3
    bandlimit: true
    evanescent: "mask"

model:
  type: "real_d2nn"
  num_layers: 10
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    enabled: false

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_nonlinear_real_10l.yaml  (Fig.4b: Nonlinear Real 10L)

```yaml
experiment:
  name: "fig4b_cls_mnist_nonlinear_real_10l"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  propagation:
    method: "ASM"
    layer_spacing_m: 3.0e-3
    bandlimit: true
    evanescent: "mask"

model:
  type: "real_d2nn"
  num_layers: 10
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "per_layer"
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_hybrid_5l.yaml  (Fig.4b: Hybrid Nonlinear 5L, 2f×6, f=1mm)

```yaml
experiment:
  name: "fig4b_cls_mnist_hybrid_5l"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  # hybrid는 내부에서 여러 번 2f를 사용(도식: x6, f=1mm)
  hybrid_2f:
    enabled: true
    f_m: 1.0e-3
    na: 0.16
    num_2f_systems: 6
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4
    bandlimit: true
    evanescent: "mask"

model:
  type: "hybrid_d2nn"
  num_layers: 5
  hybrid:
    plane_sequence: ["fourier", "real", "fourier", "real", "fourier"]  # 구현에서 그대로 사용
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "per_layer"   # hybrid는 각 stage 뒤 activation(구현 단순화)
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

## File: src/fd2nn/config/cls_mnist_hybrid_10l.yaml  (Fig.4b: Hybrid Nonlinear 10L, 2f×11, f=1mm)

```yaml
experiment:
  name: "fig4b_cls_mnist_hybrid_10l"
  seed: 42
  device: "cuda"
  dtype: "float32"
  deterministic: true
  save_dir: "runs"

optics:
  wavelength_m: 5.32e-7
  n: 1.0
  grid:
    nx: 200
    ny: 200
    dx_m: 1.0e-6
    dy_m: 1.0e-6
  hybrid_2f:
    enabled: true
    f_m: 1.0e-3
    na: 0.16
    num_2f_systems: 11
  propagation:
    method: "ASM"
    layer_spacing_m: 1.0e-4
    bandlimit: true
    evanescent: "mask"

model:
  type: "hybrid_d2nn"
  num_layers: 10
  hybrid:
    plane_sequence: ["fourier","real","fourier","real","fourier","real","fourier","real","fourier","real"]
  modulation:
    kind: "phase_only"
    phase_constraint: "sigmoid"
    phase_max_rad: 3.141592653589793
    init: "uniform"
    init_scale: 0.1
  nonlinearity:
    type: "sbn60"
    enabled: true
    position: "per_layer"
    phi_max_rad: 3.141592653589793
    intensity_norm: "per_sample_minmax"
    eps: 1.0e-12

task:
  name: "classification"
  detector:
    width_um: 12.0
    layout: "default10"
  logits:
    mode: "energy"
    normalize: "sum_to_one"

data:
  dataset: "mnist"
  root: "data/mnist/"
  split:
    train: 55000
    val: 5000
    test: 10000
  preprocess:
    grayscale: true
    normalize: "minmax_0_1"
    upsample_factor: 3
    resize_mode: "bilinear"
    pad_to: [200, 200]
    pad_mode: "constant"
    pad_value: 0.0
    pad_align: "center"

training:
  optimizer: "adam"
  lr: 0.01
  batch_size: 10
  epochs: 30
  loss: "mse_onehot"

eval:
  metric: "accuracy"

viz:
  enabled: true
  dpi: 300
  export_formats: ["png"]
  save_convergence: true
```

````

---

# 2) `docs/FIGURE_LAYOUT_SPECS.md` + 픽셀 고정 레이아웃 코드

## `docs/FIGURE_LAYOUT_SPECS.md`

```markdown
# Figure Layout Specs (pixel-precise)

목표: Fig.2/3/4를 "항상 같은 subplot 배치"로 저장.
- GridSpec 비율 기반도 가능하지만, 논문 figure처럼 재현하려면
  **픽셀 단위 bounding-box로 axis를 찍는 방식**이 가장 안정적이다.

근거:
- Fig.2/3는 4행×5열 패널 + 오른쪽 intensity/phase colorbar 배치 :contentReference[oaicite:9]{index=9}
- Fig.4는 (a)(b) 2-컬럼 + 각 서브피겨 top(구성도/샘플) + bottom(convergence) :contentReference[oaicite:10]{index=10}

---

## 좌표계 규약
- 모든 bbox는 (x_px, y_px, w_px, h_px)
- 원점(0,0)은 **캔버스의 좌상단**
- y는 아래로 증가
- matplotlib add_axes로 넣을 때는 y를 bottom-origin으로 변환 필요

---

## FIG2 Layout (layout_id = "fig2")
### 캔버스
- W=1800 px, H=1400 px, DPI=300
- 패널: 4 rows × 5 cols (각 패널 300×300 px)
- gap=10 px
- left label margin=150 px
- top margin=80 px
- bottom margin=90 px
- row별 colorbar(40×300 px) — rows 0~2: Intensity, row3: Phase

### Panels (img_r{r}_c{c})
- col x positions: 150, 460, 770, 1080, 1390
- row y positions: 80, 390, 700, 1010
- size: 300×300

### Colorbars
- cbar_x = 1700, w=40
- cbar_r0: (1700, 80, 40, 300)
- cbar_r1: (1700, 390, 40, 300)
- cbar_r2: (1700, 700, 40, 300)
- cbar_phase: (1700, 1010, 40, 300)

### Text anchors (figure-level text)
- row label x = 70
  - row0 center y=230: "Target Specimen"
  - row1 center y=540: "Co-saliency Detection [2]"
  - row2 center y=850: "F-D2NN Output"
  - row3 center y=1160: "F-D2NN Modulat. Layer"
- column group titles y=40
  - "Cell Type 1" at x=300
  - "Cell Type 2" at x=765  (cols 1-2 center)
  - "Cell Type 3" at x=1385 (cols 3-4 center)

---

## FIG3 Layout (layout_id = "fig3")
FIG2와 동일한 패널 bbox를 사용(4×5 + row colorbars).
- 다른 점은 column group titles가 없고 row0 label이 "Dynamic Scene"으로 바뀐다.

Row labels:
- row0: "Dynamic Scene"
- row1: "Co-saliency Detection [2]"
- row2: "F-D2NN Output"
- row3: "F-D2NN Modulat. Layer"

---

## FIG4 Layout (layout_id = "fig4")
### 캔버스
- W=2600 px, H=1600 px, DPI=300
- margins: left=80, right=80, top=60, bottom=80
- subfigure gutter=80
- subfigure width=1180, height=1460
- top region=850, gap=50, bottom plot=560

### Subfigure (a) — left half
- subfig_a origin = (80, 60)

Top region:
- title area height=50
- rows area: 4 rows, each h=180, gap=10

Config axes (a_cfg0..a_cfg3):
- x=80, w=420
- y=110, 300, 490, 680
- h=180

Image grid axes (a_img_r{r}_c{c}, r=0..3, c=0..2):
- panel size 180, gap 10
- x positions: 610, 800, 990
- y positions: 110, 300, 490, 680

Bottom plot axis:
- a_plot: (80, 960, 1180, 560)

### Subfigure (b) — right half
- subfig_b origin = (1340, 60)

Top region configs only (b_cfg0..b_cfg3):
- x=1340, w=1180
- y=110, 300, 490, 680
- h=180

Bottom plot:
- b_plot: (1340, 960, 1180, 560)

Subfigure labels:
- "(a)" at (90, 70)
- "(b)" at (1350, 70)
````

---

## `src/fd2nn/viz/layout_specs.py` (픽셀 bbox → matplotlib axes 생성)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

BBoxPx = Tuple[int, int, int, int]  # (x, y, w, h) in px, origin top-left


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    width_px: int
    height_px: int
    dpi: int
    axes: Dict[str, BBoxPx]
    texts: Dict[str, dict]  # free-form: {text_id: {"x":..., "y":..., "s":..., ...}}


def _px_to_axes_rect(bbox: BBoxPx, W: int, H: int) -> Tuple[float, float, float, float]:
    """
    Convert top-left pixel bbox to matplotlib add_axes rect (left,bottom,width,height) in [0,1].

    bbox: (x,y,w,h) with origin top-left, y downward.
    """
    x, y, w, h = bbox
    left = x / W
    width = w / W
    # convert top-left y to bottom-left
    bottom = 1.0 - (y + h) / H
    height = h / H
    return (left, bottom, width, height)


def make_axes_from_layout(fig, layout: LayoutSpec) -> Dict[str, object]:
    """
    Create matplotlib axes from pixel-perfect layout.
    Returns dict: axis_name -> matplotlib Axes.
    """
    axd = {}
    for name, bbox in layout.axes.items():
        rect = _px_to_axes_rect(bbox, layout.width_px, layout.height_px)
        axd[name] = fig.add_axes(rect)
    return axd


def get_layout(layout_id: str) -> LayoutSpec:
    if layout_id == "fig2":
        W, H, dpi = 1800, 1400, 300
        axes = {}

        # panel grid: 4 rows x 5 cols
        col_x = [150, 460, 770, 1080, 1390]
        row_y = [80, 390, 700, 1010]
        P = 300
        for r in range(4):
            for c in range(5):
                axes[f"img_r{r}_c{c}"] = (col_x[c], row_y[r], P, P)

        # per-row colorbars
        cbar_x, cbar_w = 1700, 40
        axes["cbar_r0"] = (cbar_x, row_y[0], cbar_w, P)
        axes["cbar_r1"] = (cbar_x, row_y[1], cbar_w, P)
        axes["cbar_r2"] = (cbar_x, row_y[2], cbar_w, P)
        axes["cbar_phase"] = (cbar_x, row_y[3], cbar_w, P)

        texts = {
            "row0": {"x": 70, "y": 230, "s": "Target\nSpecimen", "rotation": 90, "ha": "center", "va": "center"},
            "row1": {"x": 70, "y": 540, "s": "Co-saliency\nDetection [2]", "rotation": 90, "ha": "center", "va": "center"},
            "row2": {"x": 70, "y": 850, "s": "F-D2NN\nOutput", "rotation": 90, "ha": "center", "va": "center"},
            "row3": {"x": 70, "y": 1160, "s": "F-D2NN\nModulat. Layer", "rotation": 90, "ha": "center", "va": "center"},
            "title_ct1": {"x": 300, "y": 40, "s": "Cell Type 1", "ha": "center", "va": "center"},
            "title_ct2": {"x": 765, "y": 40, "s": "Cell Type 2", "ha": "center", "va": "center"},
            "title_ct3": {"x": 1385, "y": 40, "s": "Cell Type 3", "ha": "center", "va": "center"},
            "cbar_int_label": {"x": 1755, "y": 540, "s": "Intensity", "rotation": 90, "ha": "center", "va": "center"},
            "cbar_phase_label": {"x": 1755, "y": 1160, "s": "Phase", "rotation": 90, "ha": "center", "va": "center"},
        }
        return LayoutSpec("fig2", W, H, dpi, axes, texts)

    if layout_id == "fig3":
        # fig3는 bbox 동일, text만 변경
        base = get_layout("fig2")
        texts = dict(base.texts)
        texts["row0"] = {"x": 70, "y": 230, "s": "Dynamic\nScene", "rotation": 90, "ha": "center", "va": "center"}
        # column group titles 제거(원하면 빈 문자열로)
        for k in ["title_ct1", "title_ct2", "title_ct3"]:
            texts.pop(k, None)
        return LayoutSpec("fig3", base.width_px, base.height_px, base.dpi, base.axes, texts)

    if layout_id == "fig4":
        W, H, dpi = 2600, 1600, 300
        axes = {}

        # Common y positions for top rows
        row_y = [110, 300, 490, 680]
        h = 180

        # (a) subfigure
        a_x0 = 80
        axes["a_cfg0"] = (a_x0, row_y[0], 420, h)
        axes["a_cfg1"] = (a_x0, row_y[1], 420, h)
        axes["a_cfg2"] = (a_x0, row_y[2], 420, h)
        axes["a_cfg3"] = (a_x0, row_y[3], 420, h)

        img_x = [610, 800, 990]
        for r in range(4):
            for c in range(3):
                axes[f"a_img_r{r}_c{c}"] = (img_x[c], row_y[r], 180, 180)

        axes["a_plot"] = (a_x0, 960, 1180, 560)

        # (b) subfigure
        b_x0 = 1340
        axes["b_cfg0"] = (b_x0, row_y[0], 1180, h)
        axes["b_cfg1"] = (b_x0, row_y[1], 1180, h)
        axes["b_cfg2"] = (b_x0, row_y[2], 1180, h)
        axes["b_cfg3"] = (b_x0, row_y[3], 1180, h)

        axes["b_plot"] = (b_x0, 960, 1180, 560)

        texts = {
            "label_a": {"x": 90, "y": 70, "s": "(a)", "ha": "left", "va": "center"},
            "label_b": {"x": 1350, "y": 70, "s": "(b)", "ha": "left", "va": "center"},
            "a_title_cfg": {"x": 250, "y": 90, "s": "D$^2$NN Configurations", "ha": "center", "va": "center"},
            "a_title_sod": {"x": 800, "y": 90, "s": "Salient Object Detection", "ha": "center", "va": "center"},
            "a_title_cifar": {"x": 1080, "y": 90, "s": "CIFAR-10", "ha": "center", "va": "center"},
            "b_title_cfg": {"x": 1750, "y": 90, "s": "D$^2$NN Configurations", "ha": "center", "va": "center"},
        }
        return LayoutSpec("fig4", W, H, dpi, axes, texts)

    raise ValueError(f"Unknown layout_id: {layout_id}")
```

---

## 구현/해석 관점에서의 대안(Alternative options)

1. **GridSpec 기반(비율 레이아웃)**

   * 장점: 코드가 더 “matplotlib스럽고” 간결
   * 단점: 폰트/여백/저장 DPI에 따라 미세하게 흔들릴 수 있음
     → “논문 figure와 1:1 레이아웃 고정”이 목표면 지금 준 **픽셀 bbox 방식**이 더 안전.

2. **LaTeX/Illustrator로 후처리**

   * PRD 관점에서는 “자동 재현”에 어긋나서 비추천
   * 하지만 최종 논문용은 후처리 허용할 수도 있음(다만 재현성 아카이브에는 원본 PNG/PDF 저장)

---

## 지금 바로 적용할 액션 플랜

1. 위 `docs/CONFIG_TEMPLATES.md`와 `docs/FIGURE_LAYOUT_SPECS.md`, `src/fd2nn/viz/layout_specs.py`를 그대로 생성
2. `figure_factory.py`에서:

   * `layout = get_layout(cfg.viz.layout_id)`
   * `fig = plt.figure(figsize=(layout.width_px/layout.dpi, layout.height_px/layout.dpi), dpi=layout.dpi)`
   * `axd = make_axes_from_layout(fig, layout)`
   * `for text in layout.texts: fig.text(text["x"]/W, 1 - text["y"]/H, ...)` (좌표 변환 주의)
3. Fig2/3:

   * row0~2는 sample 이미지 5개 채우기
   * row3는 **phase masks 5개(레이어 1~5)**로 채우기(중요: 샘플과 무관) 
4. Fig4:

   * 각 config 실행해서 `history.json`(epoch-accuracy) 저장
   * fig4 스크립트는 저장된 history들을 로드해 `layout_id="fig4"`로 조합 렌더링 