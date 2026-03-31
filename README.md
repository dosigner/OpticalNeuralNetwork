# OpticalNeuralNetwork

A PyTorch implementation of **Diffractive Deep Neural Networks (D²NN)** and
**Fourier Diffractive Deep Neural Networks (Fourier D²NN)** – all-optical
architectures for computer vision tasks such as classification and pattern
recognition.

---

## Background

Diffractive Deep Neural Networks replace electronic computation with light
propagation.  Each *diffractive layer* is a thin phase/amplitude mask; when
stacked and separated by free-space regions, the cascade realises a deep
network whose inference runs at the speed of light.

Two families of architectures are provided:

| Model | Propagation | Reference |
|-------|-------------|-----------|
| **D²NN** | Free-space Angular Spectrum Method (ASM) between layers | Lin et al., *Science* 361, 1004–1008 (2018) |
| **Fourier D²NN** | 4-f Fourier optical system; each layer modulates spatial frequencies | Fourier-optics extension of D²NN |

---

## Repository layout

```
OpticalNeuralNetwork/
├── d2nn/
│   ├── __init__.py          # public API
│   ├── propagation.py       # ASM & Fourier-lens propagation kernels
│   ├── layers.py            # DiffractiveLayer, FourierDiffractiveLayer
│   ├── models.py            # D2NN, FourierD2NN end-to-end models
│   └── utils.py             # visualisation helpers
├── examples/
│   └── train_mnist.py       # MNIST classification demo
├── tests/
│   └── test_d2nn.py         # unit tests
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

### Programmatic API

```python
import torch
from d2nn import D2NN, FourierD2NN

# --- D²NN (free-space propagation) ---
model = D2NN(
    num_layers=5,
    field_size=(28, 28),   # H × W pixels
    num_classes=10,
    wavelength=532e-9,     # 532 nm green laser
    z=0.1,                 # 10 cm between layers
    dx=8e-6,               # 8 µm pixel pitch
)

images = torch.rand(8, 1, 28, 28)   # batch of 8 images
logits = model(images)               # (8, 10)
print(logits.argmax(1))

# --- Fourier D²NN (4-f Fourier-plane processing) ---
fmodel = FourierD2NN(num_layers=5, field_size=(28, 28), num_classes=10)
logits = fmodel(images)
```

### Train on MNIST

```bash
# Train D²NN
python examples/train_mnist.py --model d2nn --epochs 20

# Train Fourier D²NN
python examples/train_mnist.py --model fourier --epochs 20

# Enable amplitude + phase modulation
python examples/train_mnist.py --model d2nn --complex_modulation --epochs 20
```

Results (loss/accuracy curves, phase-mask visualisations, output-plane
samples) are written to `results/<model>_mnist/`.

---

## Architecture details

### DiffractiveLayer (`d2nn/layers.py`)

Each neuron at position *(i, j)* learns a **phase shift** φ(i, j) ∈ ℝ
(physically, modulo 2π).  The complex transmission function is

    t(i, j) = a(i, j) · exp(j · φ(i, j))

where *a* = 1 (phase-only, default) or a learnable sigmoid amplitude
(when `complex_modulation=True`).

After element-wise modulation the field propagates **z** metres via the
**Angular Spectrum Method**:

    U_out = IFFT{ FFT{U_in · t} · H(fx, fy) }

    H(fx, fy) = exp(j 2π z √(1/λ² − fx² − fy²))   for propagating modes
              = 0                                    for evanescent modes

### FourierDiffractiveLayer (`d2nn/layers.py`)

Implements a **4-f Fourier optical processor**:

    input  →  FFT (forward lens)  →  phase mask in Fourier plane
           →  IFFT (backward lens)  →  output

The learnable mask modulates **spatial frequencies** of the field,
acting as an optical convolutional filter.

### Detector readout (`d2nn/models.py`)

The output plane is partitioned into *C* non-overlapping rectangular
**detector regions** (one per class).  The total optical intensity
integrated over each region is the raw classification score:

    score_c = Σ_{(i,j) ∈ region_c} |U(i, j)|²

---

## Running tests

```bash
pytest tests/ -v
```

---

## License

MIT