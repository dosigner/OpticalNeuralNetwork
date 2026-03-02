# Config Templates (spec-aligned)

All canonical experiment templates are under `src/tao2019_fd2nn/config/`.
Units:
- length: meters
- phase: radians

## Saliency
- `saliency_cell.yaml`: Fig.2 unit magnification (`f1=f2=10mm`, NA `0.112`, `800x800`, `dx=2um`)
- `saliency_cell_mag2x.yaml`: Supp S1 magnification (`f1=10mm`, `f2=20mm`)
- `saliency_cifar_video.yaml`: Fig.3 (`160x160`, `f1=f2=2mm`, resize `100x100`, pad `160x160`)

## MNIST classification (Fig.4)
- `cls_mnist_linear_real_5l.yaml`
- `cls_mnist_nonlinear_real_5l.yaml`
- `cls_mnist_linear_fourier_5l_f1mm.yaml`
- `cls_mnist_nonlinear_fourier_5l_f4mm.yaml`
- `cls_mnist_linear_real_10l.yaml`
- `cls_mnist_nonlinear_real_10l.yaml`
- `cls_mnist_hybrid_5l.yaml`
- `cls_mnist_hybrid_10l.yaml`

Shared paper parameters encoded in templates:
- `lambda=532nm`
- MNIST grid `200x200`, `dx=1um`, detector width `12um`
- classification `phase_max=pi`, `epochs=30`, `batch_size=10`, `lr=0.01`
- saliency `phase_max=2pi`, `epochs=100`, `batch_size=10`, `lr=0.01`
