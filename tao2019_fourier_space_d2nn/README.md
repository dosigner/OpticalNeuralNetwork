# tao2019_fourier_space_d2nn

Config-driven simulator for Fourier-space Diffractive Deep Neural Networks (F-D2NN).

## Quick start

```bash
cd tao2019_fourier_space_d2nn
python -m pip install -e ".[dev]"
```

Classification:

```bash
tao2019-train-classifier --config src/tao2019_fd2nn/config/cls_mnist_linear_fourier_5l_f1mm.yaml
```

Saliency:

```bash
tao2019-train-saliency --config src/tao2019_fd2nn/config/saliency_cifar_video.yaml
```

Cell-GDC (for `saliency_cell.yaml`) requires external paired folders:

```text
data/cell_gdc/images/*.png
data/cell_gdc/masks/*.png
```

If your dataset is elsewhere, set in YAML:

```yaml
data:
  root: "/abs/path/to/cell_gdc"
  image_dir: "/abs/path/to/cell_gdc/images"
  mask_dir: "/abs/path/to/cell_gdc/masks"
```

Paper-style Cell patch prep (200x200 patching + split sampling):

```bash
python scripts/prepare_cell_gdc_patches.py \
  --type1-image /path/to/cell_type1_slide.png \
  --type2-image /path/to/cell_type2_slide.png \
  --type3-image /path/to/cell_type3_slide.png \
  --out-root data/cell_gdc \
  --patch-size 200 \
  --stride 200 \
  --train-count 2750 \
  --val-count 250 \
  --test-type1-count 500 \
  --test-type2-count 250 \
  --test-type3-count 250 \
  --mask-source ft
```

The script writes:

```text
data/cell_gdc/train/images, data/cell_gdc/train/masks
data/cell_gdc/val/images, data/cell_gdc/val/masks
data/cell_gdc/test_type1/images, data/cell_gdc/test_type1/masks
data/cell_gdc/test_type2/images, data/cell_gdc/test_type2/masks
data/cell_gdc/test_type3/images, data/cell_gdc/test_type3/masks
```

`tao2019-train-saliency` auto-detects `train/` and `val/` subfolders under `data.root`.

Live logs:

- Training prints real-time step/epoch metrics to terminal (loss, running metric, samples/s, ETA, GPU memory).
- The same stream is saved to `runs/<exp>/<timestamp>/logs/train.log`.
- Tune verbosity in YAML:
  - `training.log_interval_steps` (default: `20`)
  - `training.color_logs` (default: `true`)
  - `training.show_cuda_memory` (default: `true`)
- Saliency speed knobs:
  - `training.compute_train_fmax` (default: `false`)
  - `training.eval_interval_epochs` (default: `5`, last epoch always evaluated)

Run tests:

```bash
pytest
```
