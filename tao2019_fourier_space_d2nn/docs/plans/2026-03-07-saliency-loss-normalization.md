# Saliency Loss Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make saliency training compare prediction and target on a symmetric, full-frame normalization basis, and enable that behavior for the CIFAR paper-like config.

**Architecture:** Add explicit saliency loss preprocessing options in config/schema, route them through the trainer, and keep evaluation cropping limited to F-max/PR computation. Preserve old behavior as a selectable mode so existing experiments are not silently rewritten.

**Tech Stack:** Python, PyTorch, YAML config schema, pytest

---

### Task 1: Specify the failing behavior with tests

**Files:**
- Create: `tests/test_saliency_loss_preprocessing.py`
- Modify: `src/tao2019_fd2nn/training/trainer.py`
- Test: `tests/test_saliency_loss_preprocessing.py`

**Step 1: Write the failing test**

Add tests that assert:
- prediction and target are both min-max normalized per sample when the new mode is enabled
- loss uses the full frame instead of the cropped object region when the new scope is enabled

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_saliency_loss_preprocessing.py -q`

Expected: FAIL because the preprocessing helper/options do not exist yet.

### Task 2: Implement configurable saliency loss preprocessing

**Files:**
- Modify: `src/tao2019_fd2nn/config/schema.py`
- Modify: `src/tao2019_fd2nn/cli/train_saliency.py`
- Modify: `src/tao2019_fd2nn/training/trainer.py`

**Step 1: Add config schema defaults**

Add explicit saliency loss options under `training`, for example:
- `loss_normalization: "pred_only"`
- `loss_scope: "crop"`

**Step 2: Add minimal trainer support**

Implement a small helper that:
- applies optional per-sample min-max normalization to prediction
- applies optional per-sample min-max normalization to target
- selects `full` or `crop` tensors for loss only

**Step 3: Thread options from CLI into trainer**

Pass the new training options from `train_saliency.py` into `train_saliency()` / `run_saliency_epoch()`.

### Task 3: Enable the new behavior for the CIFAR paper-like config

**Files:**
- Modify: `tmp/saliency_cifar_fu2013_cat2horse_full_gpu.yaml`
- Modify: `tmp/saliency_cifar_fu2013_cat2horse_smoke_gpu.yaml`
- Modify: `src/tao2019_fd2nn/config/saliency_cifar_fu2013_cat2horse_bs10_100pad160_f2mm.yaml`

**Step 1: Update the config**

Set:
- `loss_normalization: "pred_and_target"`
- `loss_scope: "full"`

**Step 2: Keep eval behavior unchanged**

Do not change PR/F-max crop logic in the config; only loss preprocessing changes.

### Task 4: Verify

**Files:**
- Test: `tests/test_saliency_loss_preprocessing.py`
- Test: `tests/test_spec_configs_validate.py`

**Step 1: Run focused tests**

Run:
- `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_saliency_loss_preprocessing.py -q`
- `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_spec_configs_validate.py -q`

**Step 2: Run a short CUDA smoke check**

Run the CIFAR smoke config with `--max-steps-per-epoch 50` using the CUDA-safe `runpy` launch path and confirm `device=cuda` in the log.
