# GT Diagnosis And Variant Pilots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the CIFAR Fu2013 pseudo-GT is a primary cause of blob-like FD2NN saliency outputs by auditing GT morphology first and then running GT-only pilot variants.

**Architecture:** Keep the existing paper-like FD2NN, optics, and training loop fixed. Add a GT audit utility that measures mask morphology and produces visual overlays, then generate a small set of deterministic GT variants and compare them with short pilots under the same training budget.

**Tech Stack:** Python, PyTorch, PIL, NumPy, pytest, existing `tao2019_fd2nn` CLI/config system.

---

### Task 1: Document the current GT pipeline inputs

**Files:**
- Modify: `reports/analysis_s7_physics.md` (optional note if needed)
- Inspect: `src/tao2019_fd2nn/data/saliency_pairs.py`
- Inspect: `scripts/prepare_cifar10_fu2013_cosaliency.py`
- Inspect: `src/tao2019_fd2nn/config/saliency_cifar_fu2013_cat2horse_bs10_100pad160_f2mm.yaml`

**Step 1: Confirm baseline data root and split layout**

Run: `find data/cifar10_cosaliency_fu2013_g5 -maxdepth 2 -type d | sort`
Expected: `train/images`, `train/masks`, `val/images`, `val/masks`

**Step 2: Confirm loader assumptions**

Run: `sed -n '1,220p' src/tao2019_fd2nn/data/saliency_pairs.py`
Expected: mask/image filename pairing and resize+pad preprocessing are clear

### Task 2: Add failing tests for GT audit metrics and GT variants

**Files:**
- Create: `tests/test_gt_audit.py`
- Create: `tests/test_gt_variants.py`

**Step 1: Write failing GT audit metric tests**

Cover:
- center-of-mass bias on a centered blob vs off-center blob
- edge density difference between blurred and sharp masks
- connected-component count for single vs fragmented masks

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gt_audit.py tests/test_gt_variants.py -q`
Expected: FAIL because audit/variant helpers do not exist yet

### Task 3: Implement GT audit utilities

**Files:**
- Create: `src/tao2019_fd2nn/analysis/gt_audit.py`
- Create: `scripts/audit_cifar_fu2013_gt.py`

**Step 1: Implement metric helpers**

Include:
- foreground area ratio
- center-of-mass offset
- edge density
- largest connected component ratio
- component count

**Step 2: Implement report script**

Outputs under `reports/gt_audit/`:
- `summary.json`
- `train_overlay_grid.png`
- `val_overlay_grid.png`

**Step 3: Re-run tests**

Run: `pytest tests/test_gt_audit.py -q`
Expected: PASS

### Task 4: Add GT variant generation utilities

**Files:**
- Create: `src/tao2019_fd2nn/data/gt_variants.py`
- Create: `scripts/prepare_cifar10_fu2013_variants.py`

**Step 1: Implement deterministic variants**

Include:
- `raw` passthrough
- `binary_otsu` or deterministic thresholded binary mask
- `sharpened` edge-preserving variant

**Step 2: Re-run tests**

Run: `pytest tests/test_gt_variants.py -q`
Expected: PASS

### Task 5: Create pilot configs that differ only by GT root

**Files:**
- Create: `tmp/saliency_cifar_fu2013_raw_pilot.yaml`
- Create: `tmp/saliency_cifar_fu2013_binary_pilot.yaml`
- Create: `tmp/saliency_cifar_fu2013_sharpened_pilot.yaml`

**Step 1: Copy baseline config**

Keep fixed:
- optics
- `rear` SBN
- loss mode
- seed
- training budget except for pilot reduction

**Step 2: Override only these fields**

- `experiment.save_dir`
- `data.root`
- `training.epochs = 15`
- pilot `num_workers = 0`

### Task 6: Run GT audit and GT-only pilots

**Files:**
- Output: `reports/gt_audit/*`
- Output: `reports/tuning_runs/*`
- Output: `reports/gt_variant_pilots_2026-03-09.json`

**Step 1: Run GT audit**

Run: `PYTHONPATH=src python scripts/audit_cifar_fu2013_gt.py`
Expected: summary + overlay figures are created

**Step 2: Run three pilot trainings**

Run one pilot per config with the same budget

**Step 3: Aggregate metrics**

Write JSON with:
- variant name
- eval_fmax
- best_epoch
- qualitative note

### Task 7: Summarize the root-cause evidence

**Files:**
- Create: `reports/gt_quality_diagnosis_2026-03-09.md`

**Step 1: Decide whether GT is a primary blob driver**

Rule:
- if GT metrics show strong center bias and GT-only variant changes materially alter output shape, conclude GT is a primary cause
- otherwise move next to SBN operating-point experiments

**Step 2: Record next recommendation**

Include one concrete next axis only
