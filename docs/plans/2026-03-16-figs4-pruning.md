# Supplementary Figure S4 Pruning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible Supplementary Figure S4 generator that compares reconstruction outputs across six pruning conditions using the existing `n20_L4` checkpoint, and verify that the resulting panel layout, labels, and PCC trend qualitatively match the paper.

**Architecture:** Implement a new figure module `figs4_pruning.py` that reuses the existing S3 island-mask extraction and Fig. 2 evaluation flow. Keep pruning-mask derivation, pruned-phase materialization, object loading, simulation, and panel assembly in one figure module to minimize new surface area. Add a CLI wrapper, focused unit tests, and integrate the figure into the existing all-figures script.

**Tech Stack:** Python, NumPy, PyTorch, SciPy ndimage, Matplotlib, pytest

---

### Task 1: Add mask-derivation tests first

**Files:**
- Create: `luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py`
- Reference: `luo2022_random_diffusers_d2nn/tests/test_figs3_overlap_map.py`

**Step 1: Write the failing test**

Add tests for:
- row-condition masks are returned for exactly 6 conditions
- `no_layers` keeps 0 active pixels
- `islands_only` active area is less than `dilated_islands`
- `dilated_islands` active area is less than or equal to `inside_contour`
- `inside_contour` active area is less than or equal to `aperture_80lambda`
- `full_layers` and `aperture_80lambda` match the circular ROI when expected

**Step 2: Run test to verify it fails**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k masks -v`

Expected: FAIL because the S4 module and helpers do not exist yet.

**Step 3: Write minimal implementation**

Create `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py` with:
- condition-name constants
- helpers to build:
  - circular ROI
  - islands-only mask
  - dilated mask
  - inside-contour mask
  - aperture mask
- a helper returning per-condition masks and kept-area ratios

**Step 4: Run test to verify it passes**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k masks -v`

Expected: PASS

**Step 5: Commit**

```bash
git add luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py
git commit -m "test: add S4 pruning mask coverage"
```

### Task 2: Add phase-materialization tests

**Files:**
- Modify: `luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py`
- Modify: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py`

**Step 1: Write the failing test**

Add tests for:
- per-condition wrapped phase stacks preserve original phase values inside active support
- inactive support becomes zero phase modulation
- `full_layers` matches original wrapped phase stack
- `no_layers` returns all-zero phase maps

**Step 2: Run test to verify it fails**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k phase -v`

Expected: FAIL because phase materialization helpers are incomplete.

**Step 3: Write minimal implementation**

Add helpers to:
- load wrapped phase stack from a checkpoint
- apply boolean masks to wrapped phases
- produce one phase stack per pruning condition

**Step 4: Run test to verify it passes**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k phase -v`

Expected: PASS

**Step 5: Commit**

```bash
git add luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py
git commit -m "feat: add S4 pruned phase materialization"
```

### Task 3: Add OOD object loader tests

**Files:**
- Modify: `luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py`
- Modify: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py`
- Create: `luo2022_random_diffusers_d2nn/reference/ood_s4_object.png`

**Step 1: Write the failing test**

Add tests for:
- grayscale input is loaded into the target canvas size
- pure white background becomes 0 intensity
- non-white pixels retain grayscale structure
- output tensor shape matches the MNIST evaluation shape

Use a small temporary synthetic grayscale image in the test; do not depend on the real asset in the unit test.

**Step 2: Run test to verify it fails**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k ood -v`

Expected: FAIL because the OOD loader does not exist.

**Step 3: Write minimal implementation**

Add helpers to:
- load a grayscale image
- zero out pure-white background
- resize/pad into the configured final canvas
- return an amplitude tensor aligned with the existing figure code

Save the user-provided asset as `luo2022_random_diffusers_d2nn/reference/ood_s4_object.png`.

**Step 4: Run test to verify it passes**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k ood -v`

Expected: PASS

**Step 5: Commit**

```bash
git add luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py luo2022_random_diffusers_d2nn/reference/ood_s4_object.png
git commit -m "feat: add S4 OOD object loader"
```

### Task 4: Add end-to-end figure-generation test

**Files:**
- Modify: `luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py`
- Modify: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py`

**Step 1: Write the failing test**

Add a compact end-to-end test that:
- builds a small fake checkpoint with 4 phase layers
- calls `make_figs4(...)`
- verifies PNG output exists
- verifies returned metadata includes 6 row labels, 4 layer labels, 4 output-column labels
- verifies PCC arrays and kept-area ratios have expected shapes

Mock or simplify expensive parts only if needed; prefer a real small-grid run.

**Step 2: Run test to verify it fails**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k end_to_end -v`

Expected: FAIL because `make_figs4` is not fully implemented.

**Step 3: Write minimal implementation**

Implement:
- fixed digit-2 sample loading
- known/new diffuser generation
- per-condition evaluation
- contrast-enhanced display image creation
- 6x8 panel figure assembly with:
  - row condition labels on the first block
  - layer titles
  - output column titles
  - per-panel PCC text
  - kept-area text
- metadata return and `.npy` export

**Step 4: Run test to verify it passes**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k end_to_end -v`

Expected: PASS

**Step 5: Commit**

```bash
git add luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py
git commit -m "feat: implement supplementary figure S4 generator"
```

### Task 5: Add CLI entry point and figure integration

**Files:**
- Create: `luo2022_random_diffusers_d2nn/scripts/reproduce_figS4.py`
- Modify: `luo2022_random_diffusers_d2nn/scripts/generate_all_figures.py`
- Modify: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/__init__.py`
- Modify: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/cli/make_figures.py`

**Step 1: Write the failing test**

Add a small integration-oriented test or smoke assertion that the CLI wrapper calls `make_figs4` with expected arguments. If no existing CLI tests exist, add a narrow import/smoke test rather than overbuilding CLI coverage.

**Step 2: Run test to verify it fails**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k cli -v`

Expected: FAIL until the CLI wrapper and registrations are added.

**Step 3: Write minimal implementation**

Add:
- `reproduce_figS4.py`
- S4 section to `generate_all_figures.py`
- any necessary figure-module exports
- any existing figure CLI registration if that file expects it

Default output:
- `figures/figS4_pruning.png`

**Step 4: Run test to verify it passes**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -k cli -v`

Expected: PASS

**Step 5: Commit**

```bash
git add luo2022_random_diffusers_d2nn/scripts/reproduce_figS4.py luo2022_random_diffusers_d2nn/scripts/generate_all_figures.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/__init__.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/cli/make_figures.py luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py
git commit -m "feat: wire supplementary figure S4 into figure scripts"
```

### Task 6: Run targeted verification

**Files:**
- Test: `luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py`

**Step 1: Run the focused test file**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py -v`

Expected: PASS

**Step 2: Run adjacent figure tests**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_figs3_overlap_map.py luo2022_random_diffusers_d2nn/tests/test_fig1b_distortion.py -v`

Expected: PASS

**Step 3: Fix any breakage minimally**

If a regression appears, patch only the affected logic and re-run the failing tests.

**Step 4: Commit**

```bash
git add luo2022_random_diffusers_d2nn/tests/test_figs4_pruning.py luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/figs4_pruning.py
git commit -m "test: verify supplementary figure S4 flow"
```

### Task 7: Generate the real figure with the existing checkpoint

**Files:**
- Output: `luo2022_random_diffusers_d2nn/figures/figS4_pruning.png`
- Output: `luo2022_random_diffusers_d2nn/figures/figS4_pruning.npy`

**Step 1: Run the reproduction script**

Run:

```bash
cd /root/dj/D2NN/luo2022_random_diffusers_d2nn
PYTHONPATH=src python scripts/reproduce_figS4.py --checkpoint runs/n20_L4/model.pt --config configs/baseline.yaml --output figures/figS4_pruning.png
```

Expected: PNG and NPY files are created successfully.

**Step 2: Inspect metadata**

Check:
- 6 row labels
- 4 output-column labels
- PCC array shape `(6, 4)` or equivalent documented layout
- kept-area ratios increase in the expected order

**Step 3: If generation fails, fix minimally**

Patch only the specific runtime issue and re-run the script.

**Step 4: Commit**

```bash
git add luo2022_random_diffusers_d2nn/figures/figS4_pruning.png luo2022_random_diffusers_d2nn/figures/figS4_pruning.npy
git commit -m "feat: generate supplementary figure S4 outputs"
```

### Task 8: Compare against the paper and record outcome

**Files:**
- Modify: `docs/plans/2026-03-16-figs4-pruning-design.md`
- Optional: `luo2022_random_diffusers_d2nn/analysis/analysis_figS4_pruning.md`

**Step 1: Open the supplementary PDF and generated figure side by side**

Compare:
- row count and order
- layer-panel and output-panel grouping
- visible row labels
- output column labels
- PCC trend by row

**Step 2: Record comparison notes**

Document:
- what matches
- any qualitative deviations
- whether the success criterion was met

**Step 3: Commit**

```bash
git add docs/plans/2026-03-16-figs4-pruning-design.md
git commit -m "docs: record supplementary figure S4 comparison notes"
```
