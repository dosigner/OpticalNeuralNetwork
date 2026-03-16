# Fig. 3 Paper-Style Regeneration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current Fig. 3 renderer with a paper-style grouped bar chart, regenerate the figure artifacts, and add caption/analysis documentation for interpretation.

**Architecture:** Keep the existing evaluation path that produces measured grating periods for known and blind diffusers, but change the plotting layer to grouped bars with fixed labels, colors, and axis bounds. Add a focused regression test that checks the renderer's paper-style metadata and create a companion analysis note explaining the figure semantics.

**Tech Stack:** Python, pytest, matplotlib, NumPy, PyTorch

---

### Task 1: Add the failing figure-style test

**Files:**
- Create: `luo2022_random_diffusers_d2nn/tests/test_fig3_period_sweep.py`
- Modify: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/fig3_period_sweep.py`

**Step 1: Write the failing test**

Add a test that expects:
- x-axis label `Resolution Test Target Period, mm`
- y-axis label `Measured Grating Period, mm`
- panel titles matching the paper wording
- grouped bars for four `n` values in each panel
- a green dashed `True Period` indicator in the legend
- fixed y-limits starting at `4` and ending at `15`

**Step 2: Run test to verify it fails**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_fig3_period_sweep.py -v`

Expected: FAIL because the current renderer uses line/errorbar styling and different labels.

**Step 3: Write minimal implementation**

Refactor the plotting block in `fig3_period_sweep.py` so it:
- computes means/stds exactly as before
- renders grouped bars with error bars
- draws one dashed true-period marker per x-group
- applies fixed labels, titles, legend, limits, and ticks

**Step 4: Run test to verify it passes**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_fig3_period_sweep.py -v`

Expected: PASS

### Task 2: Add interpretation documentation

**Files:**
- Create: `luo2022_random_diffusers_d2nn/analysis/fig3_interpretation.ko.md`

**Step 1: Write the document**

Document:
- what each axis means
- difference between `Resolution Test Target Period` and `Measured Grating Period`
- how to interpret panel `(a)` vs `(b)`
- where to place the explanatory sentence in a caption/body discussion

**Step 2: Review for consistency**

Check the analysis note against the renderer labels and the paper wording.

### Task 3: Regenerate artifacts and verify outputs

**Files:**
- Regenerate: `luo2022_random_diffusers_d2nn/figures/fig3_period_sweep.png`
- Regenerate: `luo2022_random_diffusers_d2nn/figures/fig3_period_sweep.npy`

**Step 1: Run the targeted test suite**

Run: `pytest luo2022_random_diffusers_d2nn/tests/test_fig3_period_sweep.py -v`

Expected: PASS

**Step 2: Run the figure-generation command**

Run the existing `reproduce_fig3.py` command with the discovered checkpoints and baseline config.

**Step 3: Verify artifacts exist**

Confirm both the PNG and `.npy` files are updated and readable.

**Step 4: Commit**

```bash
git add luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/fig3_period_sweep.py \
        luo2022_random_diffusers_d2nn/tests/test_fig3_period_sweep.py \
        luo2022_random_diffusers_d2nn/analysis/fig3_interpretation.ko.md \
        luo2022_random_diffusers_d2nn/docs/plans/2026-03-16-fig3-paper-style-design.md \
        luo2022_random_diffusers_d2nn/docs/plans/2026-03-16-fig3-paper-style-implementation.md \
        luo2022_random_diffusers_d2nn/figures/fig3_period_sweep.png \
        luo2022_random_diffusers_d2nn/figures/fig3_period_sweep.npy
git commit -m "feat: regenerate fig3 in paper style"
```
