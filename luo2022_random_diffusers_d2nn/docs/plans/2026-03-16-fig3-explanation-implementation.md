# Fig. 3 Explanation Figure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new explanation figure that uses three real examples to show how the input target period, diffuser-only propagation baseline, and measured D2NN reconstruction period relate to Fig. 3.

**Architecture:** Reuse the existing grating-generation, diffuser-generation, and D2NN forward path so the explanation figure reflects the same evaluation pipeline as Fig. 3. Add a diffuser-only free-space baseline path, render a compact `3 x 4` layout with image panels plus a D2NN profile plot that marks the true and measured periods, and keep the same diffuser instance for the baseline and D2NN panels within each row.

**Tech Stack:** Python, matplotlib, NumPy, PyTorch, pytest

---

### Task 1: Add the failing renderer test

**Files:**
- Create: `luo2022_random_diffusers_d2nn/tests/test_fig3_period_explanation.py`
- Create: `luo2022_random_diffusers_d2nn/src/luo2022_d2nn/figures/fig3_period_explanation.py`

**Step 1: Write the failing test**

Add a test that expects:
- a `3 x 4` panel layout for the selected periods,
- column titles for input, diffuser-only propagation, reconstruction, and profile,
- one figure caption at the bottom,
- the right-column profile axes to include text for both true and measured periods,
- the returned metadata to include the diffuser-only images.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_fig3_period_explanation.py -v`

Expected: FAIL because the current renderer still exposes a `3 x 3` layout and does not surface the diffuser-only baseline.

**Step 3: Write minimal implementation**

Implement the new figure module and keep the API small:
- load checkpoint,
- generate one deterministic diffuser,
- generate `7.2`, `10.8`, `12.0 mm` targets,
- propagate each target through the diffuser to the output plane without D2NN,
- run reconstructions,
- compute averaged profiles and measured periods,
- render the `3 x 4` explanation layout with a bottom caption.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_fig3_period_explanation.py -v`

Expected: PASS

### Task 2: Add a reproduction script

**Files:**
- Create: `luo2022_random_diffusers_d2nn/scripts/reproduce_fig3_explanation.py`

**Step 1: Add a thin script wrapper**

The script should call the new figure function with:
- one checkpoint path,
- optional config path,
- optional output path.

**Step 2: Verify the script runs**

Run the script with the `n20_L4` checkpoint and confirm it writes a PNG.

### Task 3: Regenerate the explanation artifact

**Files:**
- Regenerate: `luo2022_random_diffusers_d2nn/figures/fig3_period_explanation.png`

**Step 1: Run the targeted tests**

Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_fig3_period_explanation.py tests/test_grating_period.py -v`

Expected: PASS

**Step 2: Generate the explanation figure**

Run the reproduction script using the chosen checkpoint.

**Step 3: Review the artifact**

Check that the caption is visible, the diffuser-only baseline is visibly degraded relative to the D2NN reconstruction, and the profile panel clearly differentiates true vs measured periods.
