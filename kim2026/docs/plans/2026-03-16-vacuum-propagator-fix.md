# Vacuum Propagator Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current alias-prone zoom Fresnel propagator so 1 km vacuum propagation produces a physically smooth Gaussian beam before any turbulence cache is regenerated.

**Architecture:** Add a stricter vacuum-shape regression test first, then replace the current two-step Fresnel implementation with a separable direct Fresnel integral that supports distinct input/output windows without the intermediate-plane chirp aliasing failure. Keep the public propagator API stable where practical, but allow internal helper and cache changes as needed.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Lock the bug with a failing vacuum-shape test

**Files:**
- Modify: `tests/test_optics_core.py`

**Step 1: Write the failing test**
- Add a test that propagates the `1550 nm`, `0.3 mrad` Gaussian from `0.03 m` to `0.72 m` over `1000 m`.
- Compare the normalized detector-plane intensity against the analytic Gaussian intensity on the full 2-D output grid.
- Fail if the max absolute error is too large.

**Step 2: Run test to verify it fails**
- Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_optics_core.py -k vacuum_shape -v`

### Task 2: Replace the propagator core

**Files:**
- Modify: `src/kim2026/optics/scaled_fresnel.py`

**Step 1: Implement the minimal replacement**
- Replace the current two-step Fresnel chain with a separable direct Fresnel integral on arbitrary output coordinates.
- Precompute and cache the input/output coordinate axes and the 1-D Fresnel kernel matrices for repeated shapes.
- Keep the existing `scaled_fresnel_propagate(...)` entrypoint.

**Step 2: Run the optics tests**
- Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_optics_core.py -v`

### Task 3: Verify the dependent pipeline

**Files:**
- Verify only: `tests/test_turbulence_pipeline.py`, `tests/test_training_core.py`

**Step 1: Run the dependent tests**
- Run: `PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_turbulence_pipeline.py tests/test_training_core.py -v`

### Task 4: Recheck the vacuum field numerically

**Files:**
- Verify only: `src/kim2026/turbulence/channel.py`

**Step 1: Run a one-off vacuum probe**
- Propagate a vacuum Gaussian with the pilot parameters and inspect the intensity map/profile numerically.
- Confirm the field is smooth and Gaussian-like before any cache regeneration.
