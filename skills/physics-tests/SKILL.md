---
name: physics-tests
description: Generate and run a physics validation test suite for D2NN/FD2NN optical pipeline. Tests energy conservation, sampling, normalization, and propagation correctness.
---

# Physics Validation Test Suite

Generate pytest tests that validate the physical correctness of the D2NN optical pipeline, then iterate until all tests pass.

## Test Categories

### 1. Energy Conservation
```python
# Input power ~= output power within tolerance at each propagation step
# Tolerance: 5% for single step, 10% cumulative
def test_energy_conservation_per_layer():
    """Power in ~= power out for each diffractive layer."""

def test_energy_conservation_end_to_end():
    """Total pipeline energy budget within 10%."""
```

### 2. Sampling & Aliasing
```python
# Verify Nyquist criterion is satisfied
# dx < lambda / (2 * NA)
def test_no_aliasing():
    """Pixel pitch satisfies Nyquist for given NA and wavelength."""

def test_pixel_pitch_units():
    """Pixel pitch is in meters (not mm or um) internally."""
```

### 3. Normalization
```python
def test_strehl_ratio_range():
    """Strehl ratio is in [0, 1] and computed from raw beams."""

def test_intensity_spatial_mean():
    """Spatial mean intensity is normalized correctly."""

def test_baseline_uses_raw_beams():
    """CO/IO baselines computed from raw input, not D2NN output."""
```

### 4. Propagation Correctness
```python
def test_beam_reducer_vs_zoom():
    """Beam reducer and zoom propagate produce different results."""

def test_free_space_propagation_symmetry():
    """Forward then backward propagation recovers input field."""

def test_focal_length_physical_realism():
    """Focal length is in realistic range [10mm, 500mm]."""
```

### 5. Training Stability
```python
def test_loss_finite():
    """Loss function outputs are finite (no NaN/Inf)."""

def test_gradients_flow():
    """Gradients are non-zero for all trainable parameters."""

def test_phase_values_bounded():
    """Phase mask values stay within expected range."""
```

## Workflow

1. Read the current optical pipeline code
2. Identify all physical assumptions
3. Generate the test suite as `tests/test_physics.py`
4. Run: `pytest tests/test_physics.py -v`
5. For every failing test:
   - Diagnose the physics bug
   - Fix the code
   - Re-run until all tests pass
6. Report: what was broken and what was fixed

## Rules
- Make reasonable physics assumptions and document in test docstrings
- Do NOT mock the optical propagation - test real physics
- Maximum 5 fix iterations before escalating to human
- Separate evidence from inference in failure diagnosis
