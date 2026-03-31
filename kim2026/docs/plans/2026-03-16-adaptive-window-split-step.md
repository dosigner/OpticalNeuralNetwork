# Adaptive Window Split-Step Propagator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current fixed-window long-range propagation path with an alias-safe adaptive-window split-step propagator that supports the `1550 nm`, `1 km`, `0.3 mrad` Gaussian beam schedule and attaches turbulence screens using per-screen equivalent `r0_j` values derived from their represented physical path cells.

**Architecture:** Build the propagation stack around an explicit schedule. The schedule owns the event planes, window ladder, interval lengths, and screen cell lengths. Most intervals use a same-window free-space propagator; only window changes use a zoom/regrid operator. Turbulence screens are inserted at physical screen planes and weighted by their midpoint-cell `Δz_j`, not by the numerical `dz_i` values. Keep the existing public channel entrypoint shape where practical, but restructure the optics layer under it.

**Tech Stack:** Python, PyTorch, pytest, YAML config

---

### Task 1: Lock the schedule and screen weighting rules with failing tests

**Files:**
- Create: `tests/test_propagation_schedule.py`
- Modify: `tests/test_turbulence_pipeline.py`

**Step 1: Write the failing schedule test**

```python
from kim2026.optics.propagation_schedule import build_adaptive_schedule


def test_adaptive_schedule_matches_expected_window_ladder() -> None:
    schedule = build_adaptive_schedule(
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
        path_length_m=1000.0,
        receiver_window_m=0.72,
        internal_receiver_window_m=0.96,
        source_window_m=0.03,
        window_ladder_m=[0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
        screen_positions_m=[125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0],
        beam_diameter_fill_fraction=0.625,
    )

    assert [round(interval.start_z_m, 2) for interval in schedule.intervals] == [
        0.0, 31.25, 62.5, 125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0
    ]
    assert [round(interval.end_z_m, 2) for interval in schedule.intervals] == [
        31.25, 62.5, 125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0, 1000.0
    ]
    assert [interval.window_m for interval in schedule.intervals] == [
        0.03, 0.06, 0.12, 0.24, 0.48, 0.48, 0.96, 0.96, 0.96, 0.96
    ]
```

**Step 2: Write the failing screen-cell test**

```python
from kim2026.optics.propagation_schedule import build_adaptive_schedule


def test_screen_cells_use_midpoint_rule_for_equivalent_r0() -> None:
    schedule = build_adaptive_schedule(
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
        path_length_m=1000.0,
        receiver_window_m=0.72,
        internal_receiver_window_m=0.96,
        source_window_m=0.03,
        window_ladder_m=[0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
        screen_positions_m=[125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0],
        beam_diameter_fill_fraction=0.625,
    )

    assert [cell.length_m for cell in schedule.screen_cells] == [
        187.5, 125.0, 125.0, 125.0, 125.0, 125.0, 187.5
    ]
```

**Step 3: Write the failing turbulence weighting test**

```python
import math

from kim2026.optics.propagation_schedule import build_adaptive_schedule
from kim2026.turbulence.channel import equivalent_r0_for_cell


def test_equivalent_r0_matches_cn2_for_screen_cells() -> None:
    schedule = build_adaptive_schedule(
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
        path_length_m=1000.0,
        receiver_window_m=0.72,
        internal_receiver_window_m=0.96,
        source_window_m=0.03,
        window_ladder_m=[0.03, 0.06, 0.12, 0.24, 0.48, 0.96],
        screen_positions_m=[125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0],
        beam_diameter_fill_fraction=0.625,
    )

    r0_values = [
        equivalent_r0_for_cell(
            wavelength_m=1.55e-6,
            cn2=2.0e-14,
            cell_length_m=cell.length_m,
        )
        for cell in schedule.screen_cells
    ]

    assert math.isclose(r0_values[0], 0.14137063694418073, rel_tol=1e-6)
    assert math.isclose(r0_values[1], 0.18030757402768166, rel_tol=1e-6)
    assert math.isclose(r0_values[-1], 0.14137063694418073, rel_tol=1e-6)
```

**Step 4: Run tests to verify they fail**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_propagation_schedule.py tests/test_turbulence_pipeline.py -k "schedule or equivalent_r0" -v
```

Expected:
- FAIL because `kim2026.optics.propagation_schedule` and `equivalent_r0_for_cell` do not exist yet.

### Task 2: Implement the schedule builder

**Files:**
- Create: `src/kim2026/optics/propagation_schedule.py`
- Modify: `src/kim2026/optics/__init__.py`

**Step 1: Write the minimal schedule data structures**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenCell:
    screen_index: int
    z_m: float
    start_z_m: float
    end_z_m: float
    length_m: float


@dataclass(frozen=True)
class PropagationInterval:
    index: int
    start_z_m: float
    end_z_m: float
    dz_m: float
    window_m: float
    zoom_to_window_m: float | None
    screen_indices_at_end: tuple[int, ...]


@dataclass(frozen=True)
class AdaptiveSchedule:
    intervals: tuple[PropagationInterval, ...]
    screen_cells: tuple[ScreenCell, ...]
```

**Step 2: Implement the screen-cell builder**

```python
def build_screen_cells(path_length_m: float, screen_positions_m: list[float]) -> tuple[ScreenCell, ...]:
    cells = []
    for index, z_m in enumerate(screen_positions_m):
        start_z_m = 0.0 if index == 0 else 0.5 * (screen_positions_m[index - 1] + z_m)
        end_z_m = path_length_m if index == len(screen_positions_m) - 1 else 0.5 * (z_m + screen_positions_m[index + 1])
        cells.append(
            ScreenCell(
                screen_index=index,
                z_m=float(z_m),
                start_z_m=float(start_z_m),
                end_z_m=float(end_z_m),
                length_m=float(end_z_m - start_z_m),
            )
        )
    return tuple(cells)
```

**Step 3: Implement the adaptive event-plane builder**

```python
def build_adaptive_schedule(...):
    # 1. Compute window switch planes from beam-diameter fill threshold.
    # 2. Merge switch planes with screen positions and path end.
    # 3. Assign the active window to each interval.
    # 4. Mark zoom transitions only where the next interval uses a larger window.
    # 5. Attach screen indices to the interval end plane where the screen lives.
```

Use the following exact event planes for the default pilot design:
- switch planes: `31.25, 62.5, 125.0, 250.0, 500.0`
- screen planes: `125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0`
- path end: `1000.0`

**Step 4: Run schedule tests**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_propagation_schedule.py -v
```

Expected:
- PASS

### Task 3: Split free-space propagation into same-window and zoom paths

**Files:**
- Create: `src/kim2026/optics/angular_spectrum.py`
- Create: `src/kim2026/optics/zoom_propagate.py`
- Modify: `src/kim2026/optics/scaled_fresnel.py`
- Modify: `src/kim2026/optics/__init__.py`
- Modify: `tests/test_optics_core.py`

**Step 1: Write the failing same-window vacuum test**

```python
from kim2026.optics.angular_spectrum import propagate_same_window


def test_same_window_propagator_preserves_vacuum_energy() -> None:
    field, _, _ = make_collimated_gaussian_field(
        n=256,
        window_m=0.24,
        wavelength_m=1.55e-6,
        half_angle_rad=3.0e-4,
    )

    propagated = propagate_same_window(
        field.unsqueeze(0),
        wavelength_m=1.55e-6,
        window_m=0.24,
        z_m=50.0,
    ).squeeze(0)

    assert math.isclose(
        float(propagated.abs().square().sum().item()),
        float(field.abs().square().sum().item()),
        rel_tol=1e-3,
    )
```

**Step 2: Implement the same-window propagator**

```python
def propagate_same_window(field: torch.Tensor, *, wavelength_m: float, window_m: float, z_m: float) -> torch.Tensor:
    # 1. Build centered spatial-frequency axes from window_m and N.
    # 2. Apply a band-limited angular-spectrum transfer function.
    # 3. Return a complex64 tensor with the original shape.
```

**Step 3: Move the current direct Fresnel integral behind a zoom-only API**

```python
def zoom_propagate(field: torch.Tensor, *, wavelength_m: float, source_window_m: float, destination_window_m: float, z_m: float) -> torch.Tensor:
    return scaled_fresnel_propagate(
        field,
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=destination_window_m,
        z_m=z_m,
    )
```

**Step 4: Run optics tests**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_optics_core.py -v
```

Expected:
- PASS

### Task 4: Rebuild the split-step driver around the schedule

**Files:**
- Modify: `src/kim2026/turbulence/channel.py`
- Modify: `src/kim2026/training/targets.py`
- Modify: `src/kim2026/models/d2nn.py`
- Modify: `src/kim2026/training/trainer.py`
- Modify: `src/kim2026/cli/evaluate_beam_cleanup.py`

**Step 1: Add the screen-cell `r0` helper**

```python
def equivalent_r0_for_cell(*, wavelength_m: float, cn2: float, cell_length_m: float) -> float:
    k = 2.0 * math.pi / wavelength_m
    return float((0.423 * (k ** 2) * cn2 * cell_length_m) ** (-3.0 / 5.0))
```

**Step 2: Replace the current fixed-window split-step loop**

```python
def propagate_split_step(...):
    schedule = build_adaptive_schedule(...)
    output = field
    for interval in schedule.intervals:
        output = propagate_same_window(
            output,
            wavelength_m=wavelength_m,
            window_m=interval.window_m,
            z_m=interval.dz_m,
        )
        for screen_index in interval.screen_indices_at_end:
            output = output * torch.exp(1j * phase_screens[screen_index]).unsqueeze(0)
        if interval.zoom_to_window_m is not None:
            output = zoom_propagate(
                output,
                wavelength_m=wavelength_m,
                source_window_m=interval.window_m,
                destination_window_m=interval.zoom_to_window_m,
                z_m=1e-9,
            )
    return output
```

Use a dedicated nonzero `regrid_distance_m` config field instead of a literal `1e-9` if the optics math requires a real propagation distance. The important design rule is: `propagate -> screen -> zoom -> next interval`.

**Step 3: Update every downstream call site**
- `training/targets.py` should use the same receiver-side propagation conventions.
- `models/d2nn.py` should keep same-window intra-layer propagation and only use zoom if a later design introduces changing windows inside the D2NN.
- `trainer.py` and `evaluate_beam_cleanup.py` should continue calling the public optics functions only, without re-implementing propagation logic.

**Step 4: Run the split-step tests**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_turbulence_pipeline.py tests/test_training_core.py -v
```

Expected:
- PASS

### Task 5: Add a vacuum-first diagnostic entrypoint

**Files:**
- Create: `src/kim2026/cli/debug_vacuum_split_step.py`
- Create: `configs/debug_vacuum_split_step.yaml`
- Modify: `src/kim2026/cli/common.py`

**Step 1: Write the failing smoke test**

```python
from pathlib import Path

from kim2026.cli.debug_vacuum_split_step import main


def test_debug_vacuum_cli_writes_preview(tmp_path: Path) -> None:
    # arrange a tiny config in tmp_path
    # call main()
    # assert that preview png exists
    ...
```

**Step 2: Implement the CLI**
- Load config.
- Build the adaptive schedule.
- Propagate a vacuum Gaussian through the full schedule only.
- Save intensity PNGs at source, each zoom plane, each screen plane, and receiver.
- Save a JSON manifest with `z_m`, `window_m`, and `event_type`.

**Step 3: Run the smoke test**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_pair_generation_smoke.py -v
```

Expected:
- PASS

### Task 6: Verify the full stack before regenerating cache

**Files:**
- Verify only: `tests/`
- Verify only: `configs/debug_vacuum_split_step.yaml`

**Step 1: Run the full test suite**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests -v
```

Expected:
- PASS

**Step 2: Run the vacuum diagnostic CLI**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m kim2026.cli.debug_vacuum_split_step --config configs/debug_vacuum_split_step.yaml
```

Expected:
- writes vacuum previews and manifest under `runs/debug_vacuum_split_step/`
- receiver-plane intensity is smooth and Gaussian-like

**Step 3: Only after vacuum passes, regenerate turbulence cache**

Run:
```bash
cd /root/dj/D2NN/worktrees/kim2026/kim2026
PYTHONPATH=src /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m kim2026.cli.generate_pairs --config configs/pilot_pair_generation_a100.yaml
```

Expected:
- no checkerboard/aliasing artifacts in vacuum previews
- cached vacuum fields remain smooth under the new adaptive split-step pipeline
