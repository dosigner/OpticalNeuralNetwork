# Pupil Canonical Dataset + Alias-Safe Reducer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the mixed-coordinate `tel15cm_n1024_br75` dataset path with a canonical 15 cm pupil dataset, generate reduced training inputs on the fly via an alias-safe ideal reducer, and block training until reducer validation approves the ideal path.

**Architecture:** Keep the canonical dataset as the single source of truth in pupil coordinates and move reducer logic into explicit optics/data modules. The loader becomes schema-aware, returning either the stored pupil plane or an on-the-fly reduced plane, while validation and training gates are driven by a reducer summary artifact.

**Tech Stack:** Python, PyTorch, NumPy, existing `kim2026` optics/scaled Fresnel utilities, pytest.

---

### Task 1: Canonical Schema + Loader Contract

**Files:**
- Create: `kim2026/src/kim2026/data/canonical_pupil.py`
- Modify: `kim2026/src/kim2026/data/dataset.py`
- Modify: `kim2026/src/kim2026/config/schema.py`
- Test: `kim2026/tests/test_canonical_pupil_dataset.py`
- Test: `kim2026/tests/test_config_schema.py`

**Step 1: Write failing schema and loader tests**

Run:

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest tests/test_canonical_pupil_dataset.py tests/test_config_schema.py -q
```

Expected: missing canonical dataset helpers / unsupported `plane_selector`.

**Step 2: Implement canonical NPZ IO**

Add canonical writer/reader helpers, required metadata validation, deterministic `4000/500/500` manifest construction, and fixed dataset path constants.

**Step 3: Implement schema-aware dataset loading**

Extend `CachedFieldDataset` with `plane_selector={"pupil","reduced_ideal"}` while preserving `u_vacuum`, `u_turb`, `metadata` return keys. Keep legacy pair-schema support intact.

**Step 4: Validate config support**

Accept `data.plane_selector` and reducer-validation gate config, reject unknown selectors, and keep old configs valid by default.

**Step 5: Re-run targeted tests**

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest tests/test_canonical_pupil_dataset.py tests/test_config_schema.py -q
```

Expected: green.

### Task 2: Alias-Safe Ideal Reducer + Full Reference Reducer

**Files:**
- Modify: `kim2026/src/kim2026/optics/beam_reducer.py`
- Modify: `kim2026/src/kim2026/optics/__init__.py`
- Test: `kim2026/tests/test_beam_reducer_alias_safe.py`

**Step 1: Write failing reducer tests**

Run:

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest tests/test_beam_reducer_alias_safe.py -q
```

Expected: missing reducer plane dataclass / missing alias-safe reducer / missing reference solver.

**Step 2: Implement the ideal reducer**

Create an explicit plane-geometry interface, deprecate the legacy bilinear path, and implement the energy-preserving ideal reducer so the exact `153.6 mm -> 2.048 mm` mapping is treated as an alias-safe coordinate remap rather than interpolation.

**Step 3: Implement the reference relay**

Add the thin-lens `f1=75 mm`, `f2=1 mm` relay using scaled Fresnel unequal-window propagation and the same output grid as the ideal reducer.

**Step 4: Re-run reducer tests**

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest tests/test_beam_reducer_alias_safe.py -q
```

Expected: green.

### Task 3: Validation Cache + Training Gate

**Files:**
- Modify: `kim2026/src/kim2026/training/trainer.py`
- Create: `kim2026/scripts/validate_beam_reducer.py`
- Test: `kim2026/tests/test_training_gate.py`

**Step 1: Write failing gate tests**

Run:

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest tests/test_training_gate.py -q
```

Expected: training does not yet block on missing/failed reducer validation summaries.

**Step 2: Implement reducer validation summary helpers**

Define summary format, per-sample metrics, aggregate metrics, deterministic `64 val + 64 test` selection, and recommendation values `ideal_ok` / `promote_full`.

**Step 3: Enforce the training gate**

Block training when:
- summary is missing and `data.reducer_validation.required=true`
- summary recommends promotion while `plane_selector="reduced_ideal"`

Allow training only when the summary passes for the selected plane.

**Step 4: Re-run gate tests**

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest tests/test_training_gate.py -q
```

Expected: green.

### Task 4: Dataset Generation + Focal Pipeline Migration

**Files:**
- Create: `kim2026/scripts/generate_canonical_pupil_dataset.py`
- Modify: `kim2026/src/kim2026/eval/focal_utils.py`
- Modify: `kim2026/autoresearch/d2nn_focal_pib_sweep.py`
- Modify: `kim2026/autoresearch/d2nn_focal_strehl_retrain.py`
- Modify: `kim2026/autoresearch/configs/focal_pib_sweep.yaml`
- Modify: `kim2026/scripts/eval_focal_pib_only.py`
- Modify: `kim2026/scripts/eval_bucket_radius_sweep.py`
- Modify: `kim2026/scripts/visualize_focal_pib_report.py`
- Modify: `kim2026/scripts/generate_focal_paper_figures.py`
- Test: `kim2026/tests/test_focal_pib_sweep_sanity.py`

**Step 1: Add generator and validation entrypoints**

Implement the canonical pupil generator and reducer-validation CLI without executing the full 5000-sample or 128-sample production runs inside unit tests.

**Step 2: Migrate focal scripts**

Switch focal training/evaluation utilities to the new dataset root and `plane_selector="reduced_ideal"` while preserving legacy metrics output keys.

**Step 3: Update focal sanity tests**

Extend the focal sweep sanity tests to assert the new dataset root and plane selector behavior.

**Step 4: Run focused regression**

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src pytest \
  tests/test_canonical_pupil_dataset.py \
  tests/test_beam_reducer_alias_safe.py \
  tests/test_training_gate.py \
  tests/test_focal_pib_sweep_sanity.py \
  tests/test_training_core.py \
  tests/test_config_schema.py -q
```

Expected: green.

### Task 5: Manual Verification Artifacts

**Files:**
- Expected output: `kim2026/data/kim2026/1km_cn2_5e-14_tel15cm_pupil1024_v1`
- Expected output: `kim2026/data/kim2026/1km_cn2_5e-14_tel15cm_pupil1024_v1/reducer_val_cache`

**Step 1: Dry-run the generator**

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/generate_canonical_pupil_dataset.py --dry-run
```

Expected: prints the canonical path, manifest counts, and storage estimate.

**Step 2: Dry-run reducer validation**

```bash
cd /root/dj/D2NN/kim2026 && PYTHONPATH=src python scripts/validate_beam_reducer.py --dry-run
```

Expected: prints deterministic `64+64` subset selection and summary output locations.

**Step 3: Record remaining gaps**

If the full 5000-sample generation or 128-sample validation is not executed in-session, report that explicitly and leave the codepaths runnable.
