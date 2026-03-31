# D2NN Lab Interpreter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first `d2nn-lab` project-specific skill as an interpreter that normalizes D2NN runs across tasks, computes task-wise Pareto frontiers, classifies bottlenecks, and proposes next experiments.

**Architecture:** Create a repo-local `skills/d2nn-lab-interpreter/` skill with reference-driven instructions plus deterministic helper scripts. The scripts should ingest heterogeneous run directories into a canonical registry, compute task-wise Pareto and elimination causes, then emit a structured interpretation report that cleanly separates registry, elimination, physics, and experiment recommendations.

**Tech Stack:** Markdown skill files, repo-local Python scripts, JSON I/O, pytest, importlib-based script tests

---

### Task 1: Scaffold The Skill Package

**Files:**
- Create: `skills/d2nn-lab-interpreter/SKILL.md`
- Create: `skills/d2nn-lab-interpreter/references/scope-and-task-families.md`
- Create: `skills/d2nn-lab-interpreter/references/canonical-registry-schema.md`
- Create: `skills/d2nn-lab-interpreter/references/root-cause-taxonomy.md`
- Create: `skills/d2nn-lab-interpreter/references/physics-modes.md`

**Step 1: Write the failing test**

Create `tests/test_d2nn_lab_interpreter_skill_files.py` with checks like:

```python
from pathlib import Path


def test_d2nn_lab_interpreter_skill_files_exist() -> None:
    root = Path("skills/d2nn-lab-interpreter")
    assert (root / "SKILL.md").exists()
    assert (root / "references" / "canonical-registry-schema.md").exists()
    assert (root / "references" / "root-cause-taxonomy.md").exists()
    assert (root / "references" / "physics-modes.md").exists()
```

**Step 2: Run test to verify it fails**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_skill_files.py -v`

Expected: FAIL because the `skills/d2nn-lab-interpreter/` files do not exist yet.

**Step 3: Write minimal implementation**

Create the skill scaffold with:

- `SKILL.md`
  - trigger conditions
  - supported task families
  - required output order
  - explicit real-space versus Fourier interpretation split
- `references/scope-and-task-families.md`
  - v1 task families
  - out-of-scope statements
- `references/canonical-registry-schema.md`
  - required row fields
- `references/root-cause-taxonomy.md`
  - bottleneck labels and definitions
- `references/physics-modes.md`
  - real-space and Fourier / 4f diagnostic rules

**Step 4: Run test to verify it passes**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_skill_files.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_d2nn_lab_interpreter_skill_files.py skills/d2nn-lab-interpreter
git commit -m "feat: scaffold d2nn lab interpreter skill"
```

### Task 2: Build Canonical Registry Ingestion

**Files:**
- Create: `skills/d2nn-lab-interpreter/scripts/build_registry.py`
- Create: `tests/test_d2nn_lab_interpreter_registry.py`

**Step 1: Write the failing test**

Add tests that create temporary run directories with:

- `config.yaml`
- `evaluation.json`
- `history.json`
- optional `figures/`

and assert the script normalizes them into rows like:

```python
assert row["task_family"] == "beam_cleanup"
assert row["propagation_mode"] == "real_space"
assert "primary_metrics" in row
assert "complexity_metrics" in row
assert "fabrication_risk_metrics" in row
```

**Step 2: Run test to verify it fails**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_registry.py -v`

Expected: FAIL because `build_registry.py` does not exist and registry rows cannot be constructed.

**Step 3: Write minimal implementation**

Implement `build_registry.py` with:

- a `discover_runs(root: Path)` function
- task-family adapters selected from config metadata or path conventions
- a canonical row builder returning a JSON-serializable dict
- missing-evidence flags
- output command like:

```bash
python skills/d2nn-lab-interpreter/scripts/build_registry.py --root /root/dj/D2NN --output /tmp/d2nn_registry.json
```

**Step 4: Run test to verify it passes**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_registry.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_d2nn_lab_interpreter_registry.py skills/d2nn-lab-interpreter/scripts/build_registry.py
git commit -m "feat: add d2nn lab canonical registry builder"
```

### Task 3: Add Pareto And Elimination Analysis

**Files:**
- Create: `skills/d2nn-lab-interpreter/scripts/analyze_registry.py`
- Modify: `tests/test_d2nn_lab_interpreter_registry.py`

**Step 1: Write the failing test**

Add tests for:

- task-wise Pareto classification
- domination relationships
- elimination reasons

Example expectation:

```python
assert result["task_families"]["beam_cleanup"]["frontier"][0]["run_id"] == "run_a"
assert result["task_families"]["beam_cleanup"]["dominated"][0]["elimination"]["label"] == "sampling_limited"
```

**Step 2: Run test to verify it fails**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_registry.py -v`

Expected: FAIL because Pareto and elimination analysis are not implemented.

**Step 3: Write minimal implementation**

Implement:

- per-task-family grouping
- Pareto comparison over:
  - normalized performance score
  - complexity cost
  - fabrication risk cost
- elimination records containing:
  - `label`
  - `dominated_by`
  - `why`

Keep the first version deterministic and evidence-based. Do not add heuristic prose generation yet.

**Step 4: Run test to verify it passes**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_registry.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_d2nn_lab_interpreter_registry.py skills/d2nn-lab-interpreter/scripts/analyze_registry.py
git commit -m "feat: add pareto and elimination analysis for d2nn lab"
```

### Task 4: Add Physics And Diffraction Diagnostics

**Files:**
- Modify: `skills/d2nn-lab-interpreter/scripts/build_registry.py`
- Modify: `skills/d2nn-lab-interpreter/scripts/analyze_registry.py`
- Create: `tests/test_d2nn_lab_interpreter_physics.py`

**Step 1: Write the failing test**

Add tests that create:

- a real-space-style config with `layer_spacing_m`, `detector_distance_m`, `receiver_window_m`, `lambda_m`
- a Fourier-style config with `f1_m`, `f2_m`, `na1`, `na2`

and assert:

```python
assert row["propagation_mode"] == "real_space"
assert "diffraction_mixing_strength" in row["diffraction_diagnostics"]
assert "spectral_support_limit" in row["sampling_diagnostics"]
```

**Step 2: Run test to verify it fails**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_physics.py -v`

Expected: FAIL because physics diagnostics are missing.

**Step 3: Write minimal implementation**

Extend registry and analysis with:

- explicit `propagation_mode`
- derived optical scales
- diffraction diagnostics for real-space runs
- spectral support and clipping diagnostics for Fourier / 4f runs
- physical explanations tied to elimination labels

Keep the first implementation conservative:

- mark unsupported claims as hypotheses
- avoid pretending to know hidden fabrication constants

**Step 4: Run test to verify it passes**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_physics.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_d2nn_lab_interpreter_physics.py skills/d2nn-lab-interpreter/scripts/build_registry.py skills/d2nn-lab-interpreter/scripts/analyze_registry.py
git commit -m "feat: add physics diagnostics to d2nn lab interpreter"
```

### Task 5: Emit Structured Interpreter Reports

**Files:**
- Modify: `skills/d2nn-lab-interpreter/SKILL.md`
- Modify: `skills/d2nn-lab-interpreter/scripts/analyze_registry.py`
- Create: `tests/test_d2nn_lab_interpreter_report.py`

**Step 1: Write the failing test**

Add a test that runs the analysis script on fixture rows and asserts the emitted report contains sections in the agreed order:

```python
assert sections == [
    "pareto_registry",
    "elimination_analysis",
    "physical_interpretation",
    "next_experiment_proposals",
]
```

**Step 2: Run test to verify it fails**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_report.py -v`

Expected: FAIL because no structured report output exists yet.

**Step 3: Write minimal implementation**

Update the analysis script so it can output:

- registry section
- elimination analysis section
- physical interpretation section
- next experiment proposals section

Provide both:

- JSON output for deterministic downstream tooling
- Markdown output for direct human use

**Step 4: Run test to verify it passes**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_report.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_d2nn_lab_interpreter_report.py skills/d2nn-lab-interpreter/SKILL.md skills/d2nn-lab-interpreter/scripts/analyze_registry.py
git commit -m "feat: add structured reporting to d2nn lab interpreter"
```

### Task 6: Verify On Real Project Roots

**Files:**
- Modify: `skills/d2nn-lab-interpreter/references/scope-and-task-families.md`
- Modify: `skills/d2nn-lab-interpreter/references/physics-modes.md`
- Modify: `skills/d2nn-lab-interpreter/references/canonical-registry-schema.md`

**Step 1: Write the failing test**

Add an integration-style test or scripted smoke check that runs the registry builder on a limited real-project fixture list and asserts non-empty outputs for at least two task families.

**Step 2: Run test to verify it fails**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_* -v`

Expected: FAIL until the real-root assumptions and report fields line up.

**Step 3: Write minimal implementation**

Refine references and defaults based on real repo findings:

- authoritative config patterns
- authoritative metric files
- known unsupported run families
- known caution labels

Do not expand scope beyond the agreed v1 task families.

**Step 4: Run test to verify it passes**

Run: `cd /root/dj/D2NN && PYTHONPATH=. /root/dj/D2NN/miniconda3/envs/d2nn/bin/python -m pytest tests/test_d2nn_lab_interpreter_* -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_d2nn_lab_interpreter_* skills/d2nn-lab-interpreter/references
git commit -m "feat: validate d2nn lab interpreter on repo fixtures"
```
