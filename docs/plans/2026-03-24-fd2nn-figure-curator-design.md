# FD2NN Figure Curator Design

**Date:** 2026-03-24

**Goal:** Create a project-specific Codex skill that curates official FD2NN codex figures by copying selected run-local figures from `kim2026/runs/02~05/.../figures` into the official figure store at `/root/dj/D2NN/kim2026/figures` using stable standard names and collision-safe rules.

## Scope

This design covers the second project-specific skill:

- Skill name: `fd2nn-figure-curator`
- Default project root: `/root/dj/D2NN/kim2026`
- Official source runs: `02~05`
- Official target figure store: `/root/dj/D2NN/kim2026/figures`

The skill is a real file-mutating curator in v1.

It does not only recommend promotion. It can perform the copy operation.

## Problem

The run-local figure folders under `02~05` contain source figures that are useful for reporting, but those figures are not yet normalized into the official figure store.

Current mismatch:

- `/root/dj/D2NN/kim2026/figures` contains curated `dashboard_*` and `report_*` figures
- run-local folders contain per-run source figures such as `fig1_epoch_curves.png`, `fig2_test_metrics.png`, `fig6_phase_masks.png`, and ROI-study figures for `05`

Without a dedicated curation layer:

- provenance becomes ambiguous
- filenames drift
- the official figure store becomes inconsistent
- downstream reporting has to keep rediscovering which run-local figure should be treated as official

## Design Choice

This skill owns figure promotion into the official store.

Why this boundary is correct:

- The first skill, `fd2nn-codex-run-interpreter`, explains experiments and recommends figure roles.
- This second skill performs deterministic curation and standard naming.
- If promotion stays advisory only, the boundary with the interpreter remains blurry.

## File Operation Policy

### Copy Policy

- `copy only`
- never move source figures
- always preserve run-local originals

### Safety Policy

- default mode is `dry-run`
- mutation requires explicit `--apply`
- overwrite is forbidden by default
- filename collisions must be reported, not silently replaced

### Naming Policy

Promoted figures must use the `codexXX_*` prefix namespace.

Examples:

- `codex02_spacing_epoch_curves.png`
- `codex03_spacing_field_full.png`
- `codex04_spacing_phase_masks.png`
- `codex05_roi_phase_metrics.png`
- `codex05_roi_support_leakage.png`

This namespace is intentionally separate from the existing:

- `dashboard_*`
- `report_*`

Those are already curated end products and must remain distinct from promoted source figures.

## Inputs

### Source Directories

- `/root/dj/D2NN/kim2026/runs/02_fd2nn_spacing-sweep_loss-old_roi-1024_codex/figures`
- `/root/dj/D2NN/kim2026/runs/03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex/figures`
- `/root/dj/D2NN/kim2026/runs/04_fd2nn_spacing-sweep_loss-shape_roi-512_codex/figures`
- `/root/dj/D2NN/kim2026/runs/05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex/figures`

### Target Directory

- `/root/dj/D2NN/kim2026/figures`

## Default Selection Policy

The default behavior is not “promote everything.”

The default behavior is “promote the core subset only.”

### Default Promotion Set

For `02~04`, promote:

- `fig1_epoch_curves.png`
- `fig2_test_metrics.png`
- `fig3_field_full_comparison.png`
- `fig6_phase_masks.png`

For `05`, promote:

- `fig1_epoch_curves.png`
- `fig2_phase_metrics.png`
- `fig3_field_comparison.png`
- `fig4_support_and_leakage.png`
- `fig5_phase_masks_raw_vs_wrapped.png`

### Optional Full Promotion

Allow an explicit `--all-figures` mode for full promotion of the run-local figure directory.

This remains opt-in because promoting all figures by default would bloat the official store and weaken its editorial value.

## Bundled Skill Contents

### `SKILL.md`

Keep the body lean:

- trigger conditions
- source and target directories
- copy-only policy
- dry-run default
- no-overwrite rule
- `codexXX_*` namespace rule

### `references/promotion-map.md`

Map each promoted source figure to:

- source path suffix
- target official filename
- report role

This file prevents naming drift.

### `references/selection-policy.md`

State:

- default promotion set
- optional full-promotion behavior
- why some figures are core and others are optional

This file prevents store bloat and makes promotion behavior explicit.

### `scripts/promote_figures.py`

Include in v1.

Required behavior:

- `--root /root/dj/D2NN/kim2026`
- `--runs 02 05`
- `--dry-run`
- `--apply`
- `--all-figures`
- collision detection
- deterministic copy plan output

No `assets/` directory is needed.

## Output Behavior

### In `--dry-run`

The script must print:

- selected source files
- target filenames
- skipped files
- collision warnings

### In `--apply`

The script must:

- copy approved source files into the official store
- preserve originals
- refuse unsafe overwrites
- report created files and skipped collisions

## Risks

### Namespace collision with existing official figures

Mitigation:
- always use `codexXX_*`
- never write into `dashboard_*` or `report_*`

### Unintended store bloat

Mitigation:
- default subset only
- `--all-figures` opt-in

### Provenance loss

Mitigation:
- preserve originals
- keep a deterministic promotion map reference

## Success Criteria

The design is successful if a fresh Codex instance can use the skill to:

1. inspect only official `02~05` run-local figure folders
2. propose a deterministic copy plan in dry-run mode
3. copy the default core subset into `/root/dj/D2NN/kim2026/figures`
4. avoid overwriting existing files silently
5. keep source figures intact
6. produce stable `codexXX_*` names that downstream report skills can reference
