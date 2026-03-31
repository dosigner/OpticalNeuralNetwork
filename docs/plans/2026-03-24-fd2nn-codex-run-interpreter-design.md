# FD2NN Codex Run Interpreter Design

**Date:** 2026-03-24

**Goal:** Create a project-specific Codex skill that reconstructs the meaning of the official `kim2026` codex experiments, maps figures to experimental claims, and produces report-ready interpretation blocks without generating PDFs or PPTX files.

## Scope

This design covers the first project-specific skill:

- Skill name: `fd2nn-codex-run-interpreter`
- Install location: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter`
- Default project root inside the skill: `/root/dj/D2NN/kim2026`

The skill interprets only the official codex run set:

- `02_fd2nn_spacing-sweep_loss-old_roi-1024_codex`
- `03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex`
- `04_fd2nn_spacing-sweep_loss-shape_roi-512_codex`
- `05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex`

The skill treats `/root/dj/D2NN/kim2026/figures` as the official figure store for explanation and report composition.

Out of scope for v1:

- PDF generation
- PPTX generation
- automatic figure promotion from run-local figure folders
- interpreting `01_*` or `*_claude` runs as first-class inputs

## Problem

The repeated cost in this project is not plotting figures. The repeated cost is reconstructing the experimental meaning of each run:

- which sweep axis is active
- which conditions are fixed
- what loss label actually means in project context
- which metrics matter for each experiment
- which figure should be used to support which narrative claim

Without project-specific guidance, a general-purpose model will repeatedly drift on these points, especially when old naming schemes and mixed run families coexist under `kim2026/runs`.

## Design Choice

The skill is intentionally not an all-in-one report generator.

It is an interpretation layer positioned between raw runs and later document-composition skills.

Why this boundary is correct:

- If interpretation and report layout are combined, the trigger becomes too broad.
- If the skill reads every run family by default, old and unofficial experiments bleed into the official narrative.
- If figure storage is not normalized, report-generation stages must rediscover provenance every time.

## Inputs

### Primary Inputs

- project root: `/root/dj/D2NN/kim2026`
- official runs under `/root/dj/D2NN/kim2026/runs`
- official figures under `/root/dj/D2NN/kim2026/figures`

### Allowed Run Files

The skill may inspect:

- `history.json`
- `test_metrics.json`
- `sweep_summary.json`
- `study_summary.json`
- run-local `figures/` directories when explanation requires provenance checks

### Default Policy

- `02~05` are the only official experiments for v1 interpretation.
- `kim2026/figures` is the only official figure store for downstream explanation.
- run-local figure folders are inspected only as supporting provenance, not as final report inputs.

## Outputs

The skill must produce four structured artifacts in its answer.

### 1. Run Registry

Per official run:

- sweep axis
- fixed conditions
- compared conditions
- key metrics
- one-sentence conclusion
- referenced source paths

### 2. Figure Registry

Per official figure:

- figure path
- source run or run family
- analytical role
- report-use recommendation
- short caption draft in Korean
- optional English caption draft

### 3. Promotion Plan

When a run-local figure is required but not yet represented in `kim2026/figures`, the skill must identify:

- source file
- target official figure name
- why the figure deserves promotion
- what report claim it supports

In v1 this is analysis only. No automatic copying is performed.

### 4. Narrative Blocks

Short report-ready paragraphs in `ko` by default, with optional `en` variants:

- experiment purpose
- comparison axis
- key observation
- interpretation caution

## Bundled Skill Contents

### `SKILL.md`

Lean workflow instructions only:

- when to trigger
- official run set
- official figure store
- required output structure
- explicit exclusions

### `references/run-schema.md`

Project-specific meaning of runs `02~05`:

- official name
- sweep axis
- loss family
- ROI condition
- fixed spacing or swept spacing
- interpretation notes

This is reference material because it is stable project knowledge that should not be re-derived each time.

### `references/figure-policy.md`

Official figure roles for `kim2026/figures` and run-local study figures:

- performance comparison
- physical interpretation
- convergence
- phase-mask inspection
- dashboard vs report figure usage

This prevents figure-role drift across future report tasks.

### `scripts/build_run_registry.py`

Deterministic helper to read the official runs and emit a structured registry draft from:

- summary files
- metric files
- history files

This is included in v1 because parsing these files is repetitive and deterministic.

### Deferred to v2

- `scripts/promote_figures.py`

This is intentionally postponed because figure-promotion policy is still evolving. The skill should recommend promotion in v1, not mutate the figure store automatically.

## Language Policy

- Default narrative language: Korean
- Optional narrative language: English
- Korean output is primary because current report-development discussion is in Korean and the immediate downstream use is Korean-first explanation

## Risks

### Mixed run families in one directory

Risk:
- A future invocation may accidentally mix official codex runs with `*_claude` experiments.

Mitigation:
- Encode the official run list explicitly in both `SKILL.md` and `references/run-schema.md`.

### Figure provenance ambiguity

Risk:
- A figure may exist only in a run-local folder while downstream reporting expects `kim2026/figures`.

Mitigation:
- Require a `promotion plan` section in the skill output.

### Metric over-interpretation

Risk:
- Different runs expose different summaries and metrics, and a model may overstate comparability.

Mitigation:
- Require each run registry entry to state both compared conditions and fixed conditions explicitly.

## Success Criteria

The design is successful if a fresh Codex instance can use the skill to:

1. read only `02~05` as the official experiment set
2. explain what each run is actually sweeping
3. map official figures to report roles
4. produce Korean-first explanation blocks without regenerating the experimental context from scratch
5. identify missing figure promotions without mutating the figure store automatically
