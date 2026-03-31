---
name: fd2nn-codex-run-interpreter
description: Interpret the official `kim2026` FD2NN codex experiments under `/root/dj/D2NN/kim2026/runs` for runs `02~05`, reconstruct each run's sweep axis and fixed conditions, map `kim2026/figures` to report claims, and produce Korean-first run registries, figure registries, promotion plans, and narrative blocks. Use when explaining these codex runs, comparing them, or preparing report-ready interpretations from their metrics and figures.
---

# FD2NN Codex Run Interpreter

Use this skill as the interpretation layer between raw `kim2026` codex runs and downstream report-writing or slide-writing work.

## Defaults

- Work from `/root/dj/D2NN/kim2026` unless the user explicitly provides another root.
- Treat only these runs as official v1 inputs:
  - `02_fd2nn_spacing-sweep_loss-old_roi-1024_codex`
  - `03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex`
  - `04_fd2nn_spacing-sweep_loss-shape_roi-512_codex`
  - `05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex`
- Treat `/root/dj/D2NN/kim2026/figures` as the official figure store.

Ignore `01_*` and `*_claude` runs unless the user explicitly asks for side-by-side comparison.

## Workflow

1. Read [references/run-schema.md](references/run-schema.md) before interpreting the run set. This file locks the project-specific meaning of `02~05`.
2. Read [references/figure-policy.md](references/figure-policy.md) before assigning report roles to figures.
3. If the user wants structured run metadata, run:

```bash
python3 /root/dj/D2NN/skills/fd2nn-codex-run-interpreter/scripts/build_run_registry.py --root /root/dj/D2NN/kim2026
```

4. Inspect run-local `history.json`, `test_metrics.json`, `sweep_summary.json`, and `study_summary.json` only as needed to justify the interpretation.
5. Inspect run-local `figures/` folders only for provenance checks. In v1, recommend figure promotion but do not copy files automatically.

## Required Output Shape

When this skill is used for analysis, structure the answer with these sections:

- `run registry`
- `figure registry`
- `promotion plan`
- `narrative blocks`

### Run Registry

For each official run, include:

- sweep axis
- fixed conditions
- compared conditions
- key metrics
- one-sentence conclusion
- source paths

### Figure Registry

For each official figure under `kim2026/figures`, include:

- figure path
- source run or run family
- analytical role
- recommended report use
- Korean caption draft
- optional English caption draft

### Promotion Plan

When a necessary figure exists only in a run-local `figures/` directory, state:

- source file
- suggested official destination name
- why it should be promoted
- which report claim it supports

In v1 this is advisory only. Do not mutate the official figure store automatically.

### Narrative Blocks

Write Korean-first short blocks for:

- experiment purpose
- comparison axis
- key observation
- interpretation caution

Add English only if the user asks or if downstream reporting clearly needs it.

## Explicit Exclusions

This skill does not:

- generate PDFs
- generate PPTX files
- auto-promote figures into `kim2026/figures`
- treat non-official runs as first-class evidence by default
