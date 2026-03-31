# D2NN Lab Interpreter Design

**Date:** 2026-03-26

**Goal:** Define the first project-specific `d2nn-lab` agent as an interpreter-first skill that reads D2NN runs across tasks, builds task-wise Pareto frontiers over performance, optical complexity, and fabrication feasibility, explains why runs fail, and proposes the next experiments needed to approach a meaningful D2NN optimum.

## Scope

This design defines the first component of a broader `d2nn-lab` skill family:

- family name: `d2nn-lab`
- v1 deliverable: `d2nn-lab-interpreter`
- future siblings:
  - `d2nn-lab-operator`
  - `d2nn-lab-curator`

The interpreter is not limited to one paper replica or one task. It is intended to work across the shared D2NN architecture space inside `/root/dj/D2NN`, where the architecture class is similar but the task, input/output field definition, loss function, and hyperparameters vary.

The v1 task families are:

- classification
- imaging
- phase / wavefront restoration
- beam shaping / beam cleanup

The fabrication baseline assumed by default is:

- phase-only transmissive diffractive layers

Out of scope for v1:

- launching new training sweeps automatically
- mutating figure stores automatically
- claiming a single task-agnostic global Pareto frontier across all task families
- replacing project-specific physical judgement with one scalar metric

## Problem

The repeated failure mode in D2NN work is not just that runs are expensive. The deeper failure mode is that runs are interpreted too shallowly.

Common project pain points:

- different tasks optimize different metrics, so direct cross-run comparisons drift
- scalar gains can hide spatial or physical failure modes
- architecture choices that look numerically strong can still be dominated by optical complexity or fabrication risk
- real-space and Fourier / 4f systems require different physical interpretations
- future experiments are often chosen from intuition rather than from structured evidence about what is actually limiting the system

The question the interpreter must answer is not simply “which run is best?”

The real question is:

- what should be optimized next if the goal is to move toward a genuinely better D2NN design, not just a superficially better metric

## Design Choice

The `d2nn-lab` family should start with an interpreter, not an operator.

Why this boundary is correct:

- if the operator exists first, it will reproduce the current ambiguity faster
- if the curator exists first, it will organize figures without resolving what they actually mean
- if the interpreter exists first, both future automation and reporting can share one grounded explanation layer

The interpreter must therefore do three things before any future orchestration:

1. normalize heterogeneous runs into a shared registry
2. compute task-wise Pareto frontiers using the agreed tradeoff axes
3. explain failure causes in both optimization language and optical / diffraction language

## High-Level Architecture

The family structure is:

- `d2nn-lab`
  - conceptual parent and future routing entry point
- `d2nn-lab-interpreter`
  - v1 implementation target
- `d2nn-lab-operator`
  - deferred
- `d2nn-lab-curator`
  - deferred

The v1 interpreter uses a task-family adapter pattern.

Each adapter reads a run family and emits a canonical run row with a shared schema. The interpreter then works only on canonical rows rather than on raw task-specific files.

This avoids two recurring errors:

- task-specific metrics being compared without normalization
- physical causes being inferred from filenames and intuition instead of structured inputs

## Pareto Objective

The objective is not a single “best model” scalar. It is a task-wise Pareto tradeoff across:

- performance
- optical complexity
- fabrication feasibility

### Performance Axis

Performance is task-specific and defined by the task adapter.

Examples:

- classification: accuracy, error rate, task-specific detection score
- imaging: PSNR, SSIM, reconstruction loss
- phase / wavefront restoration: complex overlap, phase RMSE, amplitude RMSE
- beam shaping / cleanup: overlap, Strehl, support leakage, field error

### Optical Complexity Axis

This is a cost axis. Smaller is better.

Candidate contributors:

- number of diffractive layers
- total axial depth
- effective aperture usage
- phase dynamic range
- feature density

### Fabrication Feasibility Axis

This is also a cost axis. Smaller is better.

Candidate contributors:

- minimum feature size risk
- pixel pitch risk
- alignment tolerance risk
- layer spacing tolerance risk
- wavelength sensitivity risk
- thickness or phase-depth realizability risk

## Task-Wise Frontier Policy

The interpreter must compute Pareto frontiers separately per task family.

This is intentional.

Why:

- classification and wavefront restoration do not share a stable raw performance axis
- imaging and beam cleanup often have different artifact structures and different fabrication sensitivities
- a combined frontier too early would create false comparisons and produce noisy “best” recommendations

The v1 output therefore follows this sequence:

1. task-wise Pareto registry
2. elimination-cause analysis
3. physical interpretation
4. next experiment proposal

Only after that may the interpreter produce a cross-task common-bottleneck summary.

## Canonical Registry

Each run row in the canonical registry must contain at least these fields:

- `task_family`
- `task_name`
- `architecture_family`
- `propagation_mode`
- `optical_params`
- `fabrication_params`
- `train_objective`
- `primary_metrics`
- `complexity_metrics`
- `fabrication_risk_metrics`
- `artifact_paths`
- `derived_optical_scales`
- `sampling_diagnostics`
- `diffraction_diagnostics`
- `spectral_or_spatial_bottleneck`

This schema is the main design boundary of the interpreter. Everything downstream should consume this row format rather than raw project files.

## Physics Modes

The interpreter must explicitly split physical interpretation into two modes:

- real-space D2NN
- Fourier / 4f D2NN

### Real-Space D2NN

Primary interpretation concerns:

- Fresnel propagation regime
- layer-spacing sufficiency
- detector-distance sufficiency
- diffraction mixing strength
- aperture and window interaction
- sampling adequacy in the propagated field

Typical conclusions the interpreter should be able to support:

- spacing is too short, so diffraction mixing is weak
- distance is too long, so blur grows without useful transformation
- pitch and window undersample local phase gradients
- scalar gains exist, but the field remains close to identity propagation

### Fourier / 4f D2NN

Primary interpretation concerns:

- focal-length scaling
- numerical aperture limits
- Fourier-plane spectral support
- passband clipping
- spectral resolution versus task bandwidth

Typical conclusions the interpreter should be able to support:

- task-relevant frequencies are clipped by NA
- Fourier-plane sampling is too coarse
- the passband is adequate but optimization is limiting

## Root-Cause Taxonomy

The interpreter must classify non-Pareto runs with an explicit bottleneck taxonomy.

The minimum v1 taxonomy is:

- `objective_mismatch`
- `capacity_limited`
- `propagation_geometry_limited`
- `sampling_limited`
- `fabrication_constrained`
- `optimization_limited`
- `metric_illusion`

### Meaning

- `objective_mismatch`
  - training loss does not faithfully represent the actual task goal
- `capacity_limited`
  - layer count or phase freedom is insufficient
- `propagation_geometry_limited`
  - spacing, detector distance, relay setup, or propagation regime is the dominant limitation
- `sampling_limited`
  - pitch, grid size, or window selection limits usable field or spectrum representation
- `fabrication_constrained`
  - numerically good solutions imply unrealistic device requirements
- `optimization_limited`
  - the run likely under-converged due to training dynamics
- `metric_illusion`
  - headline scalar metric improves while spatial, phase, or field evidence shows little real task improvement

The taxonomy is not just for labeling. It must drive the recommendation stage.

## Input File Contract

The interpreter should use a minimum evidence policy.

### Required Evidence

- configuration file
- final metrics file
- training history or equivalent convergence trace
- at least one figure or reconstructable raw artifact

### Preferred Sources

- `config.yaml` or equivalent
- `evaluation.json`, `test_metrics.json`, `sweep_summary.json`, `study_summary.json`
- `history.json` or checkpoint-embedded history
- run-local `figures/`
- raw field exports when present

### Failure Policy

- missing config: exclude from registry
- missing final metrics: include as incomplete evidence only
- missing history: disable confident optimization diagnosis
- missing figures or raw artifacts: disable strong `metric_illusion` claims

## Output Contract

The final interpreter output should always be organized in this order.

### 1. Pareto Registry

Per task family:

- frontier runs
- dominated runs
- core metrics and cost axes
- domination relationships when inferable

### 2. Elimination-Cause Analysis

Per dominated or weak run:

- why it failed to enter the frontier
- which axis it lost on
- which taxonomy label best explains the failure

### 3. Physical Interpretation

Per task family and per notable run:

- real-space or Fourier-space interpretation
- diffraction, geometry, spectral support, and sampling explanation
- whether the limitation is physically structural or optimization-related

### 4. Next Experiment Proposal

Per task family:

- what to vary next
- what to hold fixed
- what physical hypothesis is being tested
- what result would falsify the current explanation

## Materials That Improve v1 But Are Not Hard Blockers

The interpreter can start without these, but should request them when available:

- fabrication constraint table
  - minimum feature size
  - alignment tolerance
  - achievable phase depth or thickness range
- task-specific primary metric priority notes
- project-specific mapping of which configs and summaries are authoritative
- target-field definition notes
  - vacuum propagated field
  - analytic Gaussian
  - classifier detector pattern

## Risks

### Registry Drift

Risk:

- adapters across projects may normalize fields inconsistently

Mitigation:

- make the canonical registry schema explicit and reference-driven

### False Cross-Task Comparisons

Risk:

- users may read a global leaderboard where none should exist

Mitigation:

- keep frontier computation task-wise only

### Physics Overclaim

Risk:

- the interpreter may state strong diffraction conclusions from weak evidence

Mitigation:

- require the physical mode to be explicit
- separate evidence-backed diagnosis from plausible hypothesis

### Metric Illusion Under-Detection

Risk:

- strong scalar metrics may still conceal spatial failure

Mitigation:

- require figures or raw artifacts before confidently clearing a run of `metric_illusion`

## Success Criteria

The design is successful if a fresh Codex instance using the v1 interpreter can:

1. ingest D2NN runs from multiple task families into one canonical registry
2. compute task-wise Pareto frontiers over performance, complexity, and fabrication feasibility
3. explain why non-frontier runs fail using explicit bottleneck labels
4. distinguish real-space and Fourier / 4f physical interpretations
5. identify when metrics improve without meaningful field-level improvement
6. recommend next experiments as falsifiable physical hypotheses rather than generic tuning advice
