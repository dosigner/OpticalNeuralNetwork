# Figure Policy

Use this reference to map figures to report roles and to keep figure provenance stable across interpretations.

## Official Figure Store

The official report-facing figure directory is:

- `/root/dj/D2NN/kim2026/figures`

Current official files:

- `dashboard_field_intensity.png`
- `dashboard_field_phase.png`
- `dashboard_field_summary.png`
- `dashboard_phase_layers.png`
- `dashboard_sweep_results.png`
- `report_convergence.png`
- `report_field_intensity.png`
- `report_field_phase.png`
- `report_field_summary.png`

## Role Labels

Assign each figure one primary role.

### Performance Comparison

Use for:

- sweep-to-sweep performance ranking
- metric comparisons across conditions
- condensed dashboard summaries

Typical figures:

- `dashboard_sweep_results.png`
- `report_field_summary.png`

### Physical Interpretation

Use for:

- intensity structure
- phase structure
- input vs target vs prediction comparison

Typical figures:

- `dashboard_field_intensity.png`
- `dashboard_field_phase.png`
- `report_field_intensity.png`
- `report_field_phase.png`

### Convergence

Use for:

- epoch curves
- training stabilization
- loss or metric evolution

Typical figures:

- `report_convergence.png`

### Phase Mask Inspection

Use for:

- learned layer phases
- raw vs wrapped phase behavior
- support or leakage discussion

Typical figures:

- `dashboard_phase_layers.png`
- run-local `05/.../figures/fig4_support_and_leakage.png`
- run-local `05/.../figures/fig5_phase_masks_raw_vs_wrapped.png`

## Dashboard vs Report Usage

- `dashboard_*` figures are overview-oriented. Prefer them when the user asks for comparison at a glance or executive-level summaries.
- `report_*` figures are explanation-oriented. Prefer them when the user asks for page-by-page or figure-by-figure interpretation.

## Run-Local Figure Policy for `05`

The run-local figure directory:

- `/root/dj/D2NN/kim2026/runs/05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex/figures`

contains specialized ROI-study figures:

- `fig1_epoch_curves.png`
- `fig2_phase_metrics.png`
- `fig3_field_comparison.png`
- `fig4_support_and_leakage.png`
- `fig5_phase_masks_raw_vs_wrapped.png`

These are not automatically official in v1. When they are needed in a report, recommend promotion by stating:

- the source file
- the target official filename
- the report claim it supports

## Caption Policy

Write captions with this order:

1. what is being compared
2. which run or condition family the figure represents
3. what the reader should notice

Default caption language is Korean. Add English only when the downstream report clearly needs it.
