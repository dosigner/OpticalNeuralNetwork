# Fig. 3 Paper-Style Regeneration Design

**Date:** 2026-03-16

**Goal:** Regenerate Fig. 3 so it visually matches the paper's grouped-bar presentation, and add caption/analysis text that explains how to interpret the figure.

## Scope

- Replace the current line-plus-errorbar Fig. 3 renderer with a paper-style grouped bar chart.
- Keep the numerical evaluation pipeline unchanged: reconstruct grating targets, estimate output periods, aggregate over known and blind diffusers.
- Regenerate the PNG and raw `.npy` output for Fig. 3.
- Add a concise caption recommendation and an analysis note describing how to read the figure.

## Visual Requirements

- Two panels:
  - `(a)` All-Optical Imaging Through Last `n` Diffusers in Training
  - `(b)` All-Optical Imaging Through 20 New Diffusers
- Four colored bars per target period for `n = 1, 10, 15, 20`.
- Green dashed horizontal segment per target period showing the true period.
- Axis labels:
  - x: `Resolution Test Target Period, mm`
  - y: `Measured Grating Period, mm`
- Axis ranges and ticks should match the paper-style layout rather than autoscaled plotting.
- Legend should match the plotted semantics: `n` values plus `True Period`.

## Interpretation Requirements

- Explicitly distinguish:
  - `Resolution Test Target Period`: the known physical period of the input resolution test target.
  - `Measured Grating Period`: the period estimated from the reconstructed output image after propagation through diffuser + D2NN.
- Explain that closeness between the colored bars and the green true-period marker indicates accurate reconstruction and good generalization.
- Explain that panel `(a)` measures performance on the last training diffusers, while panel `(b)` measures performance on unseen random diffusers.

## Files

- Modify: `src/luo2022_d2nn/figures/fig3_period_sweep.py`
- Add: `tests/test_fig3_period_sweep.py`
- Add: `analysis/fig3_interpretation.ko.md`
- Regenerate: `figures/fig3_period_sweep.png`, `figures/fig3_period_sweep.npy`

## Non-Goals

- Do not change the grating-generation or period-estimation algorithms.
- Do not alter checkpoint selection or evaluation seeds unless required for reproducibility.
