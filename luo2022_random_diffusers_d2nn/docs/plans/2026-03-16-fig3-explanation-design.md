# Fig. 3 Explanation Figure Design

**Date:** 2026-03-16

**Goal:** Create a companion explanation figure that makes the meanings of `Resolution Test Target Period` and `Measured Grating Period` visually obvious using three real examples, while also showing how the same targets look after diffuser-only propagation before D2NN reconstruction.

## Scope

- Build a new explanation figure with three rows for `7.2 mm`, `10.8 mm`, and `12.0 mm`.
- Each row will show:
  - the input resolution target,
  - the diffuser-only free-space propagation result at the output plane,
  - the actual D2NN reconstruction,
  - the 1D averaged profile used for period estimation.
- Add a short caption at the bottom of the figure that explains how the x-axis and y-axis of Fig. 3 should be interpreted.

## Layout

- `3 x 4` grid:
  - Column 1: `Input Resolution Target`
  - Column 2: `Propagation Through Diffuser`
  - Column 3: `D2NN Reconstruction`
  - Column 4: `Averaged Profile and Period Readout`
- Rows:
- `7.2 mm`
- `10.8 mm`
- `12.0 mm`
- The profile panel should mark:
  - the true target period,
  - the measured period obtained from the reconstruction profile.
- The profile panel remains tied to the D2NN reconstruction only so the explanation stays focused on how Fig. 3 reads out the measured period.

## Data Source

- Use a real trained checkpoint instead of synthetic examples.
- Default to the 4-layer `n=20` checkpoint for the clearest explanation figure.
- Use one deterministic diffuser seed so the explanation output is reproducible.
- Reuse the same diffuser instance within each row for both the diffuser-only baseline and the D2NN reconstruction so the comparison is direct.

## Interpretation Requirements

- Make it obvious that:
  - `Resolution Test Target Period` = the known ground-truth spacing of the input bars.
  - `Measured Grating Period` = the spacing measured from the reconstruction profile after diffuser + D2NN.
- Make it obvious that the diffuser-only intermediate panel is a degraded baseline, not the source of the measured period shown in the rightmost column.
- The right column must visually show how the measured period comes from the spacing between reconstructed peaks.

## Files

- Create: `src/luo2022_d2nn/figures/fig3_period_explanation.py`
- Create: `scripts/reproduce_fig3_explanation.py`
- Create: `tests/test_fig3_period_explanation.py`
- Regenerate: `figures/fig3_period_explanation.png`

## Non-Goals

- Do not change the existing Fig. 3 period sweep figure.
- Do not add both thin-lens and free-space baselines in the same explanation figure.
