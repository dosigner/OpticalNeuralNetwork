# Supplementary Figure S4 Pruning Design

**Date:** 2026-03-16

**Goal:** Reproduce Supplementary Figure S4 from Luo et al. at the level of panel structure, labels, and PCC trend using the existing `n20_L4` checkpoint in this repo, without retraining.

## Scope

This design adds a new figure-generation path for Supplementary Figure S4 under `luo2022_random_diffusers_d2nn`. The target is qualitative and structural fidelity to the paper:

- 6 pruning-condition rows
- 4 diffractive-layer panels per row
- 4 reconstruction panels per row
- per-panel PCC display
- row/column labels that make each condition immediately readable

Out of scope:

- pixel-identical reproduction to the paper
- recovering the exact random seeds or exact complex object used by the authors
- retraining models

## Inputs

The figure will use a fixed baseline checkpoint:

- `luo2022_random_diffusers_d2nn/runs/n20_L4/model.pt`

The figure will use two input objects:

- an MNIST digit `2`
- a user-provided grayscale OOD object, transformed so that pure white background becomes black and the non-white foreground keeps its grayscale values

The figure will use two diffusers:

- one known diffuser
- one new diffuser

These inputs are held fixed across all rows so that the only changed variable is the pruning rule.

## Reused Code

The implementation will deliberately reuse existing code paths:

- island-mask extraction from `figs3_overlap_map.py`
- known/new diffuser generation and single-object forward evaluation from `fig2_known_new.py`
- baseline optical/config values from `configs/baseline.yaml`

## Figure Structure

Each row corresponds to one condition:

1. `Full layers`
2. `No layers`
3. `Islands only`
4. `Dilated islands`
5. `Inside contour`
6. `80lambda aperture`

Each row contains:

- 4 left panels: Layer 1-4 phase maps under that pruning condition
- 4 right panels: output images for
  - digit 2, known diffuser
  - digit 2, new diffuser
  - OOD object, known diffuser
  - OOD object, new diffuser

Each row must visibly state what the condition is. The left side of the row should carry a readable condition label so the first block of panels is self-explanatory without reading the caption.

Each output panel should show PCC on the panel itself. Each row should also expose kept-area information to make pruning severity legible.

## Pruning Definitions

All pruning rules are defined per diffractive layer.

### Full layers

Use the wrapped learned phase maps exactly as stored in the checkpoint.

### No layers

Replace every layer with zero phase modulation.

### Islands only

Use the binary phase-island masks derived by the S3 procedure. Pixels inside the mask retain the learned phase. Pixels outside the mask are set to zero phase modulation.

### Dilated islands

Take the phase-island mask and apply one binary dilation step. Retain learned phase inside the dilated region and zero phase elsewhere.

### Inside contour

Take the islands, build a closed filled support from them, and keep all pixels within that support. The concrete implementation target is a filled connected mask derived from the island support, not a convex hull.

### 80lambda aperture

Keep all pixels inside the circular aperture used in S3 and set everything outside to zero phase modulation.

## Data Flow

1. Load config and baseline checkpoint.
2. Extract wrapped phases for 4 layers.
3. Build the base circular ROI.
4. Compute island masks using the existing S3 logic.
5. Derive the six row masks from the base masks.
6. Materialize one phase stack per row.
7. Build a pruned model copy for each row by replacing each layer phase map.
8. Load the fixed digit-2 sample and the fixed OOD object.
9. Generate one known and one new diffuser.
10. Forward all object/diffuser pairs through all row models.
11. Compute PCC on raw intensities.
12. Apply display-only contrast enhancement.
13. Assemble and save the final figure plus raw arrays/metadata.

## Validation

Validation focuses on structural fidelity to the paper.

Required checks:

- output image exists and has a 6-row layout
- labels clearly identify all rows and all output columns
- kept-area increases in the expected order across pruning rules
- PCC trend is qualitatively aligned with the paper:
  - `Full layers` best
  - `No layers` poor
  - `Islands only` poor
  - `Dilated islands` not much better than islands only
  - `Inside contour` improved but not best
  - `80lambda aperture` close to full layers

The final review compares the generated figure side-by-side with the paper PDF for panel structure, labels, and trend only.

## Risks and Resolutions

- Exact paper OOD object is unavailable.
  Resolution: use the user-provided grayscale object and treat paper-level identity as out of scope.

- Exact pruning semantics for `inside contour` and `dilated islands` are under-specified.
  Resolution: choose deterministic, documented morphology operations and validate trend rather than exact image identity.

- The worktree already contains unrelated user changes.
  Resolution: avoid reverting any existing edits and keep new work isolated to S4-specific files.

## Comparison Notes

Generated outputs:

- `luo2022_random_diffusers_d2nn/figures/figS4_pruning.png`
- `luo2022_random_diffusers_d2nn/figures/figS4_pruning.npy`

Observed qualitative agreement with the paper:

- 6-row panel structure matches the paper layout.
- Row labels and output-column labels are explicit on the figure.
- `Full layers` gives the strongest reconstruction.
- `Islands only` and `Dilated islands` perform substantially worse.
- `Inside contour` recovers most of the digit quality while still trailing `Full layers`.
- `80lambda aperture` is very close to `Full layers`.

Observed PCCs for the generated figure:

- `Full layers`: `0.8906 / 0.8936` for digit 2, `0.6645 / 0.6665` for the OOD object
- `No layers`: `0.7862 / 0.8238`, `0.5839 / 0.5710`
- `Islands only`: `0.6247 / 0.6536`, `0.4470 / 0.4457`
- `Dilated islands`: `0.5300 / 0.5624`, `0.3890 / 0.3972`
- `Inside contour`: `0.8654 / 0.8849`, `0.5968 / 0.6197`
- `80lambda aperture`: `0.8901 / 0.8930`, `0.6632 / 0.6648`

Remaining deviations from the paper:

- Exact example object identity differs because the original paper object was unavailable.
- PCC magnitudes differ from the paper, especially for the `No layers` row, but the relative trend and qualitative ordering match the target success criterion.
