# Selection Policy

Use this policy to decide which run-local figures are promoted by default.

## Default Promotion Set

### Runs 02-04

Promote only:

- `fig1_epoch_curves.png`
- `fig2_test_metrics.png`
- `fig3_field_full_comparison.png`
- `fig6_phase_masks.png`

Do not promote by default:

- `fig4_field_zoom_comparison.png`
- `fig5_field_profiles.png`
- `fig7_1mm_three_way_comparison.png` for run `04`

### Run 05

Promote all five figures by default:

- `fig1_epoch_curves.png`
- `fig2_phase_metrics.png`
- `fig3_field_comparison.png`
- `fig4_support_and_leakage.png`
- `fig5_phase_masks_raw_vs_wrapped.png`

## Optional Full Promotion

Use `--all-figures` to promote optional figures for the selected runs.

This is opt-in because the official store should remain curated rather than becoming a mirror of all run-local figures.

## Operational Policy

- `--dry-run` is the default review mode
- `--apply` is required for mutation
- overwrite is forbidden
- preserve originals under `runs/.../figures`
