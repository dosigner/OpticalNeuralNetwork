# Promotion Map

This reference fixes the official target names for promoted codex figures.

## Runs 02-04

| Source | Target | Role | Default |
| --- | --- | --- | --- |
| `fig1_epoch_curves.png` | `codex0X_spacing_epoch_curves.png` | convergence | yes |
| `fig2_test_metrics.png` | `codex0X_spacing_test_metrics.png` | performance comparison | yes |
| `fig3_field_full_comparison.png` | `codex0X_spacing_field_full.png` | physical comparison | yes |
| `fig4_field_zoom_comparison.png` | `codex0X_spacing_field_zoom.png` | optional zoom detail | no |
| `fig5_field_profiles.png` | `codex0X_spacing_field_profiles.png` | optional profile detail | no |
| `fig6_phase_masks.png` | `codex0X_spacing_phase_masks.png` | phase-mask inspection | yes |

For run `04` only:

| Source | Target | Role | Default |
| --- | --- | --- | --- |
| `fig7_1mm_three_way_comparison.png` | `codex04_spacing_1mm_three_way.png` | optional special comparison | no |

## Run 05

| Source | Target | Role | Default |
| --- | --- | --- | --- |
| `fig1_epoch_curves.png` | `codex05_roi_epoch_curves.png` | convergence | yes |
| `fig2_phase_metrics.png` | `codex05_roi_phase_metrics.png` | phase metrics | yes |
| `fig3_field_comparison.png` | `codex05_roi_field_comparison.png` | field comparison | yes |
| `fig4_support_and_leakage.png` | `codex05_roi_support_leakage.png` | support/leakage analysis | yes |
| `fig5_phase_masks_raw_vs_wrapped.png` | `codex05_roi_phase_masks_raw_vs_wrapped.png` | phase-mask inspection | yes |

## Namespace Rule

- `dashboard_*` and `report_*` are already curated outputs.
- promoted source figures must always live under `codexXX_*`
- do not collapse promoted files into the `dashboard_*` or `report_*` namespaces
