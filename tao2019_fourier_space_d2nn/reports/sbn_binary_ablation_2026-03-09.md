# Binary-GT SBN Ablation

This round fixes the GT to the `binary` variant and changes only the SBN operating point.

Baseline reference:
- `reports/tuning_runs/pilot_gt_binary_s100_e15/260309_033944`
- `eval_fmax = 0.4438`

## Results

| Variant | eval_fmax | Delta vs binary baseline |
| --- | ---: | ---: |
| baseline (`background_perturbation`, learnable sat) | 0.4438 | -- |
| SBN off | 0.4503 | +0.0065 |
| `per_sample_minmax` | 0.4416 | -0.0022 |
| fixed saturation (`I_sat=0.01`) | 0.4485 | +0.0047 |
| fixed saturation (`I_sat=0.005`) | 0.4477 | +0.0039 |
| fixed saturation (`I_sat=0.05`) | 0.4430 | -0.0008 |

Metric files:
- `reports/tuning_runs/pilot_binary_sbn_off_s100_e15/260309_165919/metrics.json`
- `reports/tuning_runs/pilot_binary_sbn_perminmax_s100_e15/260309_170051/metrics.json`
- `reports/tuning_runs/pilot_binary_sbn_fixedsat_s100_e15/260309_170216/metrics.json`
- `reports/tuning_runs/pilot_binary_sbn_fixedsat0005_s100_e15/260309_171933/metrics.json`
- `reports/tuning_runs/pilot_binary_sbn_fixedsat005_s100_e15/260309_172825/metrics.json`

## Qualitative Read

Representative grids:
- baseline: `reports/tuning_runs/pilot_gt_binary_s100_e15/260309_033944/figures/saliency_grid.png`
- SBN off: `reports/tuning_runs/pilot_binary_sbn_off_s100_e15/260309_165919/figures/saliency_grid.png`
- fixed saturation: `reports/tuning_runs/pilot_binary_sbn_fixedsat_s100_e15/260309_170216/figures/saliency_grid.png`

Observed pattern:
- All variants remain blob-like.
- `SBN off` reduces the overly smooth center-lobe slightly and is the best pilot numerically.
- `per_sample_minmax` does not help.
- `learnable_saturation=false` is slightly better than `true`, suggesting saturation learning may be adding noise rather than useful adaptation.
- Within the fixed-saturation branch, `I_sat=0.01` is best; both `0.005` and `0.05` are worse.

## Diagnosis

For the current binary-GT setup:
- The main issue is **not** that SBN is too weak because `SBN off` is best.
- The issue is also **not** solved by `per_sample_minmax`.
- If we keep SBN, the evidence favors:
  - `background_perturbation`
  - `learnable_saturation = false`
  - `saturation_intensity = 0.01`

## Next Recommended Axis

SBN 쪽에서는 더 이상 큰 여지가 작아 보입니다.

권장:
- 주 기준선은 `SBN off`
- SBN을 유지해야 한다면 `background_perturbation + learnable_saturation=false + I_sat=0.01`
- 다음은 `optical setup` 축으로 넘어가서:
  - `f1=f2`
  - layer spacing
  - detector/crop geometry
  - phase init / phase range
  중 하나만 바꾸는 식으로 가는 편이 맞습니다.
