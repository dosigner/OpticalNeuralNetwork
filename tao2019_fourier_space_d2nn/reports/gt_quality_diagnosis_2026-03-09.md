# GT Quality Diagnosis For CIFAR Fu2013 Saliency

## Scope

This diagnosis isolates the `GT quality` axis while keeping the FD2NN, optics, SBN placement, and loss definition fixed.

- Baseline config: `src/tao2019_fd2nn/config/saliency_cifar_fu2013_cat2horse_bs10_100pad160_f2mm.yaml`
- Pilot budget: `15 epochs`, `100 steps/epoch`
- Save root: `reports/tuning_runs/`

## 1. Raw Fu2013 GT Audit

Source: `reports/gt_audit/summary.json`

Train split means:
- `foreground_ratio = 0.2676`
- `center_offset_norm = 0.1027`
- `component_count = 4.3392`
- `largest_component_ratio = 0.6195`
- `edge_density = 0.0950`

Interpretation:
- The raw pseudo-GT is not extremely center-collapsed by centroid.
- It is, however, fairly soft and fragmented.
- An average of `4.3` connected components and `largest_component_ratio ~ 0.62` is much closer to a broken saliency field than a single clean silhouette.

Overlay samples:
- `reports/gt_audit/train_overlay_grid.png`
- `reports/gt_audit/val_overlay_grid.png`

## 2. GT Variant Audit

Sources:
- `reports/gt_audit_variants/raw/summary.json`
- `reports/gt_audit_variants/binary/summary.json`
- `reports/gt_audit_variants/sharpened/summary.json`

Key differences on the train split:

| Variant | foreground_ratio | center_offset_norm | component_count | largest_component_ratio | edge_density |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.2676 | 0.1027 | 4.3392 | 0.6195 | 0.0950 |
| binary | 0.3137 | 0.1641 | 3.9350 | 0.7346 | 0.1645 |
| sharpened | 0.2691 | 0.1018 | 5.9202 | 0.5634 | 0.1330 |

Interpretation:
- `binary` makes masks crisper and more connected.
- `sharpened` raises edge density, but it also increases fragmentation.
- This matters because the raw GT appears to reward diffuse energy over a single coherent object region.

## 3. GT-Only Pilot Results

Source summary: `reports/gt_variant_pilots_2026-03-09.json`

| Variant | eval_fmax | best_epoch | Run Dir |
| --- | ---: | ---: | --- |
| raw | 0.2434 | 10 | `reports/tuning_runs/pilot_gt_raw_s100_e15/260309_033335` |
| binary | 0.4438 | 10 | `reports/tuning_runs/pilot_gt_binary_s100_e15/260309_033944` |
| sharpened | 0.2730 | 5 | `reports/tuning_runs/pilot_gt_sharpened_s100_e15/260309_034541` |

Important caveat:
- These F-max values are **not directly apples-to-apples across variants**, because the validation GT itself changes with the variant.

Still, the result is decisive for root-cause analysis:
- Changing only GT topology changes training behavior dramatically.
- `binary` improves alignment far more than `sharpened`.
- Therefore, the current soft fragmented GT is a primary driver of the failure mode.

## 4. Qualitative Notes

Representative figures:
- raw: `reports/tuning_runs/pilot_gt_raw_s100_e15/260309_033335/figures/saliency_grid.png`
- binary: `reports/tuning_runs/pilot_gt_binary_s100_e15/260309_033944/figures/saliency_grid.png`
- sharpened: `reports/tuning_runs/pilot_gt_sharpened_s100_e15/260309_034541/figures/saliency_grid.png`

Observed pattern:
- All three variants still produce low-frequency blob-like predictions.
- `binary` shows stronger object-correlated bright structure than `raw`.
- `sharpened` stays visually close to `raw`, suggesting edge emphasis alone does not fix the issue.

## 5. Diagnosis

Current conclusion:
- `GT quality` is a **primary cause** of blob-like CIFAR saliency behavior.
- More specifically, `soft + fragmented pseudo-GT` is the harmful part.
- However, GT is **not the only cause**, because blob-like predictions remain even under the binary variant.

## 6. Recommended Next Axis

Move next to `SBN operating point`, while **holding the binary GT fixed**.

Reason:
- GT has been shown to matter.
- The remaining gap is now more likely due to how the nonlinear phase response redistributes intensity under the fixed optics.
