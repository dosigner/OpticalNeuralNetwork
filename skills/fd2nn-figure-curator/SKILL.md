---
name: fd2nn-figure-curator
description: Curate official FD2NN codex figures by copying selected run-local figures from `/root/dj/D2NN/kim2026/runs/02~05/.../figures` into `/root/dj/D2NN/kim2026/figures` with stable `codexXX_*` names, dry-run planning, collision checks, and no-overwrite safety. Use when promoting codex run figures into the official figure store or when previewing that promotion plan.
---

# FD2NN Figure Curator

Use this skill to promote official codex run figures into the shared `kim2026/figures` store.

## Defaults

- Work from `/root/dj/D2NN/kim2026` unless the user explicitly provides another root.
- Treat only these run-local figure folders as official promotion sources:
  - `02_fd2nn_spacing-sweep_loss-old_roi-1024_codex/figures`
  - `03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex/figures`
  - `04_fd2nn_spacing-sweep_loss-shape_roi-512_codex/figures`
  - `05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex/figures`
- Treat `/root/dj/D2NN/kim2026/figures` as the only promotion target.

## Safety Rules

- Use `copy only`. Never move or delete run-local figures.
- Default to `--dry-run`.
- Require `--apply` for real copying.
- Never overwrite existing targets.
- Treat same-bytes targets as `already_present`.
- Treat different-bytes targets as `collision`.

## Naming Rules

- Promoted files must use the `codexXX_*` namespace.
- Do not rename or overwrite existing `dashboard_*` or `report_*` figures.
- Keep curated report outputs and promoted source figures in separate namespaces.

## Workflow

1. Read [references/selection-policy.md](references/selection-policy.md) for the default subset and `--all-figures` behavior.
2. Read [references/promotion-map.md](references/promotion-map.md) for stable source-to-target names.
3. Build a dry-run promotion plan first:

```bash
python3 /root/dj/D2NN/skills/fd2nn-figure-curator/scripts/promote_figures.py \
  --root /root/dj/D2NN/kim2026 \
  --dry-run
```

4. Apply only after reviewing the JSON output:

```bash
python3 /root/dj/D2NN/skills/fd2nn-figure-curator/scripts/promote_figures.py \
  --root /root/dj/D2NN/kim2026 \
  --apply
```

5. Use `--runs` to limit promotion to a subset such as `02 05`.
6. Use `--all-figures` only when the user explicitly wants optional figures promoted too.

## Output Expectations

The script returns JSON with:

- `root`
- `target_dir`
- `runs`
- `all_figures`
- `items`
- `skipped`

Each item includes:

- `run_id`
- `source`
- `target`
- `status`
- `reason`

Expected statuses include:

- `planned`
- `already_present`
- `collision`
- `missing_source`
- `copied`

## Explicit Exclusions

This skill does not:

- interpret run metrics or write report narratives
- generate PDFs or PPTX files
- mutate `dashboard_*` or `report_*`
- overwrite existing targets automatically
