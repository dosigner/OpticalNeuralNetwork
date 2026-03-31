# Official Run Schema

This reference locks the meaning of the official `kim2026` codex run set. Treat these four runs as the default evidence base for v1 interpretation.

## Run Table

| Run ID | Directory | Sweep Axis | Loss Family | Fixed Conditions | Compared Conditions | Interpretation Note |
| --- | --- | --- | --- | --- | --- | --- |
| `02` | `02_fd2nn_spacing-sweep_loss-old_roi-1024_codex` | `spacing-sweep` | `old` | `roi=1024` | `spacing_0mm`, `spacing_0p1mm`, `spacing_1mm`, `spacing_2mm` | Baseline spacing sweep for the older loss definition. |
| `03` | `03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex` | `spacing-sweep` | `shape` | `roi=1024` | `spacing_0mm`, `spacing_0p1mm`, `spacing_1mm`, `spacing_2mm` | Same ROI as `02`, but with shape-oriented loss. |
| `04` | `04_fd2nn_spacing-sweep_loss-shape_roi-512_codex` | `spacing-sweep` | `shape` | `roi=512` | `spacing_0mm`, `spacing_0p1mm`, `spacing_1mm`, `spacing_2mm` | Same loss family as `03`, but ROI changes to test support size effects. |
| `05` | `05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex` | `roi-sweep` | `phase-first` | `spacing=0p1mm` | `roi512_spacing_0p1mm`, `roi1024_spacing_0p1mm` | ROI-only comparison used to validate the phase-first loss under fixed spacing. |

## Interpretation Rules

- `02~04` are all spacing sweeps. Do not describe them as ROI sweeps.
- `05` is the only official ROI sweep in this set.
- `loss-old`, `loss-shape`, and `loss-phase-first` are different loss-family labels, not different optical geometries.
- `roi-512` and `roi-1024` are support-size conditions, not training stages.
- `kim2026/figures` is the official figure store for downstream explanation. Run-local `figures/` folders are provenance sources only unless a promotion plan says otherwise.

## Default Narrative Anchors

- `02`: older loss baseline across spacing
- `03`: shape-loss spacing behavior at large ROI
- `04`: ROI reduction effect under shape loss
- `05`: phase-first loss validation via ROI-only comparison at fixed spacing
