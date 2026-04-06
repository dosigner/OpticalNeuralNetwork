# FD2NN Loss Sweep Root-Cause PPTX Design

**Date:** 2026-04-01
**Audience:** Lab internal review
**Format:** Bilingual (KO/EN), 16:9 PPTX
**Primary goal:** Diagnose why the current FD2NN beam-cleanup loss sweeps are not delivering a robust correction result, then turn that diagnosis into implementation and operations guidance.

## Scope

- Restrict evidence to `kim2026/`.
- Use committed Markdown reports, troubleshooting notes, and committed figures/artifacts only.
- Do not use raw code as primary evidence.
- Exclude claims that the review notes mark as overly strong or physically unsafe.

## Evidence Set

### Core text sources

- `kim2026/troubleshooting/TROUBLESHOOTING_PHYSICS.md`
- `kim2026/docs/04-report/features/fd2nn-loss-sweep-comprehensive.report.md`
- `kim2026/docs/04-report/features/fd2nn-loss-sweep-comprehensive.review.md`
- `kim2026/docs/04-report/features/sweep09-d2nn-br15cm-roi-complex.report.md`
- `kim2026/autoresearch/runs/ppt_requirements.md`

### Figure sources

- `kim2026/docs/04-report/diagrams/*.png`
- `kim2026/docs/04-report/features/*.png`
- `kim2026/autoresearch/runs/ppt_white/*.png`
- `kim2026/autoresearch/runs/0401-focal-pib-sweep-padded-4loss-cn2-5e14/focal_pib_only/*.png`

## Messaging Rules

- State `co↔io trade-off` as an empirical observation in the current architecture/objective/metric setting.
- Do not present bare `Strehl` as a valid classical Strehl ratio when the amplitude reference changes.
- Do not call ring-like wrapped phase a proven Fresnel lens unless the document itself supports that interpretation conservatively.
- Keep `phase-only mask cannot directly modulate amplitude` separate from `output-plane amplitude can change through propagation/interference`.
- Keep `zero-phase D2NN` separate from `identity baseline`.

## Slide Structure

1. Title and diagnostic question
2. Evidence policy and executive diagnosis
3. Optical path and operator invariants
4. Physics guardrails for interpretation
5. Historical artifact evidence: old vs new PIB / vacuum WFE
6. Historical artifact evidence: artifact compensation / Strehl failure
7. Beam-reduced input and fixed-mask limitation
8. Sweep summary
9. Hyperparameter pattern and best/worst comparison
10. Sweep gallery and coherence-cell diagnosis
11. Corrected focal-plane evidence I
12. Corrected focal-plane evidence II
13. Clean baseline and mode-conversion interpretation
14. Root-cause synthesis and implementation rules
15. Next experiments, AO context, and provenance

## Deliverable

- PPTX generator script under `kim2026/docs/04-report/`
- Output deck under `kim2026/docs/04-report/`
