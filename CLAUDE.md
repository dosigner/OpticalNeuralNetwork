# D2NN Project — Claude Code Instructions

## Physics & Optical Simulations
- Always verify physical parameters (focal length, pixel pitch units, NA, aperture sizes) against realistic values before running sweeps.
- When working with D2NN/FD2NN code, confirm propagation method (beam reducer vs zoom propagate) before launching experiments.
- When computing baselines (CO/IO, Strehl ratio), use raw input beams, NOT D2NN outputs.
- Never use zoom propagate when a beam reducer is needed — confirm propagation pipeline before sweeps.
- Energy conservation check: if >10% energy loss at any stage, flag and investigate before proceeding.
- Standard optics reference: f=25mm Thorlabs AC127-025-C, dx_fourier=37.8μm (current FD2NN setup).

## Physics & Optical Simulations


## Report & PDF Generation
- When generating PDFs with figures, always verify image paths exist and are correctly referenced before rendering.
- For Korean reports, use English annotations in matplotlib figures to avoid font issues, or ensure Korean fonts are installed first.
- After generating any PDF/PPTX, do a self-QA pass checking: all figures render, aspect ratios are correct, margins/layout are consistent.
- When generating LaTeX PDFs, always verify: (1) all figure paths resolve, (2) math in section titles is escaped, (3) xcolor/definecolor load order is correct, (4) Korean fonts available or fallback to English.
- For PPTX generation, verify image paths exist and preserve aspect ratios — never stretch images.
- After generating any report, run a self-check: count figures referenced vs figures included, confirm zero LaTeX errors.

## Training & Sweeps
- Before running sweep experiments, ensure no duplicate data generation processes write concurrently (causes BadZipFile errors).
- Monitor training via log files, not background agents (which get killed).
- Always compare results against paper baselines with matching hyperparameters (batch size, epochs) before drawing conclusions.
- For long GPU commands: provide copy-paste commands for user to run manually, do not use Bash tool directly.

## Experiment Sweeps
- Before launching GPU training sweeps, check for existing processes on target GPUs to avoid resource contention.
- Never run duplicate data generation processes concurrently (causes BadZipFile / corruption errors).
- When a sweep is running, monitor it rather than making code changes that could interfere.

## Experiment Reproduction
- Binary files (.pt, .npy, .npz) in kim2026/autoresearch/runs/ are gitignored.
- To regenerate: `./scripts/reproduce_experiments.sh [experiment_name] --gpu N`
- Use `/reproduce-pib-experiments` skill for guided reproduction.
- Data generation must happen first: `cd kim2026 && PYTHONPATH=src python -m kim2026.data.generate`

## Execution Style
- Prioritize execution over planning. When the user asks for a report or sweep, start building immediately rather than exploring files or asking clarifying questions.
- Keep responses concise. No trailing summaries of what was just done.

## Python Environment
- Primary conda env: `d2nn` at `miniconda3/envs/d2nn/bin/python` relative to the repo root
- Always use `PYTHONPATH=src` when running project scripts.
