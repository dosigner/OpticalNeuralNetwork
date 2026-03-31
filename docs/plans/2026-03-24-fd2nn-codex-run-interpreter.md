# FD2NN Codex Run Interpreter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first project-specific Codex skill that interprets the official `kim2026` codex runs `02~05` and emits run registries, figure registries, promotion plans, and Korean-first narrative blocks.

**Architecture:** Install a new skill under `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter`. Keep the skill body lean, move project-specific run semantics and figure-role policy into reference files, and add one deterministic helper script to parse official run summaries into a registry draft.

**Tech Stack:** Markdown skill files, Python 3 standard library, local JSON files under `/root/dj/D2NN/kim2026`

---

### Task 1: Initialize the Skill Scaffold

**Files:**
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/SKILL.md`
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/agents/openai.yaml`
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/references/`
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/scripts/`

**Step 1: Initialize the scaffold**

Run:

```bash
python3 /root/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  fd2nn-codex-run-interpreter \
  --path "${CODEX_HOME:-$HOME/.codex}/skills" \
  --resources scripts,references \
  --interface display_name="FD2NN Codex Run Interpreter" \
  --interface short_description="Interpret official kim2026 codex runs and map figures to report claims." \
  --interface default_prompt="Interpret kim2026 codex runs 02-05 and produce structured run and figure registries."
```

Expected:
- new skill directory exists under `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter`

**Step 2: Verify the scaffold exists**

Run:

```bash
find "${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter" -maxdepth 2 | sort
```

Expected:
- `SKILL.md`, `agents/openai.yaml`, `references/`, and `scripts/` are present

### Task 2: Author the Skill Body and Reference Files

**Files:**
- Modify: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/SKILL.md`
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/references/run-schema.md`
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/references/figure-policy.md`

**Step 1: Write the skill trigger and workflow**

Implement in `SKILL.md`:
- trigger language for official `02~05` codex interpretation tasks
- default root `/root/dj/D2NN/kim2026`
- official run set only
- official figure store only
- required output sections: `run registry`, `figure registry`, `promotion plan`, `narrative blocks`
- explicit v1 exclusion of PDF/PPT generation and automatic figure promotion

**Step 2: Write the run schema reference**

Implement in `references/run-schema.md`:
- exact official names for `02~05`
- sweep axis per run
- fixed conditions per run
- compared conditions per run
- interpretation cautions

**Step 3: Write the figure policy reference**

Implement in `references/figure-policy.md`:
- current `kim2026/figures` inventory
- dashboard vs report figure roles
- how `05` run-local figures relate to future promotion decisions
- when a figure should be treated as performance, physics, convergence, or phase-mask evidence

**Step 4: Self-check trigger quality**

Run:

```bash
sed -n '1,220p' "${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/SKILL.md"
```

Expected:
- description is specific enough to trigger on official run interpretation tasks without implying report generation

### Task 3: Implement the Deterministic Run Registry Helper

**Files:**
- Create: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/scripts/build_run_registry.py`

**Step 1: Write the parser**

Implement a Python script that:
- uses `/root/dj/D2NN/kim2026` as the default root unless overridden
- reads only official runs `02~05`
- loads `sweep_summary.json` or `study_summary.json` when present
- loads per-run `test_metrics.json`
- emits a JSON registry with run id, sweep axis, fixed conditions, compared conditions, key metrics, and source paths

**Step 2: Run the parser**

Run:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/scripts/build_run_registry.py" \
  --root /root/dj/D2NN/kim2026
```

Expected:
- JSON output containing four entries for runs `02`, `03`, `04`, and `05`

**Step 3: Verify one known condition**

Run:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/scripts/build_run_registry.py" \
  --root /root/dj/D2NN/kim2026 | rg '"sweep_axis"|roi-sweep|spacing-sweep|phase-first|loss-shape|loss-old'
```

Expected:
- `02~04` resolve to `spacing-sweep`
- `05` resolves to `roi-sweep`

### Task 4: Validate the Skill

**Files:**
- Validate: `${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/`

**Step 1: Run skill validation**

Run:

```bash
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  "${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter"
```

Expected:
- validation succeeds with no frontmatter or naming errors

**Step 2: Inspect generated agent metadata**

Run:

```bash
sed -n '1,220p' "${CODEX_HOME:-$HOME/.codex}/skills/fd2nn-codex-run-interpreter/agents/openai.yaml"
```

Expected:
- `display_name`, `short_description`, and `default_prompt` match the actual skill scope

### Task 5: Forward-Use the Skill Once

**Files:**
- Read: `/root/dj/D2NN/kim2026/runs/`
- Read: `/root/dj/D2NN/kim2026/figures/`

**Step 1: Invoke the skill on a realistic prompt**

Prompt:

```text
Use $fd2nn-codex-run-interpreter to interpret kim2026 codex runs 02-05 and explain which figures support the main report narrative.
```

Expected:
- answer includes `run registry`, `figure registry`, `promotion plan`, and Korean-first `narrative blocks`

**Step 2: Check for drift**

Review:
- whether `01` or `*_claude` runs leaked into the answer
- whether PDF/PPT generation was incorrectly attempted
- whether `kim2026/figures` remained the only official figure store in the answer

If drift is observed:
- tighten `SKILL.md` and the two reference files before building the next project-specific skill
