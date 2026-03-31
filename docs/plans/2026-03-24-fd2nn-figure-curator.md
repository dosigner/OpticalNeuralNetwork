# FD2NN Figure Curator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the second project-specific Codex skill that promotes selected official FD2NN codex run-local figures from `kim2026/runs/02~05/.../figures` into `/root/dj/D2NN/kim2026/figures` using stable `codexXX_*` names, copy-only behavior, and collision-safe defaults.

**Architecture:** Create a new repo-local skill under `/root/dj/D2NN/skills/fd2nn-figure-curator`. Keep the skill instructions lean, store promotion naming and selection rules in references, and implement deterministic copy planning in a single script that defaults to dry-run mode and requires `--apply` for mutations.

**Tech Stack:** Markdown skill files, Python 3 standard library, local PNG files under `/root/dj/D2NN/kim2026`

---

### Task 1: Initialize the Skill Scaffold

**Files:**
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/SKILL.md`
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/agents/openai.yaml`
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/references/`
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/scripts/`

**Step 1: Initialize the scaffold**

Run:

```bash
python3 /root/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  fd2nn-figure-curator \
  --path /root/dj/D2NN/skills \
  --resources scripts,references \
  --interface display_name="FD2NN Figure Curator" \
  --interface short_description="Promote codex figures into official store" \
  --interface default_prompt="Use $fd2nn-figure-curator to dry-run or apply promotion of codex run figures into kim2026/figures."
```

Expected:
- new skill directory exists under `/root/dj/D2NN/skills/fd2nn-figure-curator`

### Task 2: Author the Skill Body and References

**Files:**
- Modify: `/root/dj/D2NN/skills/fd2nn-figure-curator/SKILL.md`
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/references/promotion-map.md`
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/references/selection-policy.md`
- Modify: `/root/dj/D2NN/skills/fd2nn-figure-curator/agents/openai.yaml`

**Step 1: Write the skill body**

Implement in `SKILL.md`:
- trigger language for codex figure promotion tasks
- source run set `02~05`
- official target store `/root/dj/D2NN/kim2026/figures`
- `copy only`
- `dry-run` default
- overwrite forbidden
- `codexXX_*` naming namespace

**Step 2: Write the promotion map**

Implement in `references/promotion-map.md`:
- source figure name
- target official filename
- run id
- report role

**Step 3: Write the selection policy**

Implement in `references/selection-policy.md`:
- default subset for `02~04`
- default subset for `05`
- optional `--all-figures`
- why `dashboard_*` and `report_*` are not part of the promotion namespace

**Step 4: Verify metadata coherence**

Run:

```bash
sed -n '1,220p' /root/dj/D2NN/skills/fd2nn-figure-curator/agents/openai.yaml
```

Expected:
- UI metadata matches actual skill scope and does not imply report generation

### Task 3: Write Tests for Promotion Planning

**Files:**
- Create: `/root/dj/D2NN/kim2026/tests/test_fd2nn_figure_curator_skill.py`

**Step 1: Write the failing tests**

Cover:
- default dry-run selects the approved subset only
- `--all-figures` includes optional figures
- target names use `codexXX_*`
- collisions are reported and not overwritten
- `--apply` copies files while preserving source files

**Step 2: Run the tests to verify failure**

Run:

```bash
miniconda3/envs/d2nn/bin/python -m pytest kim2026/tests/test_fd2nn_figure_curator_skill.py -q
```

Expected:
- failing tests because the script does not yet exist

### Task 4: Implement the Promotion Script

**Files:**
- Create: `/root/dj/D2NN/skills/fd2nn-figure-curator/scripts/promote_figures.py`

**Step 1: Implement deterministic planning**

The script must:
- use `/root/dj/D2NN/kim2026` as default root
- support `--runs`
- support `--dry-run`
- support `--apply`
- support `--all-figures`
- resolve the default subset from `references/selection-policy.md` or hardcoded v1 table
- resolve target names from `references/promotion-map.md` or hardcoded v1 table

**Step 2: Implement safety behavior**

The script must:
- copy only
- keep source files untouched
- detect target collisions
- refuse overwrite by default
- report copied, skipped, and conflicted files clearly

**Step 3: Run the tests**

Run:

```bash
miniconda3/envs/d2nn/bin/python -m pytest kim2026/tests/test_fd2nn_figure_curator_skill.py -q
```

Expected:
- all tests pass

### Task 5: Validate the Skill and Dry-Run on Real Data

**Files:**
- Validate: `/root/dj/D2NN/skills/fd2nn-figure-curator/`
- Read: `/root/dj/D2NN/kim2026/runs/02~05/.../figures`
- Read: `/root/dj/D2NN/kim2026/figures`

**Step 1: Run skill validation**

Run:

```bash
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /root/dj/D2NN/skills/fd2nn-figure-curator
```

Expected:
- skill validates successfully

**Step 2: Run real dry-run**

Run:

```bash
python3 /root/dj/D2NN/skills/fd2nn-figure-curator/scripts/promote_figures.py \
  --root /root/dj/D2NN/kim2026 \
  --runs 02 03 04 05 \
  --dry-run
```

Expected:
- deterministic promotion plan
- `codexXX_*` names
- no mutations
- clear collision report if any names already exist

### Task 6: Optional Controlled Apply

**Files:**
- Copy into: `/root/dj/D2NN/kim2026/figures`

**Step 1: Apply only after review**

Run:

```bash
python3 /root/dj/D2NN/skills/fd2nn-figure-curator/scripts/promote_figures.py \
  --root /root/dj/D2NN/kim2026 \
  --runs 02 03 04 05 \
  --apply
```

Expected:
- selected files copied
- source files unchanged
- no silent overwrite
- official figure store updated only with approved `codexXX_*` files
