---
name: parallel-report
description: Generate comprehensive D2NN reports using parallel agents for data analysis, figure generation, LaTeX compilation, QA checking, and translation. Zero-defect output.
---

# Parallel Agent Report Pipeline

Generate publication-quality reports using 5 parallel specialized agents with a QA gate that catches all issues before presentation.

## Agent Architecture

### Agent 1: Data Analyst
- Read all experiment logs, results.json, test_metrics.json
- Compute key metrics and identify best-performing configuration
- Write physics explanations for observed performance differences
- Output: `analysis.json` with structured metrics and interpretations

### Agent 2: Figure Generator
- Create all comparison plots, convergence curves, architecture diagrams
- Use matplotlib with Nature-style formatting
- Verify every output image file exists and is under 5MB
- Output: `figures/` directory with all .png files + `figure_manifest.json`

### Agent 3: LaTeX Writer
- Write the report with proper equations, parameter tables, figure references
- Use xelatex with Noto CJK fonts for Korean support
- Compile and fix ALL LaTeX errors before proceeding
- Output: `report.tex` + `report.pdf`

### Agent 4: QA Checker (GATE)
Run after Agents 1-3 complete:
```
Checklist:
[ ] All figure paths in LaTeX resolve to existing files
[ ] No missing reference warnings in compilation log
[ ] No LaTeX errors (zero error count in .log)
[ ] All tables have data (no empty cells)
[ ] Page count matches expectations
[ ] Images not stretched (aspect ratios correct)
[ ] Korean fonts render correctly (if KO version)
[ ] xcolor loaded before definecolor calls
[ ] Math in section titles properly escaped
```
- If ANY check fails: return to responsible agent for fix
- Iterate until zero issues

### Agent 5: Translator (after QA passes)
- Produce Korean version with identical structure
- Font priority: Pretendard > Noto Sans CJK KR > English fallback
- Re-run QA checker on Korean version

## Coordination Rules
1. Agents 1, 2 run in parallel (independent)
2. Agent 3 waits for Agents 1 and 2 (needs analysis + figures)
3. Agent 4 runs after Agent 3 (QA gate)
4. Agent 5 runs only after Agent 4 passes with zero issues
5. NEVER present report to user until Agent 4 confirms zero issues

## Output
```
=== Report Pipeline Complete ===
Data Analysis: DONE (12 runs analyzed, best: roi_512_hybrid)
Figures: DONE (8 figures generated, all verified)
LaTeX: DONE (compiled, 12 pages)
QA: PASSED (0 issues, 2nd iteration - fixed 1 missing figure ref)
Translation: DONE (Korean version compiled, QA passed)

Files:
  EN: /path/to/report_en.pdf (12 pages, 8 figures)
  KO: /path/to/report_ko.pdf (12 pages, 8 figures)
```
