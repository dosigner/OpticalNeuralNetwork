---
name: report
description: Generate publication-quality PDF/PPTX reports from D2NN experiment results. Includes figure verification, LaTeX QA, and optional Korean translation.
---

# Report Generation Skill

Generate comprehensive technical reports from experiment results.

## Workflow

### Step 1: Collect Results
1. Scan the specified experiment directory for:
   - `results.json`, `test_metrics.json`, `history.json`, `config.yaml`
   - All `.png` / `.pdf` figure files
2. Build a manifest of available data and figures

### Step 2: Generate Report
1. Write report content with:
   - Parameter tables with exact values from configs
   - LaTeX equations (properly escaped in titles)
   - Figure references with verified paths
   - Comparison against baselines where available

### Step 3: QA Checklist (MANDATORY before presenting)
```
□ All figure references have matching files that exist on disk
□ LaTeX compiles without errors
□ xcolor package loaded before definecolor calls
□ No missing figure warnings in compilation log
□ Page count matches expectations
□ All tables have data (no empty cells)
□ Korean fonts available if KO version requested (Noto CJK fallback)
□ PPTX images not stretched (aspect ratio preserved)
```

### Step 4: Compile & Verify
```bash
# Compile
xelatex -interaction=nonstopmode report.tex

# Verify figures
grep -o 'includegraphics.*{[^}]*}' report.tex | while read line; do
  fig=$(echo "$line" | grep -o '{[^}]*}' | tr -d '{}')
  [ -f "$fig" ] && echo "✅ $fig" || echo "❌ MISSING: $fig"
done

# Check for errors
grep -c "Error\|Warning.*undefined" report.log
```

### Step 5: Output
- Present the compiled PDF path
- List any warnings or issues found during QA
- If KO version requested, generate after EN version passes QA

## Format Options
- **PDF (LaTeX)**: Nature-style, proper math typesetting
- **PPTX**: Slide deck with embedded figures, speaker notes
- **Obsidian**: Markdown vault with wikilinks and embedded images
- **Bilingual**: EN + KO pair with identical structure

## Notes
- Always verify figure paths BEFORE compilation, not after
- Korean font priority: Pretendard > Noto Sans CJK KR > fallback to English
- For PPTX, use python-pptx with explicit image dimensions (never auto-scale)
- If compilation fails, fix and recompile — do not present a broken PDF
