# PDF Edition eLight-Style Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `reports/obsidian_to_pdf.py`를 수정해 `final_report_fd2nn_pdf.md`를 eLight 저널에 가까운 준저널 브로슈어형 PDF로 다시 렌더링한다.

**Architecture:** 렌더링 파이프라인은 유지하고, `build_css()`, `build_title_page()`, `build_html()`의 HTML/CSS 계층만 정제한다. 테스트는 스크립트 import 후 CSS와 title page에 필요한 구조적 토큰이 포함되는지 검증하는 방식으로 작성한다.

**Tech Stack:** Python, pytest, Markdown, WeasyPrint

---

### Task 1: Style Contract Test

**Files:**
- Create: `tests/test_obsidian_to_pdf_style.py`
- Test target: `reports/obsidian_to_pdf.py`

**Step 1: Write the failing test**

- `build_css()` 결과에 serif body font, journal-shell wrapper, refined figure caption/table styling 토큰이 포함되는지 검증한다.
- `build_title_page()` 결과에 `journal-kicker`, `title-deck`, `report-metadata` 구조가 포함되는지 검증한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_obsidian_to_pdf_style.py -q`

Expected: FAIL because current renderer does not emit those selectors/classes.

**Step 3: Write minimal implementation**

- `obsidian_to_pdf.py`에 필요한 CSS 클래스와 title page HTML 구조를 추가한다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_obsidian_to_pdf_style.py -q`

Expected: PASS

### Task 2: Journal-Style Renderer Refresh

**Files:**
- Modify: `reports/obsidian_to_pdf.py`

**Step 1: Update CSS**

- body serif typography
- restrained ink palette
- title page kicker/deck/meta
- wrapper classes for article body
- refined table, figure, blockquote, code, callout styling

**Step 2: Update HTML wrappers**

- `build_title_page()` to output journal-style front matter
- `build_html()` to wrap title page and body in stable containers

**Step 3: Re-run targeted test**

Run: `pytest tests/test_obsidian_to_pdf_style.py -q`

Expected: PASS

### Task 3: Render and Visual Verification

**Files:**
- Input: `reports/final_report_fd2nn_pdf.md`
- Output: `reports/final_report_fd2nn_pdf.pdf`

**Step 1: Render PDF**

Run: `python reports/obsidian_to_pdf.py reports/final_report_fd2nn_pdf.md reports/final_report_fd2nn_pdf.pdf`

Expected: PDF generated successfully.

**Step 2: Render pages to PNG for review**

Run: `pdftoppm -png reports/final_report_fd2nn_pdf.pdf tmp/final_report_fd2nn_pdf`

Expected: PNG pages generated.

**Step 3: Inspect first few pages**

- Verify cover page hierarchy
- Verify section typography
- Verify figure/table/callout tone

### Task 4: Final Regression Check

**Files:**
- Verify only: `reports/obsidian_to_pdf.py`, `tests/test_obsidian_to_pdf_style.py`, output PDF

**Step 1: Run focused tests**

Run: `pytest tests/test_obsidian_to_pdf_style.py -q`

Expected: PASS

**Step 2: Confirm output file timestamp and size updated**

Run: `ls -lh reports/final_report_fd2nn_pdf.pdf`

Expected: regenerated PDF exists with current timestamp.
