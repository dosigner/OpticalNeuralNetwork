#!/usr/bin/env python3
"""Convert markdown paper to PDF with embedded images via weasyprint."""

import sys
from pathlib import Path
import markdown
from weasyprint import HTML

SCRIPT_DIR = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 2cm 2.2cm; }
body { font-family: 'Noto Serif CJK KR', 'Noto Serif', serif; font-size: 10.5pt;
       line-height: 1.6; color: #222; }
h1 { font-size: 18pt; text-align: center; margin-top: 0.5cm; border-bottom: 2px solid #333;
     padding-bottom: 8pt; font-family: 'Noto Sans CJK KR', sans-serif; }
h2 { font-size: 14pt; margin-top: 1.2em; border-bottom: 1px solid #999;
     padding-bottom: 4pt; font-family: 'Noto Sans CJK KR', sans-serif; }
h3 { font-size: 12pt; margin-top: 1em; font-family: 'Noto Sans CJK KR', sans-serif; }
p { text-align: justify; margin: 0.4em 0; }
img { max-width: 100%; display: block; margin: 0.8em auto; border: 1px solid #ddd;
      page-break-inside: avoid; }
table { border-collapse: collapse; margin: 0.8em auto; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4pt 8pt; text-align: center; }
th { background: #f0f0f0; font-weight: bold; }
blockquote { background: #f8f8f8; border-left: 3px solid #4a90d9; padding: 8pt 12pt;
             margin: 0.6em 0; font-style: normal; font-weight: bold; }
strong { color: #1a3a5c; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.2em 0; }
code { background: #f5f5f5; padding: 1pt 3pt; font-size: 9pt; }
ol, ul { margin: 0.3em 0; padding-left: 2em; }
li { margin: 0.15em 0; }
"""


def main():
    md_path = SCRIPT_DIR / "PAPER_static_d2nn_limits_v2.md"
    pdf_path = SCRIPT_DIR / "PAPER_static_d2nn_limits_v2.pdf"

    if len(sys.argv) > 1:
        md_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        pdf_path = Path(sys.argv[2])

    text = md_path.read_text(encoding="utf-8")

    extensions = ["tables", "fenced_code", "sane_lists"]
    html_body = markdown.markdown(text, extensions=extensions)

    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{CSS}</style>
</head><body>
{html_body}
</body></html>"""

    HTML(string=full_html, base_url=str(SCRIPT_DIR)).write_pdf(str(pdf_path))
    print(f"PDF written: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
