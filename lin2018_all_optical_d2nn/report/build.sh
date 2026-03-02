#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pandoc report.md -o report.pdf \
  --pdf-engine=xelatex \
  -V mainfont="Noto Sans CJK KR" \
  -V monofont="DejaVu Sans Mono" \
  -V mathfont="Latin Modern Math" \
  -V geometry:margin=2.5cm \
  -V fontsize=11pt \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V header-includes='\usepackage{amsmath}\usepackage{amssymb}\usepackage{booktabs}\usepackage{float}\usepackage{graphicx}' \
  --highlight-style=tango \
  --toc \
  --number-sections

echo "Built: $(pwd)/report.pdf"
