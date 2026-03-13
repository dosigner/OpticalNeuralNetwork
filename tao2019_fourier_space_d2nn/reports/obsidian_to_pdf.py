#!/usr/bin/env python3
"""
Convert Obsidian markdown (final_report_fd2nn.md) to a beautifully designed PDF.

Usage:
    python obsidian_to_pdf.py [input.md] [output.pdf]

Defaults:
    input  = final_report_fd2nn.md  (in same directory as this script)
    output = final_report_fd2nn.pdf (in same directory as this script)

Dependencies:
    pip install markdown weasyprint pygments pyyaml
    npm install -g @mermaid-js/mermaid-cli
"""

import os
import re
import sys
import json
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
from textwrap import dedent

import yaml
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension  # noqa: F401
from pygments.formatters import HtmlFormatter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

CALLOUT_TYPES = {
    "note":      {"icon": "", "color": "#385170", "bg": "#f5f8fb"},
    "tip":       {"icon": "", "color": "#2f5f5b", "bg": "#f3f8f7"},
    "warning":   {"icon": "", "color": "#8a5a2b", "bg": "#fbf6ef"},
    "info":      {"icon": "", "color": "#355c7d", "bg": "#f4f7fb"},
    "example":   {"icon": "", "color": "#5b5678", "bg": "#f6f5fa"},
    "quote":     {"icon": "", "color": "#666666", "bg": "#f7f7f7"},
    "bug":       {"icon": "", "color": "#8b4343", "bg": "#fbf4f4"},
    "danger":    {"icon": "", "color": "#8b4343", "bg": "#fbf4f4"},
    "success":   {"icon": "", "color": "#345e46", "bg": "#f3f8f4"},
    "failure":   {"icon": "", "color": "#8b4343", "bg": "#fbf4f4"},
    "question":  {"icon": "", "color": "#46643d", "bg": "#f5f8f1"},
    "abstract":  {"icon": "", "color": "#355c7d", "bg": "#f4f7fb"},
    "important": {"icon": "", "color": "#7b5a33", "bg": "#faf6ef"},
    "todo":      {"icon": "", "color": "#7b5a33", "bg": "#faf6ef"},
}

# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_without_frontmatter)."""
    m = re.match(r"\A---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
    if not m:
        return {}, text
    raw = m.group(1)
    try:
        meta = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        meta = {}
    body = text[m.end():]
    return meta, body


# ---------------------------------------------------------------------------
# Callout processing  (MUST run before markdown conversion)
# ---------------------------------------------------------------------------

_MD_EXTENSIONS = ["tables", "fenced_code", "codehilite", "md_in_html"]
_MD_EXT_CONFIG = {
    "codehilite": {
        "linenums": False,
        "css_class": "codehilite",
        # NOTE: Do NOT pass 'style' or 'noclasses' here because fenced_code
        # also passes them to CodeHilite and they conflict.  We use CSS classes
        # and include a Pygments stylesheet in the HTML.
    },
}


def _render_callout_inner(lines: list[str]) -> str:
    """Convert inner content lines (already stripped of '> ') to HTML via markdown."""
    inner_md = "\n".join(lines)
    # Recursively process nested callouts inside the inner content
    inner_md = process_callouts(inner_md)
    # Process highlights
    inner_md = process_highlights(inner_md)
    # Process math (server-side KaTeX) before markdown to protect TeX syntax
    inner_md = process_math(inner_md)
    html = markdown.markdown(
        inner_md,
        extensions=_MD_EXTENSIONS,
        extension_configs=_MD_EXT_CONFIG,
    )
    return html


def process_callouts(text: str) -> str:
    """
    Find Obsidian callout blocks and replace them with styled HTML divs.
    A callout starts with `> [!type] Title` or `> [!type]- Title`
    and continues while lines start with `> ` (or are blank `>` lines).
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        # Match callout header: > [!type] optional-title  or > [!type]- optional-title
        header_match = re.match(
            r"^>\s*\[!(\w+)\](-?)\s*(.*)?$", lines[i]
        )
        if header_match:
            ctype = header_match.group(1).lower()
            # group(2) is '-' for foldable callouts (rendered expanded in PDF)
            title = (header_match.group(3) or "").strip()
            style = CALLOUT_TYPES.get(ctype, CALLOUT_TYPES["note"])

            # Collect body lines
            inner_lines: list[str] = []
            i += 1
            while i < len(lines):
                line = lines[i]
                # continuation: line starts with '>' (possibly followed by content)
                if re.match(r"^>", line):
                    # strip leading '> ' or '>'
                    stripped = re.sub(r"^>\s?", "", line)
                    inner_lines.append(stripped)
                    i += 1
                else:
                    break

            inner_html = _render_callout_inner(inner_lines)

            # Build callout HTML
            display_title = title if title else ctype.capitalize()
            icon = style["icon"]
            callout_html = (
                f'<div class="callout callout-{ctype}" '
                f'style="border-left:4px solid {style["color"]}; '
                f'background:{style["bg"]}; '
                f'padding:12px 16px; margin:16px 0; border-radius:4px; '
                f'page-break-inside:avoid;">\n'
                f'<div class="callout-title" style="font-weight:700; '
                f'color:{style["color"]}; margin-bottom:6px; font-size:0.9em; '
                f'letter-spacing:0.02em; text-transform:uppercase;">'
                f'{icon} {display_title}</div>\n'
                f'<div class="callout-body">{inner_html}</div>\n'
                f'</div>\n'
            )
            result.append(callout_html)
        else:
            result.append(lines[i])
            i += 1

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Highlights  ==text== → <mark>text</mark>
# ---------------------------------------------------------------------------

def process_highlights(text: str) -> str:
    return re.sub(r"==(.*?)==", r"<mark>\1</mark>", text)


# ---------------------------------------------------------------------------
# Mermaid diagrams → PNG (SVG foreignObject text not supported by WeasyPrint)
# ---------------------------------------------------------------------------

def process_mermaid(text: str, tmp_dir: Path) -> str:
    """Replace ```mermaid ... ``` blocks with inline PNG images."""
    pattern = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)

    def _render(match: re.Match) -> str:
        mermaid_src = match.group(1).strip()
        uid = uuid.uuid4().hex[:8]
        mmd_path = tmp_dir / f"mermaid_{uid}.mmd"
        png_path = tmp_dir / f"mermaid_{uid}.png"
        puppet_cfg = tmp_dir / "puppeteer.json"

        mmd_path.write_text(mermaid_src, encoding="utf-8")
        if not puppet_cfg.exists():
            puppet_cfg.write_text(
                json.dumps({"args": ["--no-sandbox"]}), encoding="utf-8"
            )

        mmdc = shutil.which("mmdc")
        if mmdc is None:
            return f'<pre class="mermaid-fallback"><code>{mermaid_src}</code></pre>'

        try:
            subprocess.run(
                [
                    mmdc,
                    "-i", str(mmd_path),
                    "-o", str(png_path),
                    "-p", str(puppet_cfg),
                    "-b", "white",
                    "-s", "3",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            png_uri = png_path.as_uri()
            return (
                f'<div class="mermaid-diagram" '
                f'style="text-align:center; margin:20px 0; page-break-inside:avoid;">'
                f'<img src="{png_uri}" style="max-width:100%; height:auto;" /></div>'
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[WARN] Mermaid render failed for block: {exc}", file=sys.stderr)
            return f'<pre class="mermaid-fallback"><code>{mermaid_src}</code></pre>'

    return pattern.sub(_render, text)


# ---------------------------------------------------------------------------
# Math rendering: $inline$ and $$block$$  –  server-side via `katex` CLI
# ---------------------------------------------------------------------------

_KATEX_BIN = shutil.which("katex")


def _render_katex(tex: str, display_mode: bool = False) -> str:
    """Render a single TeX string to HTML using the katex CLI.

    Falls back to raw text wrapped in <code> if katex is unavailable or fails.
    """
    if _KATEX_BIN is None:
        tag = "div" if display_mode else "span"
        return f"<{tag} class=\"math-fallback\"><code>{tex}</code></{tag}>"

    cmd = [_KATEX_BIN]
    if display_mode:
        cmd.append("--display-mode")
    try:
        result = subprocess.run(
            cmd,
            input=tex,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            html = result.stdout.strip()
            if display_mode:
                return f'<div class="math-block">{html}</div>'
            return html
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Fallback
    tag = "div" if display_mode else "span"
    return f"<{tag} class=\"math-fallback\"><code>{tex}</code></{tag}>"


def process_math(text: str) -> str:
    """Replace $$...$$ and $...$ with server-side rendered KaTeX HTML.

    Must run BEFORE markdown conversion so that TeX special characters
    (backslashes, underscores, etc.) are not mangled.
    """
    # Block math first ($$...$$, possibly multiline)
    text = re.sub(
        r"\$\$(.*?)\$\$",
        lambda m: _render_katex(m.group(1).strip(), display_mode=True),
        text,
        flags=re.DOTALL,
    )
    # Inline math ($...$) – avoid matching $$ and be careful with currency etc.
    text = re.sub(
        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
        lambda m: _render_katex(m.group(1).strip(), display_mode=False),
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def resolve_images(html: str, base_dir: Path) -> str:
    """Convert relative image src paths to absolute file:// URIs."""
    def _fix(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith(("http://", "https://", "file://", "data:")):
            return m.group(0)
        abs_path = (base_dir / src).resolve()
        if abs_path.exists():
            return f'src="file://{abs_path}"'
        return m.group(0)  # leave as-is if not found

    return re.sub(r'src="([^"]+)"', _fix, html)


def polish_article_html(html: str) -> str:
    """Add lightweight wrappers so figures and tables render like an article."""
    html = re.sub(
        (
            r"<p>(<img[^>]+/?>)</p>\s*"
            r"<p><em>(Figure:.*?)</em></p>"
        ),
        (
            r'<figure class="figure-block">\1'
            r'<figcaption class="figure-caption">\2</figcaption></figure>'
        ),
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-wrap">\1</div>',
        html,
        flags=re.DOTALL,
    )
    return html


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def build_css() -> str:
    pygments_css = HtmlFormatter(style="monokai").get_style_defs(".codehilite")
    pygments_css += "\n.codehilite { background: #18202a !important; }"
    return dedent(f"""\
    @page {{
        size: A4;
        margin: 22mm 18mm 22mm 18mm;
        @bottom-center {{
            content: counter(page);
            font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
            font-size: 8.5pt;
            color: #7a8188;
        }}
    }}

    * {{
        box-sizing: border-box;
    }}

    html {{
        color: #17212b;
        background: #ffffff;
    }}

    body {{
        font-family: "Noto Serif CJK KR", "Noto Serif CJK SC", "Noto Serif", "Baekmuk Batang", serif;
        font-size: 10.35pt;
        line-height: 1.62;
        color: #17212b;
        max-width: 100%;
        word-wrap: break-word;
        overflow-wrap: break-word;
        letter-spacing: 0.002em;
    }}

    .journal-shell {{
        width: 100%;
    }}

    .article-body {{
        width: 100%;
    }}

    .article-body > :first-child {{
        margin-top: 0;
    }}

    /* ---------- Title page ---------- */
    .title-page {{
        page-break-after: always;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        min-height: 240mm;
        padding-top: 12mm;
        position: relative;
        border-top: 2px solid #17324d;
    }}
    .journal-kicker {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 8.8pt;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #51606f;
        margin-bottom: 14mm;
    }}
    .title-deck {{
        max-width: 150mm;
        margin-bottom: 10mm;
    }}
    .title-page h1 {{
        font-family: "EB Garamond", "Noto Serif CJK KR", serif;
        font-size: 27pt;
        line-height: 1.08;
        color: #102538;
        margin: 0 0 6mm 0;
        border: none;
        font-weight: 700;
    }}
    .title-page .deck-subtitle {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 10.2pt;
        line-height: 1.6;
        color: #5b6875;
        max-width: 138mm;
    }}
    .title-page .aliases {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 9pt;
        color: #74808b;
        margin-top: 4mm;
    }}
    .report-metadata {{
        margin-top: auto;
        padding-top: 10mm;
        border-top: 1px solid #d6dde4;
        display: grid;
        grid-template-columns: 1fr;
        row-gap: 3mm;
        max-width: 160mm;
    }}
    .report-metadata .meta-row {{
        display: flex;
        gap: 6mm;
        align-items: baseline;
    }}
    .report-metadata .meta-label {{
        min-width: 28mm;
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 8.5pt;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #73808b;
    }}
    .report-metadata .meta-value {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 10pt;
        color: #223243;
    }}
    .title-page .meta-tags {{
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }}
    .title-page .meta-chip {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        border: 1px solid #c8d1da;
        color: #435160;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 8.4pt;
        background: #fbfcfd;
    }}

    /* ---------- Headings ---------- */
    h1 {{
        font-family: "EB Garamond", "Noto Serif CJK KR", serif;
        font-size: 19pt;
        color: #102538;
        border-bottom: 1.5px solid #17324d;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 14px;
        page-break-after: avoid;
        font-weight: 700;
    }}
    h2 {{
        font-family: "EB Garamond", "Noto Serif CJK KR", serif;
        font-size: 15.5pt;
        color: #17324d;
        border-bottom: 1px solid #d3dbe3;
        padding-bottom: 4px;
        margin-top: 26px;
        margin-bottom: 10px;
        page-break-after: avoid;
        font-weight: 700;
    }}
    h3 {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 12pt;
        color: #17324d;
        margin-top: 21px;
        margin-bottom: 8px;
        page-break-after: avoid;
        letter-spacing: 0.01em;
    }}
    h4 {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 10.7pt;
        color: #31485e;
        margin-top: 18px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    h5, h6 {{
        font-family: "Noto Sans CJK KR", "Noto Sans", sans-serif;
        font-size: 10pt;
        color: #31485e;
        margin-top: 14px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}

    /* ---------- Paragraphs & lists ---------- */
    p {{
        margin: 7px 0 9px 0;
        text-align: justify;
    }}
    ul, ol {{
        margin: 8px 0 10px 0;
        padding-left: 22px;
    }}
    li {{
        margin: 4px 0;
    }}

    /* ---------- Links ---------- */
    a {{
        color: #21496f;
        text-decoration: none;
    }}

    /* ---------- Blockquotes (non-callout) ---------- */
    blockquote {{
        border-left: 2px solid #cfd6dc;
        padding: 8px 14px;
        margin: 14px 0;
        color: #465462;
        background: #fafbfc;
        page-break-inside: avoid;
    }}

    /* ---------- Code ---------- */
    code {{
        font-family: "DejaVu Sans Mono", "Noto Sans Mono", monospace;
        font-size: 8.6pt;
        background: #eef2f5;
        color: #213243;
        padding: 1px 4px;
        border-radius: 3px;
    }}
    pre {{
        background: #18202a;
        color: #d9e0e7;
        padding: 14px 16px;
        border-radius: 3px;
        overflow-x: auto;
        font-size: 8.5pt;
        line-height: 1.45;
        margin: 12px 0;
        page-break-inside: avoid;
    }}
    pre code {{
        background: none;
        padding: 0;
        color: inherit;
    }}
    .codehilite {{
        background: #1e1e1e;
        padding: 14px 16px;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 8.5pt;
        line-height: 1.45;
        margin: 12px 0;
        page-break-inside: avoid;
    }}
    .codehilite pre {{
        background: none;
        padding: 0;
        margin: 0;
        border-radius: 0;
    }}
    .codehilite code {{
        background: none;
        padding: 0;
    }}
    {pygments_css}

    /* ---------- Tables ---------- */
    .table-wrap {{
        margin: 16px 0;
        page-break-inside: avoid;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }}
    th {{
        background: #f2f5f8;
        color: #1b2d40;
        font-weight: 600;
        text-align: left;
        padding: 8px 10px;
        border-top: 1.2px solid #17324d;
        border-bottom: 1px solid #ccd5dd;
        border-left: none;
        border-right: none;
    }}
    td {{
        padding: 7px 10px;
        border-top: 1px solid #e4e9ee;
        border-bottom: 1px solid #e4e9ee;
        border-left: none;
        border-right: none;
    }}
    tr:nth-child(even) {{
        background: #fbfcfd;
    }}

    /* ---------- Horizontal rule ---------- */
    hr {{
        border: none;
        border-top: 1px solid #d6dde4;
        margin: 26px 0;
    }}

    /* ---------- Highlights ---------- */
    mark {{
        background: #f7efb4;
        padding: 1px 3px;
        border-radius: 2px;
    }}

    /* ---------- Callouts (base – additional inline styles per type) ---------- */
    .callout {{
        page-break-inside: avoid;
        border-radius: 2px !important;
        box-shadow: none !important;
    }}
    .callout-body p {{
        margin: 4px 0;
    }}
    .callout-body table {{
        font-size: 9pt;
    }}
    .callout-body pre,
    .callout-body .codehilite {{
        font-size: 8pt;
    }}

    /* ---------- Mermaid ---------- */
    .mermaid-diagram img {{
        max-width: 100%;
        max-height: 180mm;
        height: auto;
        object-fit: contain;
        border-radius: 4px;
    }}
    .mermaid-fallback {{
        background: #f0f0f0;
        border: 1px dashed #999;
        padding: 12px;
        color: #333;
        font-size: 9pt;
    }}

    /* ---------- Math (KaTeX) ---------- */
    .math-block {{
        text-align: center;
        margin: 16px 0;
        overflow-x: auto;
        page-break-inside: avoid;
    }}

    /* ---------- Images ---------- */
    img {{
        width: 100%;
        height: auto;
        display: block;
        margin: 12px auto;
    }}
    .figure-block {{
        margin: 18px 0 20px 0;
        page-break-inside: avoid;
    }}
    .figure-block img {{
        margin: 0 auto 8px auto;
        border: 1px solid #e5eaef;
    }}
    .figure-caption {{
        font-family: "Noto Serif CJK KR", "Noto Serif", serif;
        font-size: 8.9pt;
        line-height: 1.45;
        color: #51606d;
        font-style: italic;
        text-align: left;
    }}
    .article-body > ul:first-of-type,
    .article-body > ol:first-of-type {{
        margin-top: 10px;
    }}
    """)


# ---------------------------------------------------------------------------
# HTML document assembly
# ---------------------------------------------------------------------------

def build_title_page(meta: dict) -> str:
    title = meta.get("title", "Report")
    date = meta.get("date", "")
    tags = meta.get("tags", [])
    aliases = meta.get("aliases", [])
    kicker = meta.get("journal", "eLight-style technical report")

    tags_html = ""
    if tags:
        tags_html = '<div class="meta-tags">' + "".join(
            f'<span class="meta-chip">{t}</span>' for t in tags
        ) + "</div>"

    aliases_html = ""
    if aliases:
        aliases_html = (
            '<div class="aliases">'
            + " / ".join(aliases) + "</div>"
        )

    return dedent(f"""\
    <div class="title-page">
        <div class="journal-kicker">{kicker}</div>
        <div class="title-deck">
            <h1>{title}</h1>
            <div class="deck-subtitle">
                Fourier-space D2NN 재현, 실험 결과, 구현 차이, 그리고 물리적 해석을
                하나의 인쇄형 기술 문서로 통합한 PDF 에디션.
            </div>
            {aliases_html}
        </div>
        <div class="report-metadata">
            <div class="meta-row">
                <div class="meta-label">Date</div>
                <div class="meta-value">{date}</div>
            </div>
            <div class="meta-row">
                <div class="meta-label">Scope</div>
                <div class="meta-value">Reproduction, analysis, optical implementation notes</div>
            </div>
            <div class="meta-row">
                <div class="meta-label">Topics</div>
                <div class="meta-value">{tags_html}</div>
            </div>
        </div>
    </div>
    """)


def _get_katex_css() -> str:
    """Read KaTeX CSS from the local npm installation.

    Rewrites relative font paths to absolute file:// URIs so that
    WeasyPrint can locate the font files.
    """
    # Try common locations
    candidates = [
        Path("/root/.nvm/versions/node/v20.20.0/lib/node_modules/katex/dist/katex.min.css"),
    ]
    # Also try to locate via npm root
    try:
        result = subprocess.run(
            ["npm", "root", "-g"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            p = Path(result.stdout.strip()) / "katex" / "dist" / "katex.min.css"
            candidates.insert(0, p)
    except (OSError, subprocess.TimeoutExpired):
        pass

    for p in candidates:
        if p.exists():
            css_text = p.read_text(encoding="utf-8")
            # Rewrite relative font URLs to absolute file:// URIs
            fonts_dir = (p.parent / "fonts").resolve()
            css_text = css_text.replace("url(fonts/", f"url(file://{fonts_dir}/")
            return css_text
    return ""  # fallback: no KaTeX CSS (math will look unstyled)


def build_html(body_html: str, meta: dict, css: str) -> str:
    title = meta.get("title", "Report")
    title_page = build_title_page(meta)
    katex_css = _get_katex_css()

    return dedent(f"""\
    <!DOCTYPE html>
    <html lang="ko">
    <head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <style>
    /* --- KaTeX CSS (embedded) --- */
    {katex_css}
    /* --- Custom CSS --- */
    {css}
    </style>
    </head>
    <body>
    <div class="journal-shell">
    {title_page}
    <main class="article-body">
    {body_html}
    </main>
    </div>
    </body>
    </html>
    """)


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(input_path: Path, output_path: Path) -> None:
    print(f"[INFO] Reading {input_path}")
    text = input_path.read_text(encoding="utf-8")

    # 1. Parse frontmatter
    meta, body = parse_frontmatter(text)
    print(f"[INFO] Frontmatter: title={meta.get('title')}, tags={meta.get('tags')}")

    # 2. Pre-processing (order matters)
    with tempfile.TemporaryDirectory(prefix="obsidian2pdf_") as tmp:
        tmp_dir = Path(tmp)

        # 2a. Mermaid → SVG  (before callout processing so code fences are intact)
        body = process_mermaid(body, tmp_dir)

        # 2b. Callouts → HTML divs  (must be before markdown, since > conflicts)
        body = process_callouts(body)

        # 2c. Highlights
        body = process_highlights(body)

        # 2d. Math – protect from markdown munging
        body = process_math(body)

        # 3. Markdown → HTML
        print("[INFO] Converting markdown to HTML")
        md = markdown.Markdown(
            extensions=_MD_EXTENSIONS + ["toc"],
            extension_configs={
                **_MD_EXT_CONFIG,
                "toc": {"permalink": False},
            },
        )
        body_html = md.convert(body)

        # 4. Resolve image paths
        body_html = resolve_images(body_html, input_path.parent)
        body_html = polish_article_html(body_html)

        # 5. Build full HTML document
        css = build_css()
        full_html = build_html(body_html, meta, css)

        # Write intermediate HTML for debugging (optional)
        html_debug = tmp_dir / "debug.html"
        html_debug.write_text(full_html, encoding="utf-8")
        print(f"[INFO] Debug HTML at {html_debug}")

        # 6. HTML → PDF with WeasyPrint
        print(f"[INFO] Generating PDF → {output_path}")
        import weasyprint

        wp_html = weasyprint.HTML(
            string=full_html,
            base_url=str(input_path.parent),
        )
        wp_html.write_pdf(str(output_path))
        print(f"[INFO] Done. PDF saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1]).resolve()
    else:
        input_path = SCRIPT_DIR / "final_report_fd2nn.md"

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).resolve()
    else:
        output_path = input_path.with_suffix(".pdf")

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
