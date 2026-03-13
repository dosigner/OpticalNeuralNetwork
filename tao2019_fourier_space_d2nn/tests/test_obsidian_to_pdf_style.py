import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    markdown_stub = types.ModuleType("markdown")
    markdown_stub.markdown = lambda text, **kwargs: text
    markdown_stub.Markdown = lambda *args, **kwargs: None
    sys.modules.setdefault("markdown", markdown_stub)

    markdown_extensions = types.ModuleType("markdown.extensions")
    sys.modules.setdefault("markdown.extensions", markdown_extensions)

    codehilite_stub = types.ModuleType("markdown.extensions.codehilite")
    codehilite_stub.CodeHiliteExtension = object
    sys.modules.setdefault("markdown.extensions.codehilite", codehilite_stub)

    pygments_stub = types.ModuleType("pygments")
    sys.modules.setdefault("pygments", pygments_stub)

    pygments_formatters = types.ModuleType("pygments.formatters")

    class _HtmlFormatter:
        def __init__(self, *args, **kwargs):
            pass

        def get_style_defs(self, selector):
            return f"{selector} {{}}"

    pygments_formatters.HtmlFormatter = _HtmlFormatter
    sys.modules.setdefault("pygments.formatters", pygments_formatters)

    script_path = (
        Path(__file__).resolve().parents[1] / "reports" / "obsidian_to_pdf.py"
    )
    spec = importlib.util.spec_from_file_location("obsidian_to_pdf", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_css_contains_journal_shell_and_serif_body():
    module = _load_module()

    css = module.build_css()

    assert 'font-family: "Noto Serif CJK KR"' in css
    assert ".journal-shell" in css
    assert ".article-body" in css
    assert ".figure-caption" in css
    assert ".journal-kicker" in css


def test_build_title_page_emits_editorial_structure():
    module = _load_module()

    html = module.build_title_page(
        {
            "title": "F-D2NN 최종 보고서 (PDF Edition)",
            "date": "2026-03-11",
            "tags": ["d2nn", "fourier-optics"],
            "aliases": ["FD2NN Report PDF"],
        }
    )

    assert 'class="title-page"' in html
    assert 'class="journal-kicker"' in html
    assert 'class="title-deck"' in html
    assert 'class="report-metadata"' in html
    assert 'class="meta-chip"' in html
