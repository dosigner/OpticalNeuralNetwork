from __future__ import annotations

import re

from kim2026.viz.paper_figure_text import FIGURE_TEXTS


HANGUL_RE = re.compile(r"[\u3131-\u318E\uAC00-\uD7A3]")


def _flatten(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)


def test_fig3_to_fig5_text_is_english_only() -> None:
    for key in ("fig2", "fig3", "fig4", "fig5", "fig6"):
        flattened = list(_flatten(FIGURE_TEXTS[key]))
        assert flattened
        for text in flattened:
            assert not HANGUL_RE.search(text), f"Hangul remains in {key}: {text}"


def test_fig3_contains_expected_english_titles() -> None:
    fig3 = FIGURE_TEXTS["fig3"]
    assert fig3["suptitle"] == "Figure 3: Deterministic Aberration Correction vs Random Turbulence"
    assert "Defocus Z4" in fig3["left_title"]
    assert "Random turbulence" in fig3["right_title"]


def test_fig5_contains_english_region_labels() -> None:
    fig5 = FIGURE_TEXTS["fig5"]
    assert "Ideal region" in fig5["ideal_region"]
    assert "PIB up" in fig5["filtering_region"]


def test_fig2_and_fig6_have_english_headers() -> None:
    fig2 = FIGURE_TEXTS["fig2"]
    assert "Figure 2" in fig2["suptitle"]
    assert "Theorem 1" in fig2["titles"][0]
    assert "Theorem 2" in fig2["titles"][1]

    fig6 = FIGURE_TEXTS["fig6"]
    assert "Figure 6" in fig6["suptitle"]
    assert "Vacuum" in fig6["field_labels"][0]
    assert "Difference map" in fig6["row_labels"][2]
