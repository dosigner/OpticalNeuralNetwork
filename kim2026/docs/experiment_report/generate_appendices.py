#!/usr/bin/env python3
"""Generate experiment report appendices from committed run artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "autoresearch" / "runs"
OUT_DIR = Path(__file__).resolve().parent

ATLAS_RUN_ORDER = [
    "0325-telescope-sweep-cn2-5e14-15cm",
    "0327-theorem-verify-defocus-1layer",
    "0328-co-sweep-strong-turb-cn2-5e14",
    "0329-paper-figures-static-d2nn",
    "0330-focal-pib-sweep-4loss-cn2-5e14",
    "0401-datagen-dn100um-lanczos50",
    "0401-focal-pib-sweep-clean-4loss-cn2-5e14",
    "0402-focal-new-losses-pitchrescale-3strat-cn2-5e14",
    "0403-combined-6strat-pitchrescale-cn2-5e14",
    "0403-focal-pib-vacuum-target-pitchrescale-3strat-cn2-5e14",
    "0405-distance-sweep-rawrp-f6p5mm",
    "0405-fd2nn-4f-sweep-pitchrescale",
    "0405-mplc-figs",
    "0406-rp-seminar-pptx",
]

RESULT_GROUPS = [
    ("Beam reducer sweep", RUNS_DIR / "0325-telescope-sweep-cn2-5e14-15cm"),
    ("Strong turbulence support sweep", RUNS_DIR / "0328-co-sweep-strong-turb-cn2-5e14"),
    ("Legacy focal PIB sweep", RUNS_DIR / "0330-focal-pib-sweep-4loss-cn2-5e14"),
    ("Clean 4-loss sweep", RUNS_DIR / "0401-focal-pib-sweep-clean-4loss-cn2-5e14"),
    ("TP-preserving losses", RUNS_DIR / "0402-focal-new-losses-pitchrescale-3strat-cn2-5e14"),
    ("Combined loss optimization", RUNS_DIR / "0403-combined-6strat-pitchrescale-cn2-5e14"),
    ("Vacuum-target exploratory sweep", RUNS_DIR / "0403-focal-pib-vacuum-target-pitchrescale-3strat-cn2-5e14"),
    ("Distance sweep", RUNS_DIR / "0405-distance-sweep-rawrp-f6p5mm"),
    ("FD2NN 4f sweep", RUNS_DIR / "0405-fd2nn-4f-sweep-pitchrescale"),
]

SUMMARY_FILES = [
    RUNS_DIR / "0325-telescope-sweep-cn2-5e14-15cm" / "summary.json",
    RUNS_DIR / "0328-co-sweep-strong-turb-cn2-5e14" / "summary.json",
    RUNS_DIR / "0401-focal-pib-sweep-clean-4loss-cn2-5e14" / "summary.json",
    RUNS_DIR / "0401-focal-pib-sweep-clean-4loss-cn2-5e14" / "16_throughput_summary.json",
    RUNS_DIR / "0401-focal-pib-sweep-clean-4loss-cn2-5e14" / "17_received_power_summary.json",
    RUNS_DIR / "0402-focal-new-losses-pitchrescale-3strat-cn2-5e14" / "summary.json",
    RUNS_DIR / "0402-focal-new-losses-pitchrescale-3strat-cn2-5e14" / "17_received_power_summary.json",
    RUNS_DIR / "0402-focal-new-losses-pitchrescale-3strat-cn2-5e14" / "absolute_power_comparison.json",
    RUNS_DIR / "0403-combined-6strat-pitchrescale-cn2-5e14" / "19_cross_strategy_summary.json",
    RUNS_DIR / "0405-distance-sweep-rawrp-f6p5mm" / "distance_sweep_summary.json",
    RUNS_DIR / "0405-distance-sweep-rawrp-f6p5mm" / "22_cross_distance_3km_model" / "cross_distance_summary.json",
    RUNS_DIR / "0405-fd2nn-4f-sweep-pitchrescale" / "summary.json",
    RUNS_DIR / "theorem_verification_results.json",
]


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(v, (dict, list)) for v in value):
            return "[" + ", ".join(format_value(v) for v in value) + "]"
        return f"<array len={len(value)}>"
    if isinstance(value, dict):
        return f"<object keys={len(value)}>"
    return str(value)


def flatten_json(obj, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                rows.extend(flatten_json(value, next_prefix))
            elif isinstance(value, list) and value and any(isinstance(v, (dict, list)) for v in value):
                rows.append((next_prefix, format_value(value)))
            else:
                rows.append((next_prefix, format_value(value)))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}.item{idx}" if prefix else f"item{idx}"
            if isinstance(value, (dict, list)):
                rows.extend(flatten_json(value, next_prefix))
            else:
                rows.append((next_prefix, format_value(value)))
    else:
        rows.append((prefix or "value", format_value(obj)))
    return rows


def relative_to_runs(path: Path) -> str:
    return str(path.relative_to(RUNS_DIR)).replace("\\", "/")


def write_appendix_full_results() -> None:
    out = []
    out.append(r"\chapter{Full Numerical Results}")
    out.append("")
    out.append(
        "This appendix records the committed machine-readable results used throughout the report. "
        "Scalar fields are expanded directly; long vectors and nested histories are summarized by length."
    )
    out.append("")
    for title, group_dir in RESULT_GROUPS:
        out.append(rf"\section{{{tex_escape(title)}}}")
        result_files = sorted(group_dir.rglob("results.json"))
        for result_file in result_files:
            rel = result_file.relative_to(ROOT)
            data = json.loads(result_file.read_text())
            rows = flatten_json(data)
            subtitle = tex_escape(str(rel).replace("\\", "/"))
            out.append(rf"\subsection{{{subtitle}}}")
            out.append(r"\begin{longtable}{p{0.42\textwidth}p{0.5\textwidth}}")
            out.append(r"\toprule")
            out.append(r"\smalltablehead{Field} & \smalltablehead{Value}\\")
            out.append(r"\midrule")
            out.append(r"\endfirsthead")
            out.append(r"\toprule")
            out.append(r"\smalltablehead{Field} & \smalltablehead{Value}\\")
            out.append(r"\midrule")
            out.append(r"\endhead")
            out.append(r"\midrule")
            out.append(r"\multicolumn{2}{r}{\small Continued on next page}\\")
            out.append(r"\endfoot")
            out.append(r"\bottomrule")
            out.append(r"\endlastfoot")
            for key, value in rows:
                out.append(f"{tex_escape(key)} & {tex_escape(value)}\\\\")
            out.append(r"\end{longtable}")
            out.append("")
        summary_candidates = [p for p in SUMMARY_FILES if group_dir in p.parents]
        if summary_candidates:
            out.append(r"\subsection{Group summaries}")
            for summary_file in summary_candidates:
                out.append(rf"\paragraph{{{tex_escape(relative_to_runs(summary_file))}}}")
                data = json.loads(summary_file.read_text())
                rows = flatten_json(data)
                out.append(r"\begin{longtable}{p{0.42\textwidth}p{0.5\textwidth}}")
                out.append(r"\toprule")
                out.append(r"\smalltablehead{Field} & \smalltablehead{Value}\\")
                out.append(r"\midrule")
                out.append(r"\endfirsthead")
                out.append(r"\toprule")
                out.append(r"\smalltablehead{Field} & \smalltablehead{Value}\\")
                out.append(r"\midrule")
                out.append(r"\endhead")
                out.append(r"\midrule")
                out.append(r"\multicolumn{2}{r}{\small Continued on next page}\\")
                out.append(r"\endfoot")
                out.append(r"\bottomrule")
                out.append(r"\endlastfoot")
                for key, value in rows:
                    out.append(f"{tex_escape(key)} & {tex_escape(value)}\\\\")
                out.append(r"\end{longtable}")
                out.append("")
    (OUT_DIR / "appendix_full_results.tex").write_text("\n".join(out) + "\n")


def write_appendix_figure_atlas() -> None:
    out = []
    out.append(r"\chapter{Full Figure Atlas}")
    out.append("")
    out.append(
        "This appendix collects every committed PNG artifact under "
        r"\texttt{kim2026/autoresearch/runs}. "
        "Core chronology runs appear first, followed by support material, comparison figures, and derived presentation renders."
    )
    out.append("")
    for run_name in ATLAS_RUN_ORDER:
        run_dir = RUNS_DIR / run_name
        images = sorted(
            run_dir.rglob("*.png"),
            key=lambda p: (len(p.relative_to(RUNS_DIR).parts), str(p.relative_to(RUNS_DIR))),
        )
        if not images:
            continue
        out.append(rf"\section{{Run {tex_escape(run_name)}}}")
        subgroup = None
        for img in images:
            rel = img.relative_to(RUNS_DIR)
            parts = rel.parts
            current_group = "/".join(parts[:-1]) if len(parts) > 1 else run_name
            if current_group != subgroup:
                subgroup = current_group
                out.append(rf"\subsection{{{tex_escape(current_group)}}}")
            out.append(r"\clearpage")
            out.append(r"\begin{center}")
            out.append(rf"\includegraphics[width=0.92\linewidth]{{{str(rel).replace(chr(92), '/')}}}")
            out.append(r"\par\medskip")
            out.append(rf"\small\texttt{{{tex_escape(str(rel).replace(chr(92), '/'))}}}")
            out.append(r"\end{center}")
            out.append("")
    (OUT_DIR / "appendix_figure_atlas.tex").write_text("\n".join(out) + "\n")


def main() -> None:
    write_appendix_full_results()
    write_appendix_figure_atlas()


if __name__ == "__main__":
    main()
