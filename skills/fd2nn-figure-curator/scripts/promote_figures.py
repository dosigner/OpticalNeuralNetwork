from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


DEFAULT_ROOT = Path("/root/dj/D2NN/kim2026")

RUN_DIRS = {
    "02": "02_fd2nn_spacing-sweep_loss-old_roi-1024_codex",
    "03": "03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex",
    "04": "04_fd2nn_spacing-sweep_loss-shape_roi-512_codex",
    "05": "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex",
}

PROMOTION_SPECS = {
    "02": {
        "fig1_epoch_curves.png": {
            "target": "codex02_spacing_epoch_curves.png",
            "role": "convergence",
            "default": True,
        },
        "fig2_test_metrics.png": {
            "target": "codex02_spacing_test_metrics.png",
            "role": "performance comparison",
            "default": True,
        },
        "fig3_field_full_comparison.png": {
            "target": "codex02_spacing_field_full.png",
            "role": "physical comparison",
            "default": True,
        },
        "fig4_field_zoom_comparison.png": {
            "target": "codex02_spacing_field_zoom.png",
            "role": "optional zoom detail",
            "default": False,
        },
        "fig5_field_profiles.png": {
            "target": "codex02_spacing_field_profiles.png",
            "role": "optional profile detail",
            "default": False,
        },
        "fig6_phase_masks.png": {
            "target": "codex02_spacing_phase_masks.png",
            "role": "phase-mask inspection",
            "default": True,
        },
    },
    "03": {
        "fig1_epoch_curves.png": {
            "target": "codex03_spacing_epoch_curves.png",
            "role": "convergence",
            "default": True,
        },
        "fig2_test_metrics.png": {
            "target": "codex03_spacing_test_metrics.png",
            "role": "performance comparison",
            "default": True,
        },
        "fig3_field_full_comparison.png": {
            "target": "codex03_spacing_field_full.png",
            "role": "physical comparison",
            "default": True,
        },
        "fig4_field_zoom_comparison.png": {
            "target": "codex03_spacing_field_zoom.png",
            "role": "optional zoom detail",
            "default": False,
        },
        "fig5_field_profiles.png": {
            "target": "codex03_spacing_field_profiles.png",
            "role": "optional profile detail",
            "default": False,
        },
        "fig6_phase_masks.png": {
            "target": "codex03_spacing_phase_masks.png",
            "role": "phase-mask inspection",
            "default": True,
        },
    },
    "04": {
        "fig1_epoch_curves.png": {
            "target": "codex04_spacing_epoch_curves.png",
            "role": "convergence",
            "default": True,
        },
        "fig2_test_metrics.png": {
            "target": "codex04_spacing_test_metrics.png",
            "role": "performance comparison",
            "default": True,
        },
        "fig3_field_full_comparison.png": {
            "target": "codex04_spacing_field_full.png",
            "role": "physical comparison",
            "default": True,
        },
        "fig4_field_zoom_comparison.png": {
            "target": "codex04_spacing_field_zoom.png",
            "role": "optional zoom detail",
            "default": False,
        },
        "fig5_field_profiles.png": {
            "target": "codex04_spacing_field_profiles.png",
            "role": "optional profile detail",
            "default": False,
        },
        "fig6_phase_masks.png": {
            "target": "codex04_spacing_phase_masks.png",
            "role": "phase-mask inspection",
            "default": True,
        },
        "fig7_1mm_three_way_comparison.png": {
            "target": "codex04_spacing_1mm_three_way.png",
            "role": "optional special comparison",
            "default": False,
        },
    },
    "05": {
        "fig1_epoch_curves.png": {
            "target": "codex05_roi_epoch_curves.png",
            "role": "convergence",
            "default": True,
        },
        "fig2_phase_metrics.png": {
            "target": "codex05_roi_phase_metrics.png",
            "role": "phase metrics",
            "default": True,
        },
        "fig3_field_comparison.png": {
            "target": "codex05_roi_field_comparison.png",
            "role": "field comparison",
            "default": True,
        },
        "fig4_support_and_leakage.png": {
            "target": "codex05_roi_support_leakage.png",
            "role": "support/leakage analysis",
            "default": True,
        },
        "fig5_phase_masks_raw_vs_wrapped.png": {
            "target": "codex05_roi_phase_masks_raw_vs_wrapped.png",
            "role": "phase-mask inspection",
            "default": True,
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_regular_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def selected_run_ids(runs: list[str] | None) -> list[str]:
    if runs is None:
        return list(RUN_DIRS)
    resolved = []
    for run_id in runs:
        if run_id not in RUN_DIRS:
            raise ValueError(f"unsupported run id: {run_id}")
        resolved.append(run_id)
    return resolved


def candidate_names(run_id: str, *, all_figures: bool) -> tuple[list[str], list[dict[str, str]]]:
    specs = PROMOTION_SPECS[run_id]
    selected = []
    skipped = []
    for source_name, meta in specs.items():
        if all_figures or meta["default"]:
            selected.append(source_name)
        else:
            skipped.append(
                {
                    "run_id": run_id,
                    "source_name": source_name,
                    "target_name": meta["target"],
                    "reason": "excluded_by_default_policy",
                }
            )
    return selected, skipped


def build_promotion_plan(root: str | Path, runs: list[str] | None = None, all_figures: bool = False) -> dict:
    root = Path(root)
    target_dir = root / "figures"
    run_ids = selected_run_ids(runs)
    items: list[dict] = []
    skipped: list[dict] = []

    for run_id in run_ids:
        run_dir = root / "runs" / RUN_DIRS[run_id] / "figures"
        selected_sources, skipped_items = candidate_names(run_id, all_figures=all_figures)
        skipped.extend(skipped_items)

        existing_sources = {path.name for path in run_dir.glob("*.png")} if run_dir.exists() else set()
        for source_name in sorted(existing_sources):
            if source_name not in PROMOTION_SPECS[run_id]:
                skipped.append(
                    {
                        "run_id": run_id,
                        "source_name": source_name,
                        "reason": "unmapped_source",
                    }
                )

        for source_name in selected_sources:
            meta = PROMOTION_SPECS[run_id][source_name]
            source_path = run_dir / source_name
            target_path = target_dir / meta["target"]

            if not is_regular_file(source_path):
                status = "missing_source"
                reason = "selected source path is missing or not a regular file"
            elif not target_path.exists():
                status = "planned"
                reason = "target file does not exist"
            elif not target_path.is_file():
                status = "collision"
                reason = "target path exists but is not a regular file"
            elif sha256(source_path) == sha256(target_path):
                status = "already_present"
                reason = "target already exists with identical bytes"
            else:
                status = "collision"
                reason = "target already exists with different bytes"

            items.append(
                {
                    "run_id": run_id,
                    "source": str(source_path),
                    "target": str(target_path),
                    "status": status,
                    "reason": reason,
                    "role": meta["role"],
                }
            )

    return {
        "root": str(root),
        "target_dir": str(target_dir),
        "runs": run_ids,
        "all_figures": all_figures,
        "items": items,
        "skipped": skipped,
    }


def apply_promotion_plan(plan: dict) -> dict:
    result = {
        "root": plan["root"],
        "target_dir": plan["target_dir"],
        "runs": list(plan["runs"]),
        "all_figures": plan["all_figures"],
        "items": [],
        "skipped": list(plan.get("skipped", [])),
    }

    Path(plan["target_dir"]).mkdir(parents=True, exist_ok=True)

    for item in plan["items"]:
        updated = dict(item)
        if item["status"] != "planned":
            result["items"].append(updated)
            continue

        source = Path(item["source"])
        target = Path(item["target"])
        if not is_regular_file(source):
            updated["status"] = "missing_source"
            updated["reason"] = "selected source path is missing or not a regular file"
            result["items"].append(updated)
            continue
        if target.exists():
            if target.is_file() and sha256(source) == sha256(target):
                updated["status"] = "already_present"
                updated["reason"] = "target already exists with identical bytes"
            else:
                updated["status"] = "collision"
                updated["reason"] = "target path is no longer safe to overwrite"
            result["items"].append(updated)
            continue
        shutil.copy2(source, target)
        updated["status"] = "copied"
        updated["reason"] = "copied from source to target"
        result["items"].append(updated)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote selected codex run figures into kim2026/figures.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="kim2026 project root")
    parser.add_argument("--runs", nargs="+", choices=sorted(RUN_DIRS), help="subset of runs to process")
    parser.add_argument("--all-figures", action="store_true", help="include optional figures for selected runs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the promotion plan without copying")
    mode.add_argument("--apply", action="store_true", help="copy planned files into the official figure store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_promotion_plan(args.root, runs=args.runs, all_figures=args.all_figures)
    if args.apply:
        output = apply_promotion_plan(plan)
    else:
        output = plan
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
