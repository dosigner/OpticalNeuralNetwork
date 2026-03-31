from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "fd2nn-figure-curator"
    / "scripts"
    / "promote_figures.py"
)


def load_module():
    assert SCRIPT_PATH.exists()
    spec = importlib.util.spec_from_file_location("promote_figures", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def touch_png(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def make_run_figures(root: Path, run_id: str, names: list[str]) -> None:
    run_dirs = {
        "02": "02_fd2nn_spacing-sweep_loss-old_roi-1024_codex",
        "03": "03_fd2nn_spacing-sweep_loss-shape_roi-1024_codex",
        "04": "04_fd2nn_spacing-sweep_loss-shape_roi-512_codex",
        "05": "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex",
    }
    fig_dir = root / "runs" / run_dirs[run_id] / "figures"
    for idx, name in enumerate(names):
        touch_png(fig_dir / name, f"{run_id}:{idx}:{name}".encode("utf-8"))


def test_default_dry_run_selects_core_subset_only(tmp_path: Path) -> None:
    make_run_figures(
        tmp_path,
        "02",
        [
            "fig1_epoch_curves.png",
            "fig2_test_metrics.png",
            "fig3_field_full_comparison.png",
            "fig4_field_zoom_comparison.png",
            "fig5_field_profiles.png",
            "fig6_phase_masks.png",
        ],
    )
    make_run_figures(
        tmp_path,
        "04",
        [
            "fig1_epoch_curves.png",
            "fig2_test_metrics.png",
            "fig3_field_full_comparison.png",
            "fig4_field_zoom_comparison.png",
            "fig5_field_profiles.png",
            "fig6_phase_masks.png",
            "fig7_1mm_three_way_comparison.png",
        ],
    )
    make_run_figures(
        tmp_path,
        "05",
        [
            "fig1_epoch_curves.png",
            "fig2_phase_metrics.png",
            "fig3_field_comparison.png",
            "fig4_support_and_leakage.png",
            "fig5_phase_masks_raw_vs_wrapped.png",
        ],
    )

    mod = load_module()
    plan = mod.build_promotion_plan(tmp_path)
    targets = [Path(item["target"]).name for item in plan["items"] if item["status"] == "planned"]

    assert "codex02_spacing_epoch_curves.png" in targets
    assert "codex02_spacing_test_metrics.png" in targets
    assert "codex02_spacing_field_full.png" in targets
    assert "codex02_spacing_phase_masks.png" in targets
    assert "codex05_roi_epoch_curves.png" in targets
    assert "codex05_roi_phase_metrics.png" in targets
    assert "codex05_roi_field_comparison.png" in targets
    assert "codex05_roi_support_leakage.png" in targets
    assert "codex05_roi_phase_masks_raw_vs_wrapped.png" in targets

    assert "codex02_spacing_field_zoom.png" not in targets
    assert "codex02_spacing_field_profiles.png" not in targets
    assert "codex04_spacing_1mm_three_way.png" not in targets


def test_all_figures_includes_optional_promotions(tmp_path: Path) -> None:
    make_run_figures(
        tmp_path,
        "04",
        [
            "fig1_epoch_curves.png",
            "fig2_test_metrics.png",
            "fig3_field_full_comparison.png",
            "fig4_field_zoom_comparison.png",
            "fig5_field_profiles.png",
            "fig6_phase_masks.png",
            "fig7_1mm_three_way_comparison.png",
        ],
    )

    mod = load_module()
    plan = mod.build_promotion_plan(tmp_path, runs=["04"], all_figures=True)
    targets = {Path(item["target"]).name for item in plan["items"] if item["status"] == "planned"}

    assert "codex04_spacing_field_zoom.png" in targets
    assert "codex04_spacing_field_profiles.png" in targets
    assert "codex04_spacing_1mm_three_way.png" in targets


def test_collision_and_already_present_are_distinguished(tmp_path: Path) -> None:
    make_run_figures(tmp_path, "05", ["fig2_phase_metrics.png", "fig3_field_comparison.png"])

    target_dir = tmp_path / "figures"
    source_phase_metrics = (
        tmp_path
        / "runs"
        / "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex"
        / "figures"
        / "fig2_phase_metrics.png"
    )
    source_field = (
        tmp_path
        / "runs"
        / "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex"
        / "figures"
        / "fig3_field_comparison.png"
    )
    touch_png(target_dir / "codex05_roi_phase_metrics.png", source_phase_metrics.read_bytes())
    touch_png(target_dir / "codex05_roi_field_comparison.png", b"different-content")

    mod = load_module()
    plan = mod.build_promotion_plan(tmp_path, runs=["05"])
    statuses = {Path(item["target"]).name: item["status"] for item in plan["items"]}

    assert statuses["codex05_roi_phase_metrics.png"] == "already_present"
    assert statuses["codex05_roi_field_comparison.png"] == "collision"
    assert source_field.read_bytes() == b"05:1:fig3_field_comparison.png"


def test_apply_copies_only_planned_files_and_preserves_sources(tmp_path: Path) -> None:
    make_run_figures(
        tmp_path,
        "02",
        [
            "fig1_epoch_curves.png",
            "fig2_test_metrics.png",
            "fig3_field_full_comparison.png",
            "fig6_phase_masks.png",
        ],
    )
    src = (
        tmp_path
        / "runs"
        / "02_fd2nn_spacing-sweep_loss-old_roi-1024_codex"
        / "figures"
        / "fig1_epoch_curves.png"
    )
    src_before = src.read_bytes()

    mod = load_module()
    plan = mod.build_promotion_plan(tmp_path, runs=["02"])
    result = mod.apply_promotion_plan(plan)

    copied = {Path(item["target"]).name for item in result["items"] if item["status"] == "copied"}
    assert "codex02_spacing_epoch_curves.png" in copied
    assert "codex02_spacing_test_metrics.png" in copied
    assert src.read_bytes() == src_before
    assert (tmp_path / "figures" / "codex02_spacing_epoch_curves.png").read_bytes() == src_before


def test_non_file_paths_do_not_abort_plan(tmp_path: Path) -> None:
    make_run_figures(tmp_path, "05", ["fig2_phase_metrics.png", "fig3_field_comparison.png"])

    target_dir = tmp_path / "figures"
    (target_dir / "codex05_roi_phase_metrics.png").mkdir(parents=True)

    source_as_dir = (
        tmp_path
        / "runs"
        / "05_fd2nn_roi-sweep_loss-phase-first_spacing-0p1mm_codex"
        / "figures"
        / "fig3_field_comparison.png"
    )
    source_as_dir.unlink()
    source_as_dir.mkdir()

    mod = load_module()
    plan = mod.build_promotion_plan(tmp_path, runs=["05"])
    statuses = {Path(item["target"]).name: item["status"] for item in plan["items"]}

    assert statuses["codex05_roi_phase_metrics.png"] == "collision"
    assert statuses["codex05_roi_field_comparison.png"] == "missing_source"


def test_skill_instructions_and_namespaces_are_pinned() -> None:
    skill_md = Path(__file__).resolve().parents[2] / "skills" / "fd2nn-figure-curator" / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")

    assert "/root/dj/D2NN/kim2026/figures" in text
    assert "copy only" in text
    assert "codexXX_*" in text
