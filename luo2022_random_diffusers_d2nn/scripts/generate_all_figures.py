"""Generate all figures from trained checkpoints.

Usage:
    python scripts/generate_all_figures.py [--runs-root runs] [--output-dir figures]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate all figures")
    parser.add_argument("--runs-root", default="runs", type=str)
    parser.add_argument("--output-dir", default="figures", type=str)
    parser.add_argument("--config", default="configs/baseline.yaml", type=str)
    parser.add_argument("--compare-runs-root", default=None, type=str,
                        help="Second runs root for phase comparison (e.g. runs for B=64)")
    parser.add_argument("--compare-label", default=None, type=str,
                        help="Label for the comparison (e.g. 'B=64 (100ep)')")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint paths
    ckpt_file = runs_root / "checkpoints.json"
    if ckpt_file.exists():
        checkpoints = json.loads(ckpt_file.read_text())
    else:
        # Auto-discover
        checkpoints = {}
        for d in runs_root.iterdir():
            if d.is_dir() and (d / "model.pt").exists():
                checkpoints[d.name] = str(d / "model.pt")
        logger.info("Auto-discovered checkpoints: %s", list(checkpoints.keys()))

    # ----- Fig 1a: Schematic (no checkpoint needed) -----
    logger.info("Generating Fig 1a...")
    from luo2022_d2nn.figures.fig1a_schematic import make_fig1a
    make_fig1a(save_path=str(output_dir / "fig1a_schematic.png"))
    logger.info("  Saved fig1a_schematic.png")

    # ----- Fig 1b: Distortion comparison -----
    baseline_ckpt = checkpoints.get("n20_L4")
    if baseline_ckpt:
        logger.info("Generating Fig 1b...")
        from luo2022_d2nn.figures.fig1b_distortion import make_fig1b
        make_fig1b(
            checkpoint_path=baseline_ckpt,
            config_path=args.config,
            save_path=str(output_dir / "fig1b_distortion.png"),
        )
        logger.info("  Saved fig1b_distortion.png")
    else:
        logger.warning("Skipping Fig 1b: n20_L4 checkpoint not found")

    # ----- Fig 2: Known vs New diffuser -----
    if baseline_ckpt:
        logger.info("Generating Fig 2...")
        from luo2022_d2nn.figures.fig2_known_new import make_fig2
        make_fig2(
            checkpoint_path=baseline_ckpt,
            config_path=args.config,
            save_path=str(output_dir / "fig2_known_new.png"),
        )
        logger.info("  Saved fig2_known_new.png")
    else:
        logger.warning("Skipping Fig 2: n20_L4 checkpoint not found")

    # ----- Supp. Fig. S3: Overlap map of phase islands -----
    if baseline_ckpt:
        logger.info("Generating Supplementary Fig. S3...")
        from luo2022_d2nn.figures.figs3_overlap_map import make_figs3
        make_figs3(
            checkpoint_path=baseline_ckpt,
            config_path=args.config,
            save_path=str(output_dir / "figS3_overlap_map.png"),
        )
        logger.info("  Saved figS3_overlap_map.png")
    else:
        logger.warning("Skipping Supplementary Fig. S3: n20_L4 checkpoint not found")

    # ----- Supp. Fig. S4: pruning-condition comparison -----
    if baseline_ckpt:
        logger.info("Generating Supplementary Fig. S4...")
        from luo2022_d2nn.figures.figs4_pruning import make_figs4
        make_figs4(
            checkpoint_path=baseline_ckpt,
            config_path=args.config,
            save_path=str(output_dir / "figS4_pruning.png"),
        )
        logger.info("  Saved figS4_pruning.png")
    else:
        logger.warning("Skipping Supplementary Fig. S4: n20_L4 checkpoint not found")

    # ----- Supp. Fig. S1: Layer phase patterns -----
    if baseline_ckpt:
        logger.info("Generating Supplementary Fig. S1...")
        from luo2022_d2nn.figures.figs1_layer_phases import make_figs1
        make_figs1(
            checkpoint_path=baseline_ckpt,
            config_path=args.config,
            save_path=str(output_dir / "figS1_layer_phases.png"),
        )
        logger.info("  Saved figS1_layer_phases.png")
    else:
        logger.warning("Skipping Supplementary Fig. S1: n20_L4 checkpoint not found")

    # ----- Fig 3: Period sweep -----
    n_ckpts = {}
    for n in [1, 10, 15, 20]:
        key = f"n{n}_L4"
        if key in checkpoints:
            n_ckpts[n] = checkpoints[key]
    if n_ckpts:
        logger.info("Generating Fig 3 with n=%s...", list(n_ckpts.keys()))
        from luo2022_d2nn.figures.fig3_period_sweep import make_fig3
        make_fig3(
            checkpoint_paths=n_ckpts,
            config_path=args.config,
            save_path=str(output_dir / "fig3_period_sweep.png"),
        )
        logger.info("  Saved fig3_period_sweep.png")
    else:
        logger.warning("Skipping Fig 3: no 4-layer checkpoints found")

    # ----- Fig 5: Memory -----
    if n_ckpts:
        logger.info("Generating Fig 5...")
        from luo2022_d2nn.figures.fig5_memory import make_fig5
        make_fig5(
            checkpoint_paths=n_ckpts,
            config_path=args.config,
            save_path=str(output_dir / "fig5_memory.png"),
        )
        logger.info("  Saved fig5_memory.png")
    else:
        logger.warning("Skipping Fig 5: no 4-layer checkpoints found")

    # ----- Fig 6: Conditions -----
    if n_ckpts:
        logger.info("Generating Fig 6...")
        from luo2022_d2nn.figures.fig6_conditions import make_fig6
        make_fig6(
            checkpoint_paths=n_ckpts,
            config_path=args.config,
            save_path=str(output_dir / "fig6_conditions.png"),
        )
        logger.info("  Saved fig6_conditions.png")
    else:
        logger.warning("Skipping Fig 6: no 4-layer checkpoints found")

    # ----- Supp. Fig. S5: Correlation length test -----
    if n_ckpts:
        logger.info("Generating Supplementary Fig. S5...")
        from luo2022_d2nn.figures.figs5_corr_length import make_figs5
        s5_ckpts = {f"n={n}": path for n, path in n_ckpts.items()}
        make_figs5(
            checkpoint_paths=s5_ckpts,
            config_path=args.config,
            save_path=str(output_dir / "figS5_corr_length.png"),
        )
        logger.info("  Saved figS5_corr_length.png")
    else:
        logger.warning("Skipping Supplementary Fig. S5: no 4-layer checkpoints found")

    # ----- Fig 7: Depth advantage -----
    depth_ckpts = {}
    for layers in [2, 4, 5]:
        for n in [1, 10, 15, 20]:
            key = f"n{n}_L{layers}"
            if key in checkpoints:
                depth_ckpts[(layers, n)] = checkpoints[key]
    if depth_ckpts:
        logger.info("Generating Fig 7 with configs=%s...", list(depth_ckpts.keys()))
        from luo2022_d2nn.figures.fig7_depth import make_fig7
        make_fig7(
            checkpoint_paths=depth_ckpts,
            config_path=args.config,
            save_path=str(output_dir / "fig7_depth.png"),
        )
        logger.info("  Saved fig7_depth.png")
    else:
        logger.warning("Skipping Fig 7: no depth-sweep checkpoints found")

    # ----- Phase comparison (optional) -----
    if args.compare_runs_root and baseline_ckpt:
        compare_root = Path(args.compare_runs_root)
        compare_ckpt = compare_root / "n20_L4" / "model.pt"
        if compare_ckpt.exists():
            logger.info("Generating phase comparison...")
            from luo2022_d2nn.figures.figs1_layer_phases import make_figs1_comparison
            current_label = f"{runs_root.name}"
            compare_label = args.compare_label or f"{compare_root.name}"
            make_figs1_comparison(
                checkpoint_paths={
                    compare_label: str(compare_ckpt),
                    current_label: baseline_ckpt,
                },
                config_path=args.config,
                save_path=str(output_dir / "figS1_phase_comparison.png"),
            )
            logger.info("  Saved figS1_phase_comparison.png")
        else:
            logger.warning("Comparison checkpoint not found: %s", compare_ckpt)

    logger.info("=" * 60)
    logger.info("Figure generation complete! Output: %s", output_dir)


if __name__ == "__main__":
    main()
