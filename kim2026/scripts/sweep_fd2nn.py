#!/usr/bin/env python
"""FD2NN hyperparameter sweep: Cn2 x layer_spacing.

Usage:
    python scripts/sweep_fd2nn.py                    # full sweep (12 runs)
    python scripts/sweep_fd2nn.py --dry-run           # generate configs only
    python scripts/sweep_fd2nn.py --generate-only     # generate pairs only
    python scripts/sweep_fd2nn.py --train-only        # train only (pairs must exist)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

# Sweep axes
CN2_VALUES = [1e-15, 5e-15, 1e-14]
LAYER_SPACING_VALUES = [1e-6, 3e-6, 5e-6, 10e-6]

BASE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "sweep_base_fd2nn.yaml"
SWEEP_DIR = Path(__file__).resolve().parent.parent / "runs" / "sweep_fd2nn"


def make_run_id(cn2: float, spacing: float) -> str:
    cn2_str = f"cn2_{cn2:.0e}".replace("+", "").replace("-", "m")
    sp_str = f"sp_{spacing * 1e6:.0f}um"
    return f"{cn2_str}_{sp_str}"


def generate_configs() -> list[tuple[str, dict]]:
    """Generate all sweep configs from base."""
    with open(BASE_CONFIG, "r") as f:
        base = yaml.safe_load(f)

    configs = []
    for cn2 in CN2_VALUES:
        for spacing in LAYER_SPACING_VALUES:
            run_id = make_run_id(cn2, spacing)
            cfg = copy.deepcopy(base)
            cfg["experiment"]["id"] = run_id
            cfg["experiment"]["save_dir"] = str(SWEEP_DIR / run_id)
            cfg["channel"]["cn2"] = cn2
            cfg["model"]["layer_spacing_m"] = spacing
            cfg["data"]["cache_dir"] = str(SWEEP_DIR / "cache" / f"cn2_{cn2:.0e}")
            cfg["data"]["split_manifest_path"] = str(
                SWEEP_DIR / "cache" / f"cn2_{cn2:.0e}" / "split_manifest.json"
            )
            cfg["visualization"]["output_dir"] = str(SWEEP_DIR / run_id / "figures")
            configs.append((run_id, cfg))
    return configs


def save_configs(configs: list[tuple[str, dict]]) -> list[Path]:
    """Save each config to a YAML file."""
    config_dir = SWEEP_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_id, cfg in configs:
        path = config_dir / f"{run_id}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        paths.append(path)
    return paths


def run_sweep(*, dry_run: bool = False, generate_only: bool = False, train_only: bool = False) -> None:
    configs = generate_configs()
    config_paths = save_configs(configs)

    print(f"Generated {len(configs)} sweep configs:")
    for run_id, cfg in configs:
        cn2 = cfg["channel"]["cn2"]
        sp = cfg["model"]["layer_spacing_m"]
        nf = (2e-6) ** 2 / (1.55e-6 * sp)
        print(f"  {run_id}: Cn2={cn2:.0e}, spacing={sp*1e6:.0f}um, N_F={nf:.2f}")

    if dry_run:
        print("\n--dry-run: configs saved, no training.")
        return

    # Import here to avoid loading torch for dry-run
    from kim2026.config.schema import validate_config
    from kim2026.training.trainer import train_model
    from kim2026.utils.seed import set_global_seed

    # Group by Cn2 for pair generation (same Cn2 = same pairs)
    cn2_groups: dict[float, list[tuple[str, dict, Path]]] = {}
    for (run_id, cfg), path in zip(configs, config_paths):
        cn2 = cfg["channel"]["cn2"]
        cn2_groups.setdefault(cn2, []).append((run_id, cfg, path))

    if not train_only:
        print("\n=== Phase 1: Pair Generation ===")
        for cn2, group in cn2_groups.items():
            _, cfg, _ = group[0]
            validated = validate_config(cfg)
            cache_dir = Path(validated["data"]["cache_dir"])
            manifest = Path(validated["data"]["split_manifest_path"])
            if manifest.exists():
                print(f"  Cn2={cn2:.0e}: pairs already exist, skipping.")
                continue
            print(f"  Cn2={cn2:.0e}: generating pairs...")
            # Use frozen_flow with frames_per_episode=1 as static equivalent
            # (full static pair gen not yet implemented)
            print(f"    [TODO] Pair generation requires GPU. Skipping in script.")
            print(f"    Run: kim2026-generate-pairs --config {group[0][2]}")

    if generate_only:
        print("\n--generate-only: pair generation phase complete.")
        return

    print(f"\n=== Phase 2: Training ({len(configs)} runs) ===")
    results = {}
    for i, (run_id, cfg, config_path) in enumerate(
        [(rid, c, p) for (rid, c), p in zip(configs, config_paths)]
    ):
        print(f"\n--- Run {i+1}/{len(configs)}: {run_id} ---")
        validated = validate_config(cfg)
        run_dir = Path(validated["experiment"]["save_dir"])
        checkpoint = run_dir / "checkpoint.pt"
        if checkpoint.exists():
            print(f"  Checkpoint exists, skipping.")
            results[run_id] = {"status": "skipped (checkpoint exists)"}
            continue
        try:
            set_global_seed(int(validated["runtime"]["seed"]))
            result = train_model(validated, run_dir=run_dir)
            final_loss = result["history"][-1] if result["history"] else {}
            results[run_id] = {"status": "completed", **final_loss}
            print(f"  Completed. Final loss: {final_loss}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[run_id] = {"status": f"failed: {e}"}

    # Save sweep results
    results_path = SWEEP_DIR / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only")
    parser.add_argument("--generate-only", action="store_true", help="Generate pairs only")
    parser.add_argument("--train-only", action="store_true", help="Train only (pairs must exist)")
    args = parser.parse_args()
    run_sweep(dry_run=args.dry_run, generate_only=args.generate_only, train_only=args.train_only)


if __name__ == "__main__":
    main()
