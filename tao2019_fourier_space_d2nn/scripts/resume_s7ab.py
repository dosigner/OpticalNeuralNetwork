"""Resume training for missing S7(a)(b) models, then generate plots."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from _common import run_with_overrides


def _deep_update(dst: dict, src: dict) -> dict:
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _is_completed(runs_root: Path, exp_name: str) -> bool:
    exp_dir = runs_root / exp_name
    if not exp_dir.exists():
        return False
    for rd in sorted(exp_dir.iterdir()):
        if rd.is_dir() and (rd / "metrics.json").exists():
            return True
    return False


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "src" / "tao2019_fd2nn" / "config"
    runs_root = project_root / "runs"
    bs = 10
    epochs = 30

    jobs: list[tuple[str, Path, dict[str, Any]]] = []

    # ── Missing S7(a): multi-sbn 3l, 4l, 5l ──
    for n_layers in [3, 4, 5]:
        exp_name = f"s7a_nonlinear_fourier_muti-sbn_{n_layers}l"
        jobs.append((
            exp_name,
            cfg_dir / "cls_mnist_nonlinear_fourier_multi_sbn_1l.yaml",
            {
                "experiment": {"name": exp_name},
                "model": {"num_layers": n_layers, "nonlinearity": {"intensity_norm": "per_sample_minmax"}},
                "training": {"batch_size": bs, "epochs": epochs},
            },
        ))

    # ── Missing S7(b): 4 configs ──
    s7b_jobs = [
        ("s7b_nonlinear_fourier_sbn_front", "cls_mnist_nonlinear_fourier_1l_f1mm.yaml",
         {"model": {"num_layers": 5, "nonlinearity": {"position": "front", "intensity_norm": "per_sample_minmax"}}}),
        ("s7b_nonlinear_fourier_sbn_rear", "cls_mnist_nonlinear_fourier_1l_f1mm.yaml",
         {"model": {"num_layers": 5, "nonlinearity": {"position": "rear", "intensity_norm": "per_sample_minmax"}}}),
        ("s7b_nonlinear_fourier_and_real_sbn_front", "cls_mnist_hybrid_5l.yaml",
         {"model": {"nonlinearity": {"position": "front", "intensity_norm": "per_sample_minmax"}}}),
        ("s7b_nonlinear_fourier_and_real_sbn_rear", "cls_mnist_hybrid_5l.yaml",
         {"model": {"nonlinearity": {"position": "rear", "intensity_norm": "per_sample_minmax"}}}),
    ]
    for exp_name, cfg_name, model_ov in s7b_jobs:
        base_ov: dict[str, Any] = {
            "experiment": {"name": exp_name},
            "training": {"batch_size": bs, "epochs": epochs},
        }
        jobs.append((exp_name, cfg_dir / cfg_name, _deep_update(base_ov, model_ov)))

    # ── Run missing jobs ──
    for exp_name, base_cfg, overrides in jobs:
        if _is_completed(runs_root, exp_name):
            print(f"SKIP {exp_name} (already completed)")
            continue
        print(f"\n{'='*60}")
        print(f"Training {exp_name}...")
        print(f"{'='*60}")
        run_with_overrides(base_cfg, overrides=overrides, task="classification")

    print("\n\nAll remaining models trained!")
    print("Now run: python scripts/reproduce_supp_s7ab.py --skip-train-s7a --skip-train-s7b --batch-size 10 --epochs 30")


if __name__ == "__main__":
    main()
