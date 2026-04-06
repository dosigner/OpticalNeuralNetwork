#!/usr/bin/env python
"""Validate the ideal reducer against the full thin-lens reference relay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kim2026.data.canonical_pupil import (
    CANONICAL_DATASET_NAME,
    REDUCER_RECOMMENDATION_IDEAL_OK,
    REDUCER_RECOMMENDATION_PROMOTE_FULL,
    read_canonical_pupil_npz,
)
from kim2026.optics.beam_reducer import (
    BeamReducerPlane,
    apply_beam_reducer,
    apply_physical_beam_reducer_reference,
)
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.metrics import complex_overlap, strehl_ratio_correct

RADIUS_UM = [10.0, 25.0, 50.0]


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "kim2026" / CANONICAL_DATASET_NAME


def _select_subset(manifest: dict[str, list[str]], *, val_count: int, test_count: int) -> list[tuple[str, str]]:
    return [("val", name) for name in manifest.get("val", [])[:val_count]] + [
        ("test", name) for name in manifest.get("test", [])[:test_count]
    ]


def _energy(field: torch.Tensor, *, window_m: float) -> float:
    dx = float(window_m) / field.shape[-1]
    return float((field.abs().square().sum() * (dx * dx)).item())


def _focus(field: torch.Tensor, *, window_m: float, wavelength_m: float) -> tuple[torch.Tensor, float]:
    return lens_2f_forward(
        field.unsqueeze(0).to(torch.complex64),
        dx_in_m=float(window_m) / field.shape[-1],
        wavelength_m=wavelength_m,
        f_m=4.5e-3,
        na=None,
        apply_scaling=False,
    )


def _compute_pibs(field: torch.Tensor, *, dx_focal: float) -> dict[str, float]:
    intensity = field.abs().square()
    n = field.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(torch.arange(n) - c, torch.arange(n) - c, indexing="ij")
    rr = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    total = intensity.sum().clamp_min(1.0e-12)
    return {
        f"pib_{int(radius)}um": float(((intensity * (rr <= radius * 1e-6)).sum() / total).item())
        for radius in RADIUS_UM
    }


def _relative_delta(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=_default_root())
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    cache_dir = dataset_root / "cache"
    cache_out = dataset_root / "reducer_val_cache"
    manifest_path = dataset_root / "split_manifest.json"

    print(f"Dataset root: {dataset_root}")
    print(f"Validation cache: {cache_out}")
    if args.dry_run and not manifest_path.exists():
        print(f"Manifest missing: {manifest_path}")
        print(f"Expected subset rule: first {args.val_count} val + first {args.test_count} test entries")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subset = _select_subset(manifest, val_count=args.val_count, test_count=args.test_count)
    print(f"Subset size: {len(subset)} ({args.val_count} val + {args.test_count} test)")
    if args.dry_run:
        for split_name, filename in subset[:5]:
            print(f"  {split_name}: {filename}")
        return

    cache_out.mkdir(parents=True, exist_ok=True)
    per_sample = []
    for split_name, filename in subset:
        record = read_canonical_pupil_npz(cache_dir / filename)
        metadata = record["metadata"]
        input_plane = BeamReducerPlane(
            window_m=float(metadata["receiver_window_m"]),
            n=int(metadata["crop_n"]),
            aperture_diameter_m=float(metadata["telescope_diameter_m"]),
        )
        output_plane = BeamReducerPlane(
            window_m=float(metadata["reducer_output_window_m"]),
            n=int(metadata["crop_n"]),
            aperture_diameter_m=float(metadata["reducer_output_window_m"]),
        )
        ideal_vac = apply_beam_reducer(record["u_vacuum_pupil"], input_plane=input_plane, output_plane=output_plane)
        ideal_turb = apply_beam_reducer(record["u_turb_pupil"], input_plane=input_plane, output_plane=output_plane)
        full_vac = apply_physical_beam_reducer_reference(record["u_vacuum_pupil"], input_plane=input_plane, output_plane=output_plane)
        full_turb = apply_physical_beam_reducer_reference(record["u_turb_pupil"], input_plane=input_plane, output_plane=output_plane)

        focal_ideal_turb, dx_focal = _focus(ideal_turb, window_m=output_plane.window_m, wavelength_m=float(metadata["wvl"]))
        focal_full_turb, _ = _focus(full_turb, window_m=output_plane.window_m, wavelength_m=float(metadata["wvl"]))
        pibs_ideal = _compute_pibs(focal_ideal_turb[0].cpu(), dx_focal=float(dx_focal))
        pibs_full = _compute_pibs(focal_full_turb[0].cpu(), dx_focal=float(dx_focal))

        metrics = {
            "split": split_name,
            "filename": filename,
            "energy_error": _relative_delta(
                _energy(ideal_turb.cpu(), window_m=output_plane.window_m),
                _energy(full_turb.cpu(), window_m=output_plane.window_m),
            ),
            "co_error": _relative_delta(
                float(complex_overlap(ideal_turb.unsqueeze(0), full_turb.unsqueeze(0)).item()),
                1.0,
            ),
            "strehl_error": _relative_delta(
                float(strehl_ratio_correct(ideal_turb.unsqueeze(0)).item()),
                float(strehl_ratio_correct(full_turb.unsqueeze(0)).item()),
            ),
        }
        for key, value in pibs_ideal.items():
            metrics[f"{key}_error"] = _relative_delta(value, pibs_full[key])
        per_sample.append(metrics)

        np.savez_compressed(
            cache_out / filename,
            u_vacuum_reduced_ideal=ideal_vac.cpu().numpy(),
            u_turb_reduced_ideal=ideal_turb.cpu().numpy(),
            u_vacuum_reduced_full=full_vac.cpu().numpy(),
            u_turb_reduced_full=full_turb.cpu().numpy(),
            metrics_json=np.array(json.dumps(metrics)),
        )

    summary = {
        "dataset_root": str(dataset_root),
        "sample_count": len(per_sample),
        "max_energy_error": max(item["energy_error"] for item in per_sample),
        "max_co_error": max(item["co_error"] for item in per_sample),
        "max_strehl_error": max(item["strehl_error"] for item in per_sample),
        "max_pib_10um_error": max(item["pib_10um_error"] for item in per_sample),
        "max_pib_25um_error": max(item["pib_25um_error"] for item in per_sample),
        "max_pib_50um_error": max(item["pib_50um_error"] for item in per_sample),
    }
    summary["passed"] = all(
        [
            summary["max_energy_error"] <= 0.01,
            summary["max_co_error"] <= 0.02,
            summary["max_strehl_error"] <= 0.02,
            summary["max_pib_10um_error"] <= 0.02,
            summary["max_pib_25um_error"] <= 0.02,
            summary["max_pib_50um_error"] <= 0.02,
        ]
    )
    summary["recommendation"] = (
        REDUCER_RECOMMENDATION_IDEAL_OK if summary["passed"] else REDUCER_RECOMMENDATION_PROMOTE_FULL
    )
    summary["per_sample_metrics_path"] = str(cache_out / "per_sample_metrics.json")
    (cache_out / "per_sample_metrics.json").write_text(json.dumps(per_sample, indent=2), encoding="utf-8")
    (cache_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
