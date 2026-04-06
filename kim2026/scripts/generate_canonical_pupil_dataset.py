#!/usr/bin/env python
"""Generate the canonical telescope-pupil dataset for kim2026."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kim2026.data.canonical_pupil import (
    CANONICAL_DATASET_NAME,
    build_canonical_split_manifest,
    write_canonical_pupil_npz,
)
from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0
from kim2026.fso.config import SimulationConfig
from kim2026.fso.phase_screen import generate_phase_screens
from kim2026.fso.propagation import ang_spec_multi_prop, make_gaussian_source
from kim2026.fso.sampling import analyze_sampling
from kim2026.optics.aperture import circular_aperture

WAVELENGTH_M = 1.55e-6
THETA_DIV = 3.0e-4
PATH_LENGTH_M = 1000.0
CN2 = 5.0e-14
TELESCOPE_DIAMETER_M = 0.15
RECEIVER_WINDOW_M = 0.1536
BEAM_REDUCER_RATIO = 75
OUTPUT_WINDOW_M = 0.002048
CROP_N = 1024
SEED = 20260401
N_REALIZATIONS = 5000


def _canonical_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "kim2026" / CANONICAL_DATASET_NAME


def _estimate_storage_bytes(*, count: int, n: int) -> int:
    per_field_bytes = n * n * 2 * 4
    per_sample = 2 * per_field_bytes + 2 * n * 4 + 4096
    return count * per_sample


def _apply_telescope_pupil(field: torch.Tensor, *, window_m: float, diameter_m: float) -> torch.Tensor:
    mask = circular_aperture(n=field.shape[-1], window_m=window_m, diameter_m=diameter_m, device=field.device)
    return field * mask.to(field.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_canonical_root())
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--crop-n", type=int, default=CROP_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    cache_dir = out_dir / "cache"
    storage_gb = _estimate_storage_bytes(count=args.n_realizations, n=args.crop_n) / (1024**3)

    print(f"Output root: {out_dir}")
    print(f"Cache dir:    {cache_dir}")
    print(f"Realizations: {args.n_realizations}")
    print(f"Split counts: train=4000 val=500 test=500")
    print(f"Crop N:       {args.crop_n}")
    print(f"Storage est.: {storage_gb:.1f} GiB (raw upper bound)")
    if args.dry_run:
        return

    if args.n_realizations != 5000:
        split_manifest = build_canonical_split_manifest(
            total_realizations=args.n_realizations,
            split_counts={
                "train": int(round(args.n_realizations * 0.8)),
                "val": int(round(args.n_realizations * 0.1)),
                "test": args.n_realizations - int(round(args.n_realizations * 0.8)) - int(round(args.n_realizations * 0.1)),
            },
        )
    else:
        split_manifest = build_canonical_split_manifest(total_realizations=args.n_realizations)

    cache_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        Dz=PATH_LENGTH_M,
        Cn2=CN2,
        theta_div=THETA_DIV,
        D_roi=RECEIVER_WINDOW_M,
        delta_n=RECEIVER_WINDOW_M / args.crop_n,
        D_aperture=TELESCOPE_DIAMETER_M,
        n_reals=args.n_realizations,
        wvl=WAVELENGTH_M,
    )
    sampling = analyze_sampling(config)
    atm = compute_atmospheric_params(config.k, CN2, PATH_LENGTH_M)
    r0_vals = optimize_screen_r0(atm["r0_sw"], atm["sigma2_chi_sw"], config.k, PATH_LENGTH_M, sampling.n_scr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    u_in = make_gaussian_source(sampling.N, sampling.delta1, config.w0, device=device)
    _, _, u_vac_full = ang_spec_multi_prop(
        u_in,
        WAVELENGTH_M,
        sampling.delta1,
        sampling.delta_n,
        sampling.z_planes,
        phase_screens=None,
        device=device,
    )

    center = sampling.N // 2
    half = args.crop_n // 2
    u_vac_crop = u_vac_full[center - half : center + half, center - half : center + half]
    u_vacuum_pupil = _apply_telescope_pupil(u_vac_crop, window_m=RECEIVER_WINDOW_M, diameter_m=TELESCOPE_DIAMETER_M).detach().cpu()
    coords = (np.arange(args.crop_n, dtype=np.float32) - args.crop_n / 2 + 0.5) * np.float32(RECEIVER_WINDOW_M / args.crop_n)

    t0 = time.time()
    for split_name, filenames in split_manifest.items():
        print(f"{split_name}: {len(filenames)}")
        for filename in filenames:
            realization = int(filename.split("_")[1].split(".")[0])
            out_path = cache_dir / filename
            if out_path.exists():
                continue
            torch.manual_seed(args.seed + realization * 1009)
            screens = generate_phase_screens(r0_vals.tolist(), sampling.N, sampling.delta_values, device=device)
            _, _, u_turb_full = ang_spec_multi_prop(
                u_in,
                WAVELENGTH_M,
                sampling.delta1,
                sampling.delta_n,
                sampling.z_planes,
                phase_screens=screens,
                device=device,
            )
            u_turb_crop = u_turb_full[center - half : center + half, center - half : center + half]
            u_turb_pupil = _apply_telescope_pupil(u_turb_crop, window_m=RECEIVER_WINDOW_M, diameter_m=TELESCOPE_DIAMETER_M).detach().cpu()
            write_canonical_pupil_npz(
                out_path,
                u_vacuum_pupil=u_vacuum_pupil,
                u_turb_pupil=u_turb_pupil,
                x_pupil_m=coords,
                y_pupil_m=coords.copy(),
                metadata={
                    "plane": "telescope_pupil",
                    "generator_version": "pupil1024_v1",
                    "realization": realization,
                    "seed": args.seed + realization * 1009,
                    "Dz": PATH_LENGTH_M,
                    "Cn2": CN2,
                    "wvl": WAVELENGTH_M,
                    "theta_div": THETA_DIV,
                    "receiver_window_m": RECEIVER_WINDOW_M,
                    "telescope_diameter_m": TELESCOPE_DIAMETER_M,
                    "crop_n": args.crop_n,
                    "delta_n_pupil_m": RECEIVER_WINDOW_M / args.crop_n,
                    "beam_reducer_ratio": BEAM_REDUCER_RATIO,
                    "reducer_output_window_m": OUTPUT_WINDOW_M,
                    "vacuum_shared_across_realizations": True,
                },
            )

    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
