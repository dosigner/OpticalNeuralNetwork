#!/usr/bin/env python
"""Generate beam-reduced dataset from existing large-window data.

Applies an ideal afocal telescope to compress the field from the receiver
aperture onto the metalens/D2NN input plane.

Example:
    python scripts/generate_beam_reduced_dataset.py \
        --input-dir data/kim2026/crop_w15cm_n1024_dx150um \
        --output-dir data/kim2026/br_15cm_to_5mm_n1024 \
        --aperture-diameter-m 0.15 \
        --metalens-window-m 0.00512
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from kim2026.optics.beam_reducer import apply_beam_reducer


def _load_complex_field(npz: dict, prefix: str) -> torch.Tensor:
    real = torch.from_numpy(npz[f"{prefix}_real"].astype(np.float32))
    imag = torch.from_numpy(npz[f"{prefix}_imag"].astype(np.float32))
    return torch.complex(real, imag)


def _split_complex(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    arr = field.numpy()
    return arr.real.astype(np.float32), arr.imag.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Source dataset directory")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--aperture-diameter-m", type=float, required=True,
                        help="Telescope entrance aperture diameter [m]")
    parser.add_argument("--metalens-window-m", type=float, required=True,
                        help="Metalens/D2NN input window size [m]")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_in = input_dir / "cache"
    cache_out = output_dir / "cache"
    cache_out.mkdir(parents=True, exist_ok=True)

    aperture_m = args.aperture_diameter_m
    metalens_m = args.metalens_window_m
    magnification = aperture_m / metalens_m

    # Copy split manifest
    for manifest in ["split_manifest.json", "episodes.json"]:
        src = input_dir / manifest
        if src.exists():
            shutil.copy2(src, output_dir / manifest)

    # Discover input window from first file
    files = sorted(cache_in.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {cache_in}")

    sample = np.load(files[0])
    if "metadata_json" in sample:
        meta = json.loads(str(sample["metadata_json"]))
        x = sample["x_m"]
        input_window_m = float(meta.get("window_m", x.max() - x.min() + (x[1] - x[0])))
    else:
        x = sample["x_m"]
        input_window_m = float(x.max() - x.min() + (x[1] - x[0]))

    n = int(sample["x_m"].shape[0])
    dx_out = metalens_m / n
    x_out = (np.arange(n) - n / 2 + 0.5) * dx_out
    y_out = x_out.copy()

    print(f"Input:  window={input_window_m*1e3:.1f}mm, N={n}, dx={input_window_m/n*1e6:.1f}µm")
    print(f"Aperture: {aperture_m*1e3:.1f}mm")
    print(f"Output: window={metalens_m*1e3:.2f}mm, N={n}, dx={dx_out*1e6:.2f}µm")
    print(f"Magnification: {magnification:.1f}x")
    print(f"Files: {len(files)}")
    print()

    for i, fpath in enumerate(files):
        data = np.load(fpath)
        u_vac = _load_complex_field(data, "u_vacuum")
        u_turb = _load_complex_field(data, "u_turb")

        u_vac_br = apply_beam_reducer(
            u_vac.unsqueeze(0),
            input_window_m=input_window_m,
            aperture_diameter_m=aperture_m,
            output_window_m=metalens_m,
        ).squeeze(0)

        u_turb_br = apply_beam_reducer(
            u_turb.unsqueeze(0),
            input_window_m=input_window_m,
            aperture_diameter_m=aperture_m,
            output_window_m=metalens_m,
        ).squeeze(0)

        vac_real, vac_imag = _split_complex(u_vac_br)
        turb_real, turb_imag = _split_complex(u_turb_br)

        # Build metadata
        if "metadata_json" in data:
            orig_meta = json.loads(str(data["metadata_json"]))
        else:
            orig_meta = {}
        new_meta = {
            **orig_meta,
            "beam_reducer": {
                "input_window_m": input_window_m,
                "aperture_diameter_m": aperture_m,
                "output_window_m": metalens_m,
                "magnification": magnification,
            },
            "window_m": metalens_m,
            "delta_n": dx_out,
            "N": n,
        }

        out_path = cache_out / fpath.name
        np.savez_compressed(
            out_path,
            u_vacuum_real=vac_real,
            u_vacuum_imag=vac_imag,
            u_turb_real=turb_real,
            u_turb_imag=turb_imag,
            x_m=x_out,
            y_m=y_out,
            metadata_json=json.dumps(new_meta),
        )

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(files)}] {fpath.name} -> {out_path.name}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
