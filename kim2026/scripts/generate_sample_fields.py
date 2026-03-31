#!/usr/bin/env python
"""Generate sample_fields.npz for runs that only have checkpoint.pt.

Loads the trained model from checkpoint, runs inference on test_ds[0],
and saves input/pred/target fields as split real/imag arrays.

Usage:
    python scripts/generate_sample_fields.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.training.targets import apply_receiver_aperture

PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ / "data" / "kim2026" / "1km_cn2e-14_w2m_n1024_dx2mm" / "cache"
MANIFEST = DATA_DIR.parent / "split_manifest.json"

# Common optical parameters (shared across all runs)
COMMON = dict(
    wavelength_m=1.55e-6,
    receiver_window_m=0.002048,
    aperture_diameter_m=0.002,
    n=1024,
    num_layers=5,
    layer_spacing_m=1e-3,
    dual_2f_f1_m=1e-3,
    dual_2f_f2_m=1e-3,
    dual_2f_na1=0.16,
    dual_2f_na2=0.16,
    dual_2f_apply_scaling=False,
)

# Runs that need sample_fields generated
TARGETS = [
    {
        "run_dir": PROJ / "runs" / "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude" / "tanh_2pi",
        "constraint": "symmetric_tanh",
        "phase_max": 2 * math.pi,
    },
    {
        "run_dir": PROJ / "runs" / "03_fd2nn_phasorloss_roi1024_phase_range_sweep_claude" / "sig_2pi",
        "constraint": "sigmoid",
        "phase_max": 2 * math.pi,
    },
    {
        "run_dir": PROJ / "runs" / "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude" / "tanh_2pi",
        "constraint": "symmetric_tanh",
        "phase_max": 2 * math.pi,
    },
]


def generate_one(run_dir: Path, constraint: str, phase_max: float,
                 test_ds, device: torch.device) -> None:
    out_path = run_dir / "sample_fields.npz"
    if out_path.exists():
        print(f"  SKIP (already exists): {run_dir.parent.name}/{run_dir.name}")
        return

    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        print(f"  SKIP (no checkpoint): {run_dir.parent.name}/{run_dir.name}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = BeamCleanupFD2NN(
        n=COMMON["n"],
        wavelength_m=COMMON["wavelength_m"],
        window_m=COMMON["receiver_window_m"],
        num_layers=COMMON["num_layers"],
        layer_spacing_m=COMMON["layer_spacing_m"],
        phase_max=phase_max,
        phase_constraint=constraint,
        phase_init="uniform",
        phase_init_scale=0.1,
        dual_2f_f1_m=COMMON["dual_2f_f1_m"],
        dual_2f_f2_m=COMMON["dual_2f_f2_m"],
        dual_2f_na1=COMMON["dual_2f_na1"],
        dual_2f_na2=COMMON["dual_2f_na2"],
        dual_2f_apply_scaling=COMMON["dual_2f_apply_scaling"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    w = COMMON["receiver_window_m"]
    ap = COMMON["aperture_diameter_m"]

    with torch.no_grad():
        sample = test_ds[0]
        u_t = sample["u_turb"].unsqueeze(0).to(device)
        u_v = sample["u_vacuum"].unsqueeze(0).to(device)
        tgt = apply_receiver_aperture(u_v, receiver_window_m=w, aperture_diameter_m=ap)
        inp = apply_receiver_aperture(u_t, receiver_window_m=w, aperture_diameter_m=ap)
        pred = model(inp)

        np.savez(
            out_path,
            input_real=inp[0].real.cpu().numpy(),
            input_imag=inp[0].imag.cpu().numpy(),
            pred_real=pred[0].real.cpu().numpy(),
            pred_imag=pred[0].imag.cpu().numpy(),
            target_real=tgt[0].real.cpu().numpy(),
            target_imag=tgt[0].imag.cpu().numpy(),
        )

    print(f"  SAVED: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = CachedFieldDataset(
        cache_dir=str(DATA_DIR), manifest_path=str(MANIFEST), split="test"
    )
    print(f"Test dataset: {len(test_ds)} samples\n")

    for cfg in TARGETS:
        print(f"Processing: {cfg['run_dir'].parent.name}/{cfg['run_dir'].name}")
        generate_one(cfg["run_dir"], cfg["constraint"], cfg["phase_max"],
                     test_ds, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
