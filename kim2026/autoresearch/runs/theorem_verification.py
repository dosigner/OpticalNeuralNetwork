#!/usr/bin/env python3
"""Theorem 1 verification: CO(HU_t, HU_v) vs CO(U_t, U_v).

GPT54 review's most critical point: the paper may have verified CO(HU_t, U_v)
instead of CO(HU_t, HU_v). This script computes ALL three quantities:

  1. CO(U_t, U_v)         -- baseline (input plane, no D2NN)
  2. CO(HU_t, HU_v)       -- theorem 1 prediction (both through D2NN)
  3. CO(HU_t, U_v_det)    -- what the paper actually measured (mixed)

Plus L2 distance verification for Theorem 2:
  4. ||U_t - U_v||_2      -- baseline
  5. ||HU_t - HU_v||_2    -- theorem 2 prediction
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.data.dataset import CachedFieldDataset
from kim2026.training.metrics import complex_overlap
from kim2026.training.targets import apply_receiver_aperture, make_detector_plane_target

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75"
STRATEGIES = {
    "pib_only": Path(__file__).resolve().parent / "d2nn_loss_strategy" / "pib_only" / "checkpoint.pt",
    "co_pib_hybrid": Path(__file__).resolve().parent / "d2nn_loss_strategy" / "co_pib_hybrid" / "checkpoint.pt",
    "strehl_only": Path(__file__).resolve().parent / "d2nn_loss_strategy" / "strehl_only" / "checkpoint.pt",
    "intensity_overlap": Path(__file__).resolve().parent / "d2nn_loss_strategy" / "intensity_overlap" / "checkpoint.pt",
}

WAVELENGTH_M = 1.55e-6
WINDOW_M = 2.048e-3
APERTURE_M = 2.0e-3
NUM_LAYERS = 5
LAYER_SPACING_M = 10e-3
DETECTOR_DISTANCE_M = 10e-3
TOTAL_DISTANCE_M = (NUM_LAYERS - 1) * LAYER_SPACING_M + DETECTOR_DISTANCE_M


def load_model(ckpt_path, device):
    model = BeamCleanupD2NN(
        n=1024,
        wavelength_m=WAVELENGTH_M,
        window_m=WINDOW_M,
        num_layers=NUM_LAYERS,
        layer_spacing_m=LAYER_SPACING_M,
        detector_distance_m=DETECTOR_DISTANCE_M,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_co(a, b):
    """CO for single pair (add batch dim)."""
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    return complex_overlap(a, b).item()


def compute_l2(a, b):
    """L2 distance for single pair."""
    return torch.linalg.vector_norm((a - b).flatten()).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test dataset
    dataset = CachedFieldDataset(
        cache_dir=str(DATA_DIR / "cache"),
        manifest_path=str(DATA_DIR / "split_manifest.json"),
        split="test",
    )
    print(f"Test samples: {len(dataset)}")

    all_results = {}

    for strategy_name, ckpt_path in STRATEGIES.items():
        if not ckpt_path.exists():
            print(f"SKIP {strategy_name}: checkpoint not found")
            continue

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*60}")

        model = load_model(ckpt_path, device)

        records = []

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                u_turb = sample["u_turb"].unsqueeze(0).to(device)
                u_vac = sample["u_vacuum"].unsqueeze(0).to(device)

                # Apply aperture to both fields (input plane)
                u_turb_ap = apply_receiver_aperture(
                    u_turb, receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M
                )
                u_vac_ap = apply_receiver_aperture(
                    u_vac, receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M
                )

                # ── Input-plane metrics (before D2NN) ──
                co_input = compute_co(u_turb_ap, u_vac_ap)
                l2_input = compute_l2(u_turb_ap, u_vac_ap)

                # ── Pass BOTH through D2NN ──
                hu_turb = model(u_turb_ap)   # H(U_turb)
                hu_vac = model(u_vac_ap)     # H(U_vac)

                # ── Detector-plane vacuum target (single propagation, no D2NN) ──
                u_vac_det = make_detector_plane_target(
                    u_vac, wavelength_m=WAVELENGTH_M,
                    receiver_window_m=WINDOW_M, aperture_diameter_m=APERTURE_M,
                    total_distance_m=TOTAL_DISTANCE_M, complex_mode=True,
                )

                # ── Theorem 1: CO(HU_t, HU_v) should equal CO(U_t, U_v) ──
                co_both_d2nn = compute_co(hu_turb, hu_vac)

                # ── Mixed: CO(HU_t, U_v_det) — what paper measured ──
                co_mixed = compute_co(hu_turb, u_vac_det)

                # ── Theorem 2: L2 distance ──
                l2_both_d2nn = compute_l2(hu_turb, hu_vac)

                # ── Throughput check ──
                throughput_turb = (hu_turb.abs().square().sum() / u_turb_ap.abs().square().sum()).item()
                throughput_vac = (hu_vac.abs().square().sum() / u_vac_ap.abs().square().sum()).item()

                records.append({
                    "idx": i,
                    "co_input": co_input,
                    "co_both_d2nn": co_both_d2nn,
                    "co_mixed": co_mixed,
                    "l2_input": l2_input,
                    "l2_both_d2nn": l2_both_d2nn,
                    "throughput_turb": throughput_turb,
                    "throughput_vac": throughput_vac,
                })

        # ── Summary ──
        n = len(records)
        co_inputs = [r["co_input"] for r in records]
        co_boths = [r["co_both_d2nn"] for r in records]
        co_mixeds = [r["co_mixed"] for r in records]
        l2_inputs = [r["l2_input"] for r in records]
        l2_boths = [r["l2_both_d2nn"] for r in records]

        co_diffs = [abs(a - b) for a, b in zip(co_inputs, co_boths)]
        l2_diffs = [abs(a - b) for a, b in zip(l2_inputs, l2_boths)]
        co_rel = [abs(a - b) / max(a, 1e-12) for a, b in zip(co_inputs, co_boths)]

        summary = {
            "strategy": strategy_name,
            "n_samples": n,
            "theorem1_CO": {
                "CO(U_t,U_v)_mean": float(np.mean(co_inputs)),
                "CO(U_t,U_v)_std": float(np.std(co_inputs)),
                "CO(HU_t,HU_v)_mean": float(np.mean(co_boths)),
                "CO(HU_t,HU_v)_std": float(np.std(co_boths)),
                "abs_diff_mean": float(np.mean(co_diffs)),
                "abs_diff_max": float(np.max(co_diffs)),
                "rel_diff_mean": float(np.mean(co_rel)),
                "VERDICT": "PRESERVED" if np.mean(co_diffs) < 1e-4 else "VIOLATED" if np.mean(co_diffs) > 0.01 else "APPROX",
            },
            "mixed_CO": {
                "CO(HU_t,U_v_det)_mean": float(np.mean(co_mixeds)),
                "CO(HU_t,U_v_det)_std": float(np.std(co_mixeds)),
                "delta_from_baseline": float(np.mean(co_mixeds) - np.mean(co_inputs)),
            },
            "theorem2_L2": {
                "L2(U_t-U_v)_mean": float(np.mean(l2_inputs)),
                "L2(HU_t-HU_v)_mean": float(np.mean(l2_boths)),
                "abs_diff_mean": float(np.mean(l2_diffs)),
                "abs_diff_max": float(np.max(l2_diffs)),
                "VERDICT": "PRESERVED" if np.mean(l2_diffs) / np.mean(l2_inputs) < 1e-4 else "APPROX",
            },
        }

        all_results[strategy_name] = {"summary": summary, "per_sample": records}

        # Print
        t1 = summary["theorem1_CO"]
        t2 = summary["theorem2_L2"]
        mx = summary["mixed_CO"]

        print(f"\n--- Theorem 1 (Inner Product / CO Preservation) ---")
        print(f"  CO(U_t, U_v)      = {t1['CO(U_t,U_v)_mean']:.6f} +/- {t1['CO(U_t,U_v)_std']:.6f}")
        print(f"  CO(HU_t, HU_v)    = {t1['CO(HU_t,HU_v)_mean']:.6f} +/- {t1['CO(HU_t,HU_v)_std']:.6f}")
        print(f"  |diff| mean/max   = {t1['abs_diff_mean']:.2e} / {t1['abs_diff_max']:.2e}")
        print(f"  Verdict: {t1['VERDICT']}")

        print(f"\n--- Mixed CO (what paper measured) ---")
        print(f"  CO(HU_t, U_v_det) = {mx['CO(HU_t,U_v_det)_mean']:.6f} +/- {mx['CO(HU_t,U_v_det)_std']:.6f}")
        print(f"  Delta from input   = {mx['delta_from_baseline']:+.6f}")

        print(f"\n--- Theorem 2 (L2 Distance Preservation) ---")
        print(f"  ||U_t - U_v||     = {t2['L2(U_t-U_v)_mean']:.6f}")
        print(f"  ||HU_t - HU_v||   = {t2['L2(HU_t-HU_v)_mean']:.6f}")
        print(f"  |diff| mean/max   = {t2['abs_diff_mean']:.2e} / {t2['abs_diff_max']:.2e}")
        print(f"  Verdict: {t2['VERDICT']}")

    # Save
    out_path = Path(__file__).resolve().parent / "theorem_verification_results.json"
    # Convert to serializable
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = v["summary"]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
