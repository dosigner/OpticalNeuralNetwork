"""Entry point that orchestrates the full FSO beam propagation simulation pipeline.

Runs vacuum propagation, turbulence Monte-Carlo realizations, and automated
verification checks, saving all outputs to a structured directory.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from kim2026.fso.config import SimulationConfig
from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0
from kim2026.fso.sampling import analyze_sampling
from kim2026.fso.phase_screen import generate_phase_screens
from kim2026.fso.propagation import (
    make_gaussian_source,
    ang_spec_multi_prop,
    compute_irradiance,
)
from kim2026.fso.verification import verify_phase_screens, verify_coherence_factor


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    return obj


def _save_json(data: dict, path: str) -> None:
    """Save a dict as pretty-printed JSON, handling numpy/torch types."""
    with open(path, "w") as f:
        json.dump(_to_serializable(data), f, indent=2)


def run_simulation(config: SimulationConfig, output_dir: str = "output") -> dict:
    """Full pipeline for FSO beam propagation through atmospheric turbulence.

    Steps:
        1. Compute atmospheric params (r0_pw, r0_sw, sigma2_chi)
        2. Analyze sampling (delta1, N, n_scr)
        3. Optimize screen r0 distribution
        4. Vacuum propagation (phase_screens=None)
        5. Turbulence Monte-Carlo (n_reals iterations)
        6. Automated verification (structure function + coherence factor)
        7. Save outputs

    Parameters
    ----------
    config : SimulationConfig
        Fully-specified simulation configuration.
    output_dir : str
        Root directory for outputs (default ``'output'``).

    Returns
    -------
    results : dict
        Summary of simulation results including atmospheric params,
        sampling analysis, verification outcomes, and file paths.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Create output directory structure ----
    out = Path(output_dir)
    vacuum_dir = out / "vacuum"
    turb_dir = out / "turbulence"
    verif_dir = out / "verification"
    for d in [out, vacuum_dir, turb_dir, verif_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Atmospheric parameters ----
    atm_params = compute_atmospheric_params(config.k, config.Cn2, config.Dz)
    r0_pw = atm_params["r0_pw"]
    r0_sw = atm_params["r0_sw"]
    sigma2_chi = atm_params["sigma2_chi_sw"]

    # ---- Step 2: Sampling analysis ----
    sampling = analyze_sampling(config)
    delta1 = sampling.delta1
    delta_n = sampling.delta_n
    N = sampling.N
    n_scr = sampling.n_scr
    z_planes = sampling.z_planes
    delta_values = sampling.delta_values

    _save_json(
        {
            "delta1": delta1,
            "delta_n": delta_n,
            "N": N,
            "n_scr": n_scr,
            "z_planes": z_planes,
            "delta_values": delta_values,
            "D1_prime": sampling.D1_prime,
            "D2_prime": sampling.D2_prime,
            "dz_max": sampling.dz_max,
        },
        str(out / "sampling_analysis.json"),
    )

    # ---- Step 3: Optimize screen r0 distribution ----
    r0_vals = optimize_screen_r0(r0_sw, sigma2_chi, config.k, config.Dz, n_scr)

    _save_json(
        {"r0_values": r0_vals.tolist(), "n_scr": n_scr},
        str(out / "screen_r0.json"),
    )

    # ---- Step 4: Vacuum propagation ----
    U_in = make_gaussian_source(N, delta1, config.w0, device=device)
    xn, yn, U_vac = ang_spec_multi_prop(
        U_in, config.wvl, delta1, delta_n, z_planes,
        phase_screens=None, device=device,
    )
    I_vac = compute_irradiance(U_vac)

    # Save vacuum results (move to CPU to free GPU memory)
    torch.save(U_vac.detach().cpu(), str(vacuum_dir / "field.pt"))
    torch.save(I_vac.detach().cpu(), str(vacuum_dir / "irradiance.pt"))

    # Save coordinate grids
    torch.save(
        {"xn": xn.detach().cpu(), "yn": yn.detach().cpu()},
        str(out / "coordinates.pt"),
    )

    # ---- Step 5: Turbulence Monte-Carlo ----
    turb_fields: List[torch.Tensor] = []
    # Collect phase screens per plane for verification
    phase_screens_by_plane: List[List[torch.Tensor]] = [[] for _ in range(n_scr)]

    for real_idx in range(config.n_reals):
        # Generate phase screens for this realization
        screens = generate_phase_screens(
            r0_vals.tolist(), N, delta_values, device=device,
        )

        # Store for verification
        for plane_idx, scr in enumerate(screens):
            phase_screens_by_plane[plane_idx].append(scr.detach().clone())

        # Propagate through turbulence
        _, _, U_turb = ang_spec_multi_prop(
            U_in, config.wvl, delta1, delta_n, z_planes,
            phase_screens=screens, device=device,
        )
        I_turb = compute_irradiance(U_turb)

        # Save each realization (move to CPU)
        torch.save(
            U_turb.detach().cpu(),
            str(turb_dir / f"field_{real_idx:04d}.pt"),
        )
        torch.save(
            I_turb.detach().cpu(),
            str(turb_dir / f"irradiance_{real_idx:04d}.pt"),
        )

        turb_fields.append(U_turb.detach())

    # ---- Step 6: Automated verification ----
    # 6a: Phase-screen structure function verification
    sf_report = verify_phase_screens(
        phase_screens_by_plane,
        r0_vals.tolist(),
        delta_values,
    )
    _save_json(
        {
            "all_pass": sf_report["all_pass"],
            "planes": [
                {
                    "rel_error": p["rel_error"],
                    "pass": p["pass"],
                }
                for p in sf_report["planes"]
            ],
        },
        str(verif_dir / "structure_function_report.json"),
    )

    # 6b: Coherence factor verification
    cf_report = verify_coherence_factor(
        turb_fields,
        delta_n,
        r0_sw,
        D_aperture=config.D_aperture,
        D_roi=config.D_roi,
        field_vacuum=U_vac,
    )
    _save_json(
        {
            "e_inv_width": cf_report["e_inv_width"],
            "rho_0_theory": cf_report["rho_0_theory"],
            "agreement_level": cf_report["agreement_level"],
        },
        str(verif_dir / "coherence_factor_report.json"),
    )

    # ---- Step 7: Save config and summary ----
    config_dict = {
        "Dz": config.Dz,
        "Cn2": config.Cn2,
        "theta_div": config.theta_div,
        "D_roi": config.D_roi,
        "delta_n": config.delta_n,
        "D_aperture": config.D_aperture,
        "N": N,
        "n_reals": config.n_reals,
        "wvl": config.wvl,
        "w0": config.w0,
        "k": config.k,
        "D1": config.D1,
    }
    _save_json(config_dict, str(out / "config.json"))

    # Free GPU memory by moving stored fields to CPU
    del turb_fields
    del phase_screens_by_plane
    if device == "cuda":
        torch.cuda.empty_cache()

    results = {
        "atmospheric_params": atm_params,
        "sampling": {
            "delta1": delta1,
            "delta_n": delta_n,
            "N": N,
            "n_scr": n_scr,
        },
        "r0_values": r0_vals.tolist(),
        "verification": {
            "structure_function": {
                "all_pass": sf_report["all_pass"],
            },
            "coherence_factor": {
                "agreement_level": cf_report["agreement_level"],
                "e_inv_width": cf_report["e_inv_width"],
                "rho_0_theory": cf_report["rho_0_theory"],
            },
        },
        "output_dir": str(out.resolve()),
    }

    return results
