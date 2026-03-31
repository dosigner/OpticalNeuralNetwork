"""Shared utilities for focal-plane PIB evaluation and visualization.

Eliminates duplication across eval_focal_pib_only.py, eval_bucket_radius_sweep.py,
visualize_focal_pib_report.py, and generate_focal_paper_figures.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.optics.lens_2f import lens_2f_forward
from kim2026.training.targets import apply_receiver_aperture, center_crop_field
from kim2026.training.metrics import complex_overlap

# ── Optics Constants ─────────────────────────────────────────────────────────
WAVELENGTH_M = 1.55e-6
GRID_SIZE = 1024
WINDOW_M = 0.002048
APERTURE_DIAMETER_M = 0.002
DX_M = WINDOW_M / GRID_SIZE
FOCAL_LENGTH_M = 4.5e-3
DX_FOCAL_M = WAVELENGTH_M * FOCAL_LENGTH_M / (GRID_SIZE * DX_M)

# ── D2NN Architecture ────────────────────────────────────────────────────────
D2NN_ARCH = dict(
    num_layers=5,
    layer_spacing_m=10e-3,
    detector_distance_m=10e-3,
)

# ── Strategy Metadata ────────────────────────────────────────────────────────
BUCKET_RADII_UM = [5.0, 10.0, 25.0, 50.0]
EE_RADII_UM = np.linspace(1, 100, 200)

FOCAL_STRATEGIES = [
    "focal_pib_only",
    "focal_strehl_only",
    "focal_intensity_overlap",
    "focal_co_pib_hybrid",
]

STRATEGY_LABELS = {
    "focal_pib_only": "Focal PIB",
    "focal_strehl_only": "Focal Strehl",
    "focal_intensity_overlap": "Focal IO",
    "focal_co_pib_hybrid": "CO+fPIB",
}

STRATEGY_COLORS = {
    "focal_pib_only": "#e74c3c",
    "focal_strehl_only": "#3498db",
    "focal_intensity_overlap": "#2ecc71",
    "focal_co_pib_hybrid": "#9b59b6",
}


# ── Path Helpers ─────────────────────────────────────────────────────────────

def get_kim2026_root() -> Path:
    """Return kim2026/ project root."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_focal_dirs() -> dict[str, Path]:
    """Return standard directories for focal PIB experiments."""
    root = get_kim2026_root()
    return {
        "data": root / "data" / "kim2026" / "1km_cn2_5e-14_tel15cm_n1024_br75",
        "focal_sweep": root / "autoresearch" / "runs" / "d2nn_focal_pib_sweep",
        "old_sweep": root / "autoresearch" / "runs" / "d2nn_loss_strategy",
    }


# ── Field Preparation ────────────────────────────────────────────────────────

def prepare_field(
    field: torch.Tensor,
    window_m: float = WINDOW_M,
    aperture_m: float = APERTURE_DIAMETER_M,
    crop_n: int = GRID_SIZE,
) -> torch.Tensor:
    """Apply receiver aperture and center crop."""
    return center_crop_field(
        apply_receiver_aperture(field, receiver_window_m=window_m, aperture_diameter_m=aperture_m),
        crop_n=crop_n,
    )


# ── Focal Optics ─────────────────────────────────────────────────────────────

def apply_focal_lens(
    field: torch.Tensor,
    dx_in_m: float = DX_M,
    wavelength_m: float = WAVELENGTH_M,
    f_m: float = FOCAL_LENGTH_M,
    na: float | None = None,
    apply_scaling: bool = False,
) -> tuple[torch.Tensor, float]:
    """Apply focusing lens and propagate to focal plane."""
    with torch.no_grad():
        f, dx = lens_2f_forward(
            field.to(torch.complex64),
            dx_in_m=dx_in_m,
            wavelength_m=wavelength_m,
            f_m=f_m,
            na=na,
            apply_scaling=apply_scaling,
        )
    return f, dx


# ── Model Loading ────────────────────────────────────────────────────────────

def load_checkpoint(
    base_path: Path,
    strategy_name: str,
    arch: dict | None = None,
    n: int = GRID_SIZE,
    wavelength_m: float = WAVELENGTH_M,
    window_m: float = WINDOW_M,
    map_location: str = "cpu",
) -> BeamCleanupD2NN | None:
    """Load D2NN checkpoint for a strategy. Returns None if checkpoint missing."""
    if arch is None:
        arch = D2NN_ARCH

    ckpt_path = base_path / strategy_name / "checkpoint.pt"
    if not ckpt_path.exists():
        return None

    model = BeamCleanupD2NN(n=n, wavelength_m=wavelength_m, window_m=window_m, **arch)
    state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def load_all_strategies(
    base_path: Path,
    strategies: list[str] | None = None,
    device: str = "cpu",
    **kwargs,
) -> dict[str, BeamCleanupD2NN]:
    """Load all available strategy checkpoints. Skips missing ones."""
    if strategies is None:
        strategies = FOCAL_STRATEGIES
    models = {}
    for name in strategies:
        m = load_checkpoint(base_path, name, map_location=device, **kwargs)
        if m is not None:
            models[name] = m
    return models


# ── PIB Computation (Torch) ──────────────────────────────────────────────────

def compute_pib_torch(
    focal_field: torch.Tensor,
    dx_focal: float,
    radii_um: list[float],
) -> dict[float, torch.Tensor]:
    """Compute Power-in-Bucket at multiple radii.

    Returns dict mapping radius_um -> tensor of PIB values (one per batch element).
    """
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(
        torch.arange(n, device=focal_field.device) - c,
        torch.arange(n, device=focal_field.device) - c,
        indexing="ij",
    )
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-12)

    results = {}
    for rad in radii_um:
        mask = (r <= rad * 1e-6).float()
        results[rad] = (intensity * mask).sum(dim=(-2, -1)) / total
    return results


def compute_pib_torch_mean(
    focal_field: torch.Tensor,
    dx_focal: float,
    radii_um: list[float],
) -> dict[float, float]:
    """Compute PIB and return batch-averaged floats."""
    pibs = compute_pib_torch(focal_field, dx_focal, radii_um)
    return {rad: float(pib.mean()) for rad, pib in pibs.items()}


# ── PIB Computation (NumPy) ──────────────────────────────────────────────────

def compute_pib_numpy(field_np: np.ndarray, dx: float, radius_um: float) -> float:
    """Compute PIB from a numpy complex field at a single radius."""
    irr = np.abs(field_np) ** 2
    n = field_np.shape[-1]
    c = n // 2
    yy, xx = np.mgrid[-c : n - c, -c : n - c]
    r = np.sqrt((xx * dx) ** 2 + (yy * dx) ** 2)
    mask = r <= (radius_um * 1e-6)
    return float(irr[mask].sum() / max(irr.sum(), 1e-30))


# ── Encircled Energy ─────────────────────────────────────────────────────────

def compute_ee_curve(
    focal_field: torch.Tensor,
    dx_focal: float,
    radii_um: np.ndarray | list[float] | None = None,
) -> list[float]:
    """Compute encircled energy curve at fine radii."""
    if radii_um is None:
        radii_um = EE_RADII_UM
    intensity = focal_field.abs().square()
    n = focal_field.shape[-1]
    c = n // 2
    yy, xx = torch.meshgrid(
        torch.arange(n, device=focal_field.device) - c,
        torch.arange(n, device=focal_field.device) - c,
        indexing="ij",
    )
    r = torch.sqrt((xx * dx_focal) ** 2 + (yy * dx_focal) ** 2)
    total = intensity.sum(dim=(-2, -1)).clamp(min=1e-12)
    ee = []
    for rad in radii_um:
        mask = (r <= rad * 1e-6).float()
        ee.append(float(((intensity * mask).sum(dim=(-2, -1)) / total).mean()))
    return ee


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_test_dataset(
    data_dir: Path | None = None,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[CachedFieldDataset, DataLoader]:
    """Load dataset and return (dataset, dataloader)."""
    if data_dir is None:
        data_dir = get_focal_dirs()["data"]
    ds = CachedFieldDataset(
        cache_dir=str(data_dir / "cache"),
        manifest_path=str(data_dir / "split_manifest.json"),
        split=split,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return ds, loader


# ── Geometry Helpers ─────────────────────────────────────────────────────────

def get_extent_box(n: int, dx: float, unit: float = 1.0) -> list[float]:
    """Generate [left, right, bottom, top] extent for matplotlib imshow."""
    h = n * dx / 2 / unit
    return [-h, h, -h, h]
