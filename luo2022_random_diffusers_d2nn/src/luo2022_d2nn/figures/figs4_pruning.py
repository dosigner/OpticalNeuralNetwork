"""Supplementary Figure S4: pruning-condition comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from scipy.spatial import ConvexHull, QhullError
import torch
import torch.nn.functional as F

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.data.mnist import MNISTAmplitude
from luo2022_d2nn.eval.pcc import compute_pcc
from luo2022_d2nn.figures.fig2_known_new import (
    _forward_single,
    _generate_known_diffusers,
    _generate_new_diffusers,
    _load_model,
)
from luo2022_d2nn.figures.figs3_overlap_map import (
    _compute_island_mask,
    _load_wrapped_phases,
    _make_circular_roi,
)
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.utils.viz import contrast_enhance, save_figure

CONDITION_ORDER = (
    "full_layers",
    "no_layers",
    "islands_only",
    "dilated_islands",
    "inside_contour",
    "aperture_80lambda",
)

CONDITION_LABELS = {
    "full_layers": "Full layers",
    "no_layers": "No layers",
    "islands_only": "Islands only",
    "dilated_islands": "Dilated islands",
    "inside_contour": "Inside contour",
    "aperture_80lambda": "80lambda aperture",
}

OUTPUT_COLUMN_LABELS = [
    "Digit 2\nKnown diffuser",
    "Digit 2\nNew diffuser",
    "OOD object\nKnown diffuser",
    "OOD object\nNew diffuser",
]


def _mask_from_convex_hull(mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    points_rc = np.argwhere(mask)
    if len(points_rc) < 3:
        return mask & roi_mask

    try:
        hull = ConvexHull(points_rc)
    except QhullError:
        return mask & roi_mask

    polygon_xy = points_rc[hull.vertices][:, ::-1]
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    coords_xy = np.column_stack([xx.ravel(), yy.ravel()])
    hull_mask = matplotlib.path.Path(polygon_xy).contains_points(coords_xy).reshape(mask.shape)
    return (hull_mask | mask) & roi_mask


def build_condition_masks(
    base_masks: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Derive pruning-condition masks from S3 phase-island masks."""
    if base_masks.ndim != 3:
        raise ValueError("base_masks must have shape (layers, height, width)")

    roi_stack = np.broadcast_to(roi_mask[None, :, :], base_masks.shape)
    full_stack = np.ones_like(base_masks, dtype=bool)
    no_stack = np.zeros_like(base_masks, dtype=bool)

    dilated_masks = np.zeros_like(base_masks, dtype=bool)
    contour_masks = np.zeros_like(base_masks, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)

    for layer_idx in range(base_masks.shape[0]):
        island_mask = base_masks[layer_idx] & roi_mask
        dilated = binary_dilation(island_mask, structure=structure)
        contour = (_mask_from_convex_hull(island_mask, roi_mask) | dilated) & roi_mask
        dilated_masks[layer_idx] = dilated & roi_mask
        contour_masks[layer_idx] = contour & roi_mask

    condition_masks = {
        "full_layers": full_stack,
        "no_layers": no_stack,
        "islands_only": base_masks & roi_stack,
        "dilated_islands": dilated_masks,
        "inside_contour": contour_masks,
        "aperture_80lambda": roi_stack.copy(),
    }
    kept_ratios = {
        name: float(mask.mean())
        for name, mask in condition_masks.items()
    }
    return condition_masks, kept_ratios


def materialize_condition_phases(
    wrapped_phases: np.ndarray,
    condition_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return one wrapped-phase stack per pruning condition."""
    variants: dict[str, np.ndarray] = {}
    for name, mask in condition_masks.items():
        if name == "full_layers":
            variants[name] = wrapped_phases.astype(np.float32, copy=True)
        else:
            variants[name] = np.where(mask, wrapped_phases, 0.0).astype(np.float32, copy=False)
    return variants


def load_ood_amplitude_from_image(
    image_path: str,
    resize_to: int,
    final_size: int,
) -> torch.Tensor:
    """Load grayscale OOD object and zero out pure-white background."""
    image = Image.open(image_path).convert("L")
    foreground = image.point(lambda px: 0 if px >= 250 else 255)

    resized = image.resize((resize_to, resize_to), Image.Resampling.BILINEAR)
    resized_mask = foreground.resize((resize_to, resize_to), Image.Resampling.NEAREST)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mask = np.asarray(resized_mask, dtype=np.float32) / 255.0
    arr[mask < 0.5] = 0.0

    amplitude = torch.from_numpy(arr).unsqueeze(0)
    pad_total = final_size - resize_to
    if pad_total < 0:
        raise ValueError("final_size must be >= resize_to")
    pad_each = pad_total // 2
    remainder = pad_total - 2 * pad_each
    amplitude = F.pad(
        amplitude,
        (pad_each, pad_each + remainder, pad_each, pad_each + remainder),
        mode="constant",
        value=0.0,
    )
    return amplitude.to(torch.float32)


def _default_ood_asset_path() -> str:
    return str(Path(__file__).resolve().parents[3] / "reference" / "ood_s4_object.png")


def _load_digit_two(cfg: dict[str, Any]) -> torch.Tensor:
    dataset = MNISTAmplitude(
        root="data",
        split="test",
        resize_to=int(cfg["dataset"].get("resize_to_px", 160)),
        final_size=int(cfg["dataset"].get("final_resolution_px", 240)),
    )
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if int(sample["label"]) == 2:
            return sample["amplitude"].to(torch.float32)
    raise RuntimeError("Failed to find digit 2 in MNIST test split")


def _compute_island_masks(wrapped_phases: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    return np.stack(
        [_compute_island_mask(phase, roi_mask) for phase in wrapped_phases],
        axis=0,
    )


def _assign_phase_stack(model, phase_stack: np.ndarray, device: torch.device) -> None:
    with torch.no_grad():
        for layer_idx, phase_map in enumerate(phase_stack):
            target = torch.from_numpy(phase_map).to(device=device, dtype=model.layers[layer_idx].phase.dtype)
            model.layers[layer_idx].phase.copy_(target)


def _forward_without_diffractive_layers(
    amplitude: torch.Tensor,
    diffuser_t: torch.Tensor,
    H_obj_to_diff: torch.Tensor,
    H_diff_to_output: torch.Tensor,
    pad_factor: int,
) -> torch.Tensor:
    """Forward pass with the diffractive layers physically removed."""
    field = amplitude.unsqueeze(0).to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    field_at_diff = field_at_diff * diffuser_t.unsqueeze(0)
    field_out = bl_asm_propagate(field_at_diff, H_diff_to_output, pad_factor=pad_factor)
    return field_out.abs() ** 2


def _phase_for_display(phase_stack: np.ndarray) -> np.ndarray:
    display = np.mod(phase_stack, 2.0 * np.pi)
    zero_mask = np.isclose(phase_stack, 0.0)
    display[zero_mask] = 0.0
    return display


def _build_objects(
    cfg: dict[str, Any],
    digit_amplitude: torch.Tensor | None,
    ood_amplitude: torch.Tensor | None,
    ood_image_path: str | None,
) -> list[tuple[str, torch.Tensor]]:
    final_size = int(cfg["dataset"].get("final_resolution_px", 240))
    resize_to = int(cfg["dataset"].get("resize_to_px", 160))

    digit = digit_amplitude if digit_amplitude is not None else _load_digit_two(cfg)
    if ood_amplitude is not None:
        ood = ood_amplitude
    else:
        asset_path = ood_image_path or _default_ood_asset_path()
        ood = load_ood_amplitude_from_image(asset_path, resize_to=resize_to, final_size=final_size)

    return [
        ("Digit 2", digit.squeeze(0).to(torch.float32)),
        ("OOD object", ood.squeeze(0).to(torch.float32)),
    ]


def _build_diffusers(
    cfg: dict[str, Any],
    device: torch.device,
    known_diffuser: dict[str, Any] | None,
    new_diffuser: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    known = known_diffuser if known_diffuser is not None else _generate_known_diffusers(cfg, 1, device)[0]
    new = new_diffuser if new_diffuser is not None else _generate_new_diffusers(cfg, 1, device)[0]
    return [("Known diffuser", known), ("New diffuser", new)]


def make_figs4(
    checkpoint_path: str,
    config_path: str = "configs/baseline.yaml",
    save_path: str | None = None,
    *,
    digit_amplitude: torch.Tensor | None = None,
    ood_amplitude: torch.Tensor | None = None,
    known_diffuser: dict[str, Any] | None = None,
    new_diffuser: dict[str, Any] | None = None,
    ood_image_path: str | None = None,
    figure_title: str = "Supp. Fig. S4: Comparison Under Different Levels of Pruning",
) -> dict[str, Any]:
    """Generate Supplementary Figure S4 using pruning-condition comparisons."""
    device = torch.device("cpu")
    cfg = load_and_validate_config(config_path)

    wrapped_phases = _load_wrapped_phases(checkpoint_path)
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    dx_mm = float(cfg["grid"]["pitch_mm"])
    roi_radius_px = 0.5 * (80.0 * wavelength_mm / dx_mm)
    roi_mask = _make_circular_roi(wrapped_phases.shape[1:], radius_px=roi_radius_px)

    island_masks = _compute_island_masks(wrapped_phases, roi_mask)
    condition_masks, kept_ratios = build_condition_masks(island_masks, roi_mask)
    phase_variants = materialize_condition_phases(wrapped_phases, condition_masks)
    row_labels = [CONDITION_LABELS[name] for name in CONDITION_ORDER]

    model = _load_model(checkpoint_path, cfg, device)
    obj_to_diff_mm = float(cfg["geometry"]["object_to_diffuser_mm"])
    pad_factor = int(cfg["grid"].get("pad_factor", 2))
    H_obj_to_diff = bl_asm_transfer_function(
        int(cfg["grid"]["nx"]),
        dx_mm,
        wavelength_mm,
        obj_to_diff_mm,
        pad_factor=pad_factor,
    )
    total_free_space_mm = (
        float(cfg["geometry"]["diffuser_to_layer1_mm"])
        + (int(cfg["geometry"]["num_layers"]) - 1) * float(cfg["geometry"]["layer_to_layer_mm"])
        + float(cfg["geometry"]["last_layer_to_output_mm"])
    )
    H_diff_to_output = bl_asm_transfer_function(
        int(cfg["grid"]["nx"]),
        dx_mm,
        wavelength_mm,
        total_free_space_mm,
        pad_factor=pad_factor,
    )

    objects = _build_objects(cfg, digit_amplitude, ood_amplitude, ood_image_path)
    diffusers = _build_diffusers(cfg, device, known_diffuser, new_diffuser)

    viz_cfg = cfg["visualization"]["contrast_enhancement"]
    lo_pct = float(viz_cfg.get("lower_percentile", 1.0))
    hi_pct = float(viz_cfg.get("upper_percentile", 99.0))

    num_rows = len(CONDITION_ORDER)
    layer_display_phases = np.zeros((num_rows, wrapped_phases.shape[0], *wrapped_phases.shape[1:]), dtype=np.float32)
    raw_outputs = np.zeros((num_rows, 4, *wrapped_phases.shape[1:]), dtype=np.float32)
    display_outputs = np.zeros_like(raw_outputs)
    pccs = np.zeros((num_rows, 4), dtype=np.float32)

    for row_idx, condition in enumerate(CONDITION_ORDER):
        phase_stack = phase_variants[condition]
        _assign_phase_stack(model, phase_stack, device)
        layer_display_phases[row_idx] = _phase_for_display(phase_stack)

        output_idx = 0
        for _, obj_amp in objects:
            target = obj_amp.unsqueeze(0)
            for _, diffuser in diffusers:
                if condition == "no_layers":
                    intensity = _forward_without_diffractive_layers(
                        amplitude=obj_amp,
                        diffuser_t=diffuser["transmittance"],
                        H_obj_to_diff=H_obj_to_diff,
                        H_diff_to_output=H_diff_to_output,
                        pad_factor=pad_factor,
                    )
                else:
                    intensity = _forward_single(
                        obj_amp,
                        diffuser["transmittance"],
                        model,
                        H_obj_to_diff,
                        pad_factor,
                    )
                raw = intensity.squeeze(0).detach().cpu().numpy().astype(np.float32)
                raw_outputs[row_idx, output_idx] = raw
                display_outputs[row_idx, output_idx] = contrast_enhance(raw, lo_pct, hi_pct).astype(np.float32)
                pccs[row_idx, output_idx] = float(compute_pcc(intensity, target).item())
                output_idx += 1

    fig, axes = plt.subplots(num_rows, 8, figsize=(16.5, 12.5), facecolor="white")
    for row_idx, condition in enumerate(CONDITION_ORDER):
        kept_pct = 100.0 * kept_ratios[condition]
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(layer_display_phases[row_idx, col_idx], cmap="viridis", vmin=0.0, vmax=2.0 * np.pi)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Layer {col_idx + 1}", fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.text(
                    -0.20,
                    0.50,
                    f"{CONDITION_LABELS[condition]}\nkeep {kept_pct:.1f}%",
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    ha="right",
                    va="center",
                )

        for out_idx in range(4):
            ax = axes[row_idx, 4 + out_idx]
            ax.imshow(display_outputs[row_idx, out_idx], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(OUTPUT_COLUMN_LABELS[out_idx], fontsize=9, fontweight="bold")
            ax.text(
                0.96,
                0.05,
                f"PCC={pccs[row_idx, out_idx]:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                ha="right",
                va="bottom",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

    fig.suptitle(figure_title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(
        0.73,
        0.015,
        "PCC uses raw images; display panels use contrast enhancement only.",
        fontsize=8,
        ha="center",
    )
    fig.tight_layout(rect=[0.07, 0.03, 1.0, 0.97])

    result = {
        "row_labels": row_labels,
        "output_column_labels": OUTPUT_COLUMN_LABELS,
        "kept_ratios": np.array([kept_ratios[name] for name in CONDITION_ORDER], dtype=np.float32),
        "pccs": pccs,
        "layer_display_phases": layer_display_phases,
        "raw_outputs": raw_outputs,
        "display_outputs": display_outputs,
        "condition_masks": {name: mask.copy() for name, mask in condition_masks.items()},
    }

    if save_path is not None:
        save_figure(fig, save_path)
        np.save(
            str(Path(save_path).with_suffix(".npy")),
            {
                "row_labels": row_labels,
                "output_column_labels": OUTPUT_COLUMN_LABELS,
                "kept_ratios": result["kept_ratios"],
                "pccs": pccs,
            },
        )

    plt.close(fig)
    return result
