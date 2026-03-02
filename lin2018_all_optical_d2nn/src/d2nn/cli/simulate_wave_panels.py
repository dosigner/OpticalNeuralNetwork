"""Simulate free-space and D2NN propagation, then generate panel-style figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from d2nn.physics.asm import asm_propagate, asm_transfer_function
from d2nn.viz.propagation import (
    generate_phase_masks,
    plot_stacked_xy_comparison,
    plot_wave_propagation_figure_s6,
    plot_xz_cross_section_comparison,
    plot_xz_cross_section_volume,
    simulate_d2nn_volume,
    simulate_free_space_volume,
)


def make_three_spot_input_field(
    *,
    N: int,
    spacing_px: int | None = None,
    sigma_px: float = 1.2,
) -> torch.Tensor:
    """Create a 3-spot amplitude object (zero phase), shape (N, N)."""

    if spacing_px is None:
        spacing_px = max(2, N // 4)
    y, x = torch.meshgrid(
        torch.arange(N, dtype=torch.float32),
        torch.arange(N, dtype=torch.float32),
        indexing="ij",
    )
    center_y = float((N - 1) / 2.0)
    center_x = float((N - 1) / 2.0)
    centers = [center_x - spacing_px, center_x, center_x + spacing_px]

    amp = torch.zeros((N, N), dtype=torch.float32)
    if sigma_px <= 0.0:
        cy = int(round(center_y))
        for cx in centers:
            cxi = int(round(cx))
            if 0 <= cy < N and 0 <= cxi < N:
                amp[cy, cxi] = 1.0
    else:
        for cx in centers:
            amp = amp + torch.exp(-((x - cx) ** 2 + (y - center_y) ** 2) / (2.0 * sigma_px**2))
    amp = amp / amp.max().clamp_min(1e-8)
    return torch.complex(amp, torch.zeros_like(amp))


def optimize_phase_masks_for_object(
    input_field: torch.Tensor,
    *,
    num_layers: int,
    dx: float,
    wavelength: float,
    layer_spacing: float,
    steps: int = 300,
    lr: float = 0.03,
    seed: int = 1234,
) -> list[np.ndarray]:
    """Optimize phase-only masks to reconstruct input amplitude at output plane."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(int(seed))
    device = input_field.device
    N = int(input_field.shape[-1])
    phases = torch.nn.Parameter(torch.zeros((num_layers, N, N), device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([phases], lr=float(lr))

    H = asm_transfer_function(
        N=N,
        dx=float(dx),
        wavelength=float(wavelength),
        z=float(layer_spacing),
        device=device,
    )
    target_amp = torch.abs(input_field).detach()

    for _ in range(int(steps)):
        out = input_field
        for l in range(num_layers):
            out = asm_propagate(out, H)
            phi = torch.remainder(phases[l], 2.0 * torch.pi)
            out = out * torch.exp(1j * phi)

        pred_amp = torch.abs(out)
        recon = torch.mean((pred_amp - target_amp) ** 2)
        smooth_x = torch.mean((phases[:, :, 1:] - phases[:, :, :-1]) ** 2)
        smooth_y = torch.mean((phases[:, 1:, :] - phases[:, :-1, :]) ** 2)
        loss = recon + 1e-4 * (smooth_x + smooth_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        phase_masks: list[np.ndarray] = []
        for l in range(num_layers):
            phi = torch.remainder(phases[l], 2.0 * torch.pi)
            phase_masks.append(phi.detach().cpu().numpy().astype(np.float32))
    return phase_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate D2NN vs free-space propagation panels")
    parser.add_argument("--output-dir", default="runs/wave_panels", help="Output directory for images and npz")
    parser.add_argument("--N", type=int, default=200, help="Grid size [pixels]")
    parser.add_argument("--input-field-npy", default=None, help="Optional path to input complex field (.npy, shape N x N)")
    parser.add_argument("--dx", type=float, default=0.4e-3, help="Pixel pitch [m]")
    parser.add_argument("--wavelength", type=float, default=0.75e-3, help="Wavelength [m]")
    parser.add_argument("--total-z", type=float, default=0.12, help="Total propagation distance [m]")
    parser.add_argument("--num-segments", type=int, default=10, help="Free-space segmentation count")
    parser.add_argument("--num-layers", type=int, default=10, help="Number of diffractive layers")
    parser.add_argument(
        "--mask-mode",
        choices=["fresnel", "random", "adaptive", "optimized"],
        default="optimized",
        help="Phase mask mode",
    )
    parser.add_argument("--focal-length", type=float, default=None, help="Fresnel focal length [m] (default: total-z/2)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for random masks")
    parser.add_argument("--sigma-px", type=float, default=1.2, help="Input spot sigma in pixels (0: Dirac-like)")
    parser.add_argument("--spot-spacing-px", type=int, default=None, help="Input spot spacing in pixels")
    parser.add_argument("--y-index", type=int, default=None, help="Fixed y index for x-z cross-section")
    parser.add_argument("--num-depths", type=int, default=10, help="Depth count for stacked x-y maps")
    parser.add_argument("--amp-cmap", default="viridis", help="Amplitude colormap")
    parser.add_argument("--phase-cmap", default="hsv", help="Phase colormap")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Simulation device")
    parser.add_argument("--opt-steps", type=int, default=300, help="Optimization steps for --mask-mode optimized")
    parser.add_argument("--opt-lr", type=float, default=0.03, help="Optimization learning rate for --mask-mode optimized")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.input_field_npy:
        input_field_np = np.load(args.input_field_npy)
        if input_field_np.ndim == 3:
            if input_field_np.shape[0] != 1:
                raise ValueError(f"input field with ndim=3 must have leading size 1; got {input_field_np.shape}")
            input_field_np = input_field_np[0]
        if input_field_np.ndim != 2 or input_field_np.shape[0] != input_field_np.shape[1]:
            raise ValueError(f"input field must have square shape (N, N); got {input_field_np.shape}")
        if not np.iscomplexobj(input_field_np):
            input_field_np = input_field_np.astype(np.float32)
            input_field_np = input_field_np + 0j
        N = int(input_field_np.shape[0])
        input_field = torch.from_numpy(input_field_np.astype(np.complex64)).to(device)
    else:
        N = int(args.N)
        input_field = make_three_spot_input_field(N=N, spacing_px=args.spot_spacing_px, sigma_px=args.sigma_px).to(device)

    free_volume_xyz, free_z = simulate_free_space_volume(
        input_field,
        dx=args.dx,
        wavelength=args.wavelength,
        total_distance=args.total_z,
        num_segments=args.num_segments,
        device=device,
    )

    layer_spacing = args.total_z / float(args.num_layers)
    focal_length = args.focal_length if args.focal_length is not None else args.total_z / 2.0
    phase_masks = None
    if args.mask_mode == "optimized":
        phase_masks = optimize_phase_masks_for_object(
            input_field,
            num_layers=args.num_layers,
            dx=args.dx,
            wavelength=args.wavelength,
            layer_spacing=layer_spacing,
            steps=args.opt_steps,
            lr=args.opt_lr,
            seed=args.seed,
        )
    elif args.mask_mode != "adaptive":
        phase_masks = generate_phase_masks(
            num_layers=args.num_layers,
            N=N,
            mode=args.mask_mode,
            dx=args.dx,
            wavelength=args.wavelength,
            focal_length=focal_length,
            seed=args.seed,
        )
    d2nn_volume_xyz, d2nn_z = simulate_d2nn_volume(
        input_field,
        dx=args.dx,
        wavelength=args.wavelength,
        num_layers=args.num_layers,
        layer_spacing=layer_spacing,
        phase_masks=phase_masks,
        mask_mode=args.mask_mode,
        device=device,
    )

    np.savez_compressed(
        output_dir / "propagation_volumes_xyz.npz",
        free_space_volume_xyz=free_volume_xyz,
        free_space_z_m=free_z,
        d2nn_volume_xyz=d2nn_volume_xyz,
        d2nn_z_m=d2nn_z,
        input_field_yx=input_field.detach().cpu().numpy(),
    )

    plot_xz_cross_section_volume(
        d2nn_volume_xyz,
        z_positions=d2nn_z,
        dx=args.dx,
        y_index=args.y_index,
        quantity="amplitude",
        cmap=args.amp_cmap,
        title="D$^2$NN x-z Amplitude",
        save_path=output_dir / "d2nn_xz_amplitude.png",
    )
    plot_xz_cross_section_volume(
        free_volume_xyz,
        z_positions=free_z,
        dx=args.dx,
        y_index=args.y_index,
        quantity="amplitude",
        cmap=args.amp_cmap,
        title="Free-space x-z Amplitude",
        save_path=output_dir / "free_space_xz_amplitude.png",
    )
    plot_xz_cross_section_comparison(
        d2nn_volume_xyz,
        free_volume_xyz,
        d2nn_z_positions=d2nn_z,
        free_space_z_positions=free_z,
        dx=args.dx,
        y_index=args.y_index,
        cmap=args.amp_cmap,
        save_path=output_dir / "xz_amplitude_comparison.png",
    )
    plot_stacked_xy_comparison(
        d2nn_volume_xyz,
        free_volume_xyz,
        d2nn_z_positions=d2nn_z,
        free_space_z_positions=free_z,
        num_depths=args.num_depths,
        amp_cmap=args.amp_cmap,
        phase_cmap=args.phase_cmap,
        save_path=output_dir / "stacked_xy_amp_phase_comparison.png",
    )
    plot_wave_propagation_figure_s6(
        d2nn_volume_xyz,
        free_volume_xyz,
        d2nn_z_positions=d2nn_z,
        free_space_z_positions=free_z,
        dx=args.dx,
        y_index=args.y_index,
        amp_cmap=args.amp_cmap,
        phase_cmap=args.phase_cmap,
        save_path=output_dir / "figure_s6_style.png",
    )

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "free_space_volume_shape_xyz": tuple(free_volume_xyz.shape),
        "d2nn_volume_shape_xyz": tuple(d2nn_volume_xyz.shape),
        "mask_mode": args.mask_mode,
        "num_layers": args.num_layers,
        "num_segments": args.num_segments,
    }
    print(summary)


if __name__ == "__main__":
    main()
