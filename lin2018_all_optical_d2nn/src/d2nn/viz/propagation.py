"""Propagation simulation and visualization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from d2nn.physics.asm import asm_propagate, asm_transfer_function
from d2nn.viz.style import apply_style


def _as_complex_field_yx(field: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert input to a single complex field with shape (N, N) in (y, x) order."""

    if isinstance(field, np.ndarray):
        x = torch.from_numpy(field)
    else:
        x = field

    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"field with ndim=3 must have batch size 1; got shape={tuple(x.shape)}")
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"field must have shape (N, N) or (1, N, N); got shape={tuple(x.shape)}")
    if x.shape[0] != x.shape[1]:
        raise ValueError(f"field must be square; got shape={tuple(x.shape)}")

    if not torch.is_complex(x):
        xr = x.to(torch.float32)
        x = torch.complex(xr, torch.zeros_like(xr))
    return x


def _to_xyz_volume(fields_zyx: list[torch.Tensor]) -> np.ndarray:
    """Convert list of fields from (z, y, x) into numpy volume (x, y, z)."""

    stack_zyx = torch.stack(fields_zyx, dim=0)  # (z, y, x)
    stack_np = stack_zyx.detach().cpu().numpy()
    return np.transpose(stack_np, (2, 1, 0))  # (x, y, z)


def make_fresnel_lens_phase(
    *,
    N: int,
    dx: float,
    wavelength: float,
    focal_length: float,
) -> np.ndarray:
    """Build wrapped Fresnel-lens phase map in radians.

    Returns:
        ndarray, shape (N, N), range [0, 2*pi).
    """

    if focal_length <= 0.0:
        raise ValueError(f"focal_length must be positive; got {focal_length}")

    coords = (np.arange(N, dtype=np.float64) - (N - 1) / 2.0) * dx
    X, Y = np.meshgrid(coords, coords, indexing="xy")  # (y, x)
    phase = -(np.pi / (wavelength * focal_length)) * (X**2 + Y**2)
    return np.mod(phase, 2.0 * np.pi).astype(np.float32)


def generate_phase_masks(
    *,
    num_layers: int,
    N: int,
    mode: str = "fresnel",
    dx: float | None = None,
    wavelength: float | None = None,
    focal_length: float | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate per-layer phase masks in (y, x) order.

    Args:
        mode: "fresnel" or "random"

    Returns:
        List of arrays, each shape (N, N), radians.
    """

    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive; got {num_layers}")
    if mode not in {"fresnel", "random"}:
        raise ValueError(f"mode must be 'fresnel' or 'random'; got {mode}")

    if mode == "fresnel":
        if dx is None or wavelength is None or focal_length is None:
            raise ValueError("dx, wavelength, and focal_length are required for fresnel masks")
        mask = make_fresnel_lens_phase(N=N, dx=float(dx), wavelength=float(wavelength), focal_length=float(focal_length))
        return [mask.copy() for _ in range(num_layers)]

    rng = np.random.default_rng(seed)
    return [rng.uniform(0.0, 2.0 * np.pi, size=(N, N)).astype(np.float32) for _ in range(num_layers)]


def simulate_free_space_volume(
    input_field: torch.Tensor | np.ndarray,
    *,
    dx: float,
    wavelength: float,
    total_distance: float,
    num_segments: int = 10,
    bandlimit: bool = True,
    fftshifted: bool = False,
    dtype: str = "complex64",
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate free-space propagation and store complex volume.

    Args:
        input_field: complex field, shape (N, N) or (1, N, N), in (y, x) order.
        total_distance: total propagation distance [m]
        num_segments: number of equal-distance propagation segments

    Returns:
        volume_xyz: complex ndarray, shape (N, N, num_segments + 1), axis order (x, y, z)
        z_positions: float ndarray [m], shape (num_segments + 1,)
    """

    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive; got {num_segments}")
    if total_distance < 0.0:
        raise ValueError(f"total_distance must be non-negative; got {total_distance}")

    field_yx = _as_complex_field_yx(input_field)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    out = field_yx.to(dev)
    N = int(out.shape[-1])
    dz = float(total_distance) / float(num_segments)

    H = asm_transfer_function(
        N=N,
        dx=float(dx),
        wavelength=float(wavelength),
        z=dz,
        bandlimit=bool(bandlimit),
        fftshifted=bool(fftshifted),
        dtype=str(dtype),
        device=dev,
    )

    fields_zyx: list[torch.Tensor] = [out]
    for _ in range(num_segments):
        out = asm_propagate(out, H, fftshifted=bool(fftshifted))
        fields_zyx.append(out)

    volume_xyz = _to_xyz_volume(fields_zyx)
    z_positions = np.linspace(0.0, float(total_distance), num_segments + 1, dtype=np.float64)
    return volume_xyz, z_positions


def simulate_d2nn_volume(
    input_field: torch.Tensor | np.ndarray,
    *,
    dx: float,
    wavelength: float,
    num_layers: int = 10,
    layer_spacing: float,
    phase_masks: list[torch.Tensor | np.ndarray] | None = None,
    mask_mode: str = "fresnel",
    focal_length: float | None = None,
    seed: int = 0,
    bandlimit: bool = True,
    fftshifted: bool = False,
    dtype: str = "complex64",
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate D2NN propagation with phase-only layers.

    Sequence per layer:
        free-space propagation -> phase modulation

    Args:
        input_field: complex field, shape (N, N) or (1, N, N), in (y, x) order.
        phase_masks: optional list of radians maps (y, x), each shape (N, N).
            If omitted, masks are generated from `mask_mode`.
        mask_mode: "fresnel", "random", or "adaptive" (input-conditioned phase cancellation)

    Returns:
        volume_xyz: complex ndarray, shape (N, N, num_layers + 1), axis order (x, y, z)
        z_positions: float ndarray [m], shape (num_layers + 1,)
    """

    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive; got {num_layers}")
    if layer_spacing <= 0.0:
        raise ValueError(f"layer_spacing must be positive; got {layer_spacing}")

    field_yx = _as_complex_field_yx(input_field)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    out = field_yx.to(dev)
    N = int(out.shape[-1])

    adaptive_mode = phase_masks is None and mask_mode == "adaptive"
    if phase_masks is None and not adaptive_mode:
        phase_masks = generate_phase_masks(
            num_layers=num_layers,
            N=N,
            mode=mask_mode,
            dx=float(dx),
            wavelength=float(wavelength),
            focal_length=focal_length,
            seed=seed,
        )
    masks_t: list[torch.Tensor] = []
    if not adaptive_mode:
        if phase_masks is None:
            raise ValueError("phase_masks must be provided unless mask_mode='adaptive'")
        if len(phase_masks) != num_layers:
            raise ValueError(f"phase_masks length must match num_layers ({num_layers}); got {len(phase_masks)}")
        for idx, mask in enumerate(phase_masks):
            if isinstance(mask, np.ndarray):
                mt = torch.from_numpy(mask)
            else:
                mt = mask
            if mt.shape != (N, N):
                raise ValueError(f"phase mask #{idx} must have shape {(N, N)}; got {tuple(mt.shape)}")
            masks_t.append(mt.to(device=dev, dtype=out.real.dtype))

    H = asm_transfer_function(
        N=N,
        dx=float(dx),
        wavelength=float(wavelength),
        z=float(layer_spacing),
        bandlimit=bool(bandlimit),
        fftshifted=bool(fftshifted),
        dtype=str(dtype),
        device=dev,
    )

    fields_zyx: list[torch.Tensor] = [out]
    for layer_idx in range(num_layers):
        propagated = asm_propagate(out, H, fftshifted=bool(fftshifted))
        # Store the wavefront at each layer plane before modulation.
        fields_zyx.append(propagated)
        if adaptive_mode:
            # Input-conditioned "oracle" phase flattening for visualization:
            # each layer cancels current phase to emulate idealized focusing behavior.
            mask = -torch.angle(propagated)
        else:
            mask = masks_t[layer_idx]
        out = propagated * torch.exp(1j * mask)

    volume_xyz = _to_xyz_volume(fields_zyx)
    z_positions = np.linspace(0.0, float(layer_spacing) * num_layers, num_layers + 1, dtype=np.float64)
    return volume_xyz, z_positions


def extract_xz_cross_section(
    volume_xyz: np.ndarray,
    *,
    y_index: int | None = None,
    quantity: str = "amplitude",
) -> np.ndarray:
    """Extract x-z cross-section from (x, y, z) volume.

    Returns:
        ndarray, shape (z, x)
    """

    volume = np.asarray(volume_xyz)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D (x, y, z); got shape={volume.shape}")
    if y_index is None:
        y_index = volume.shape[1] // 2
    if not (0 <= y_index < volume.shape[1]):
        raise ValueError(f"y_index out of range: {y_index}")

    if quantity == "phase":
        arr = np.angle(volume)
    elif quantity == "amplitude":
        arr = np.abs(volume)
    else:
        raise ValueError("quantity must be 'amplitude' or 'phase'")

    xz = arr[:, y_index, :]  # (x, z)
    return xz.T  # (z, x)


def _extent_mm(N: int, dx: float) -> tuple[float, float]:
    half = 0.5 * N * dx * 1e3
    return -half, half


def plot_xz_cross_section_volume(
    volume_xyz: np.ndarray,
    *,
    z_positions: np.ndarray,
    dx: float,
    y_index: int | None = None,
    quantity: str = "amplitude",
    cmap: str | None = None,
    title: str = "x-z cross section",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot x-z cross section from (x, y, z) complex volume."""

    import matplotlib.pyplot as plt

    apply_style()
    xz = extract_xz_cross_section(volume_xyz, y_index=y_index, quantity=quantity)
    N = int(volume_xyz.shape[0])
    x0, x1 = _extent_mm(N, float(dx))
    z0 = float(np.min(z_positions)) * 1e3
    z1 = float(np.max(z_positions)) * 1e3

    if cmap is None:
        cmap = "twilight" if quantity == "phase" else "magma"
    vmin = -np.pi if quantity == "phase" else 0.0
    vmax = np.pi if quantity == "phase" else float(np.max(xz))

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(xz, cmap=cmap, origin="lower", aspect="auto", extent=[x0, x1, z0, z1], vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("phase [rad]" if quantity == "phase" else "amplitude [a.u.]")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_xz_cross_section_comparison(
    d2nn_volume_xyz: np.ndarray,
    free_space_volume_xyz: np.ndarray,
    *,
    d2nn_z_positions: np.ndarray,
    free_space_z_positions: np.ndarray,
    dx: float,
    y_index: int | None = None,
    cmap: str = "magma",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot side-by-side x-z amplitude cross-sections for D2NN and free-space."""

    import matplotlib.pyplot as plt

    apply_style()
    xz_d2nn = extract_xz_cross_section(d2nn_volume_xyz, y_index=y_index, quantity="amplitude")
    xz_free = extract_xz_cross_section(free_space_volume_xyz, y_index=y_index, quantity="amplitude")

    vmax = float(max(np.max(xz_d2nn), np.max(xz_free)))
    N = int(d2nn_volume_xyz.shape[0])
    x0, x1 = _extent_mm(N, float(dx))
    z_d0 = float(np.min(d2nn_z_positions)) * 1e3
    z_d1 = float(np.max(d2nn_z_positions)) * 1e3
    z_f0 = float(np.min(free_space_z_positions)) * 1e3
    z_f1 = float(np.max(free_space_z_positions)) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=False)
    im0 = axes[0].imshow(
        xz_d2nn,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=[x0, x1, z_d0, z_d1],
        vmin=0.0,
        vmax=vmax,
    )
    axes[0].set_title("D$^2$NN Propagation (x-z Amp.)")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("z [mm]")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("amplitude [a.u.]")

    im1 = axes[1].imshow(
        xz_free,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=[x0, x1, z_f0, z_f1],
        vmin=0.0,
        vmax=vmax,
    )
    axes[1].set_title("Free-space Propagation (x-z Amp.)")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("z [mm]")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("amplitude [a.u.]")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_stacked_xy_comparison(
    d2nn_volume_xyz: np.ndarray,
    free_space_volume_xyz: np.ndarray,
    *,
    d2nn_z_positions: np.ndarray | None = None,
    free_space_z_positions: np.ndarray | None = None,
    num_depths: int = 10,
    amp_cmap: str = "magma",
    phase_cmap: str = "twilight",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot stacked x-y amplitude/phase maps for D2NN and free-space volumes."""

    import matplotlib.pyplot as plt

    apply_style()
    nz_d2nn = int(d2nn_volume_xyz.shape[2])
    nz_free = int(free_space_volume_xyz.shape[2])
    nrows = max(1, min(int(num_depths), nz_d2nn, nz_free))
    idx_d2nn = np.linspace(0, nz_d2nn - 1, nrows, dtype=int)
    idx_free = np.linspace(0, nz_free - 1, nrows, dtype=int)

    d2nn_amp_max = float(np.max(np.abs(d2nn_volume_xyz[:, :, idx_d2nn])))
    free_amp_max = float(np.max(np.abs(free_space_volume_xyz[:, :, idx_free])))
    amp_vmax = max(d2nn_amp_max, free_amp_max, 1e-8)

    fig, axes = plt.subplots(nrows, 4, figsize=(11.2, 1.2 * nrows + 1.5), squeeze=False)
    titles = ["D$^2$NN Amp.", "D$^2$NN Phase", "Free-space Amp.", "Free-space Phase"]
    for col, text in enumerate(titles):
        axes[0, col].set_title(text)

    amp_mappable = None
    phase_mappable = None
    for row in range(nrows):
        iz_d = int(idx_d2nn[row])
        iz_f = int(idx_free[row])

        d_amp = np.abs(d2nn_volume_xyz[:, :, iz_d]).T
        d_phase = np.angle(d2nn_volume_xyz[:, :, iz_d]).T
        f_amp = np.abs(free_space_volume_xyz[:, :, iz_f]).T
        f_phase = np.angle(free_space_volume_xyz[:, :, iz_f]).T

        amp_mappable = axes[row, 0].imshow(d_amp, cmap=amp_cmap, origin="lower", vmin=0.0, vmax=amp_vmax)
        phase_mappable = axes[row, 1].imshow(d_phase, cmap=phase_cmap, origin="lower", vmin=-np.pi, vmax=np.pi)
        axes[row, 2].imshow(f_amp, cmap=amp_cmap, origin="lower", vmin=0.0, vmax=amp_vmax)
        axes[row, 3].imshow(f_phase, cmap=phase_cmap, origin="lower", vmin=-np.pi, vmax=np.pi)

        z_label = ""
        if d2nn_z_positions is not None:
            z_label = f"z={float(d2nn_z_positions[iz_d]) * 1e3:.1f} mm"
        elif free_space_z_positions is not None:
            z_label = f"z={float(free_space_z_positions[iz_f]) * 1e3:.1f} mm"
        if z_label:
            axes[row, 0].set_ylabel(z_label)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    if amp_mappable is not None and phase_mappable is not None:
        amp_axes = [axes[r, 0] for r in range(nrows)] + [axes[r, 2] for r in range(nrows)]
        phase_axes = [axes[r, 1] for r in range(nrows)] + [axes[r, 3] for r in range(nrows)]
        cbar_amp = fig.colorbar(amp_mappable, ax=amp_axes, fraction=0.022, pad=0.01)
        cbar_amp.set_label("amplitude [a.u.]")
        cbar_phase = fig.colorbar(phase_mappable, ax=phase_axes, fraction=0.022, pad=0.01)
        cbar_phase.set_label("phase [rad]")
        cbar_phase.set_ticks([-np.pi, 0.0, np.pi])
        cbar_phase.set_ticklabels(["-π", "0", "π"])

    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_propagation_stack(
    fields: list[np.ndarray],
    *,
    quantity: str = "amplitude",
    title: str = "Propagation stack",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot layer-wise field stack.

    Args:
        fields: list of complex arrays, each shape (N, N)
        quantity: "amplitude" or "phase"
    """

    import matplotlib.pyplot as plt

    apply_style()
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), squeeze=False)

    for i, field in enumerate(fields):
        ax = axes[0, i]
        if quantity == "phase":
            arr = np.angle(field)
            cmap = "twilight"
        else:
            arr = np.abs(field)
            cmap = "viridis"
        ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(f"L{i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_xz_cross_section(
    fields: list[np.ndarray],
    *,
    x_index: int,
    quantity: str = "amplitude",
    title: str = "x-z cross section",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Build x-z map from a fixed x-index across layer stack."""

    import matplotlib.pyplot as plt

    apply_style()
    lines = []
    for field in fields:
        if quantity == "phase":
            arr = np.angle(field)
        else:
            arr = np.abs(field)
        lines.append(arr[:, x_index])

    xz = np.stack(lines, axis=0)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.imshow(xz, cmap="magma", aspect="auto", origin="lower")
    ax.set_xlabel("y index")
    ax.set_ylabel("z-step")
    ax.set_title(title)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def _normalize_for_display(arr: np.ndarray, *, percentile: float = 99.5, gamma: float = 0.65) -> np.ndarray:
    """Normalize non-negative amplitude map for display."""

    x = np.asarray(arr, dtype=np.float64)
    vmax = float(np.percentile(x, percentile))
    if vmax <= 1e-12:
        vmax = 1.0
    y = np.clip(x / vmax, 0.0, 1.0)
    if gamma != 1.0:
        y = y ** float(gamma)
    return y


def _box_blur_1d(arr: np.ndarray, *, k: int, axis: int) -> np.ndarray:
    """Simple reflect-padded box blur along one axis."""

    if k <= 1:
        return arr
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width=pad_width, mode="reflect")
    kernel = np.ones(k, dtype=np.float64) / float(k)
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), axis=axis, arr=padded)
    return out


def _blur_complex2d(arr: np.ndarray, *, ky: int = 1, kx: int = 1) -> np.ndarray:
    """Light complex-domain box blur for visualization stability."""

    real = np.asarray(arr.real, dtype=np.float64)
    imag = np.asarray(arr.imag, dtype=np.float64)
    real = _box_blur_1d(real, k=ky, axis=0)
    real = _box_blur_1d(real, k=kx, axis=1)
    imag = _box_blur_1d(imag, k=ky, axis=0)
    imag = _box_blur_1d(imag, k=kx, axis=1)
    return real + 1j * imag


def _build_stacked_xy_image(
    volume_xyz: np.ndarray,
    *,
    quantity: str,
    depth_indices: np.ndarray,
    separator_px: int = 2,
    blur_ky: int = 1,
    blur_kx: int = 1,
) -> tuple[np.ndarray, list[int]]:
    """Build vertically stacked x-y slice image.

    Returns:
        stacked image and list of separator row indices.
    """

    slices: list[np.ndarray] = []
    seps: list[int] = []
    for i, iz in enumerate(depth_indices):
        plane = volume_xyz[:, :, int(iz)].T  # (y, x)
        plane = _blur_complex2d(plane, ky=blur_ky, kx=blur_kx)
        if quantity == "phase":
            img = np.mod(np.angle(plane), 2.0 * np.pi)
        elif quantity == "amplitude":
            img = _normalize_for_display(np.abs(plane))
        else:
            raise ValueError("quantity must be 'amplitude' or 'phase'")

        slices.append(img)
        if i < len(depth_indices) - 1 and separator_px > 0:
            fill = 0.0
            sep = np.full((separator_px, img.shape[1]), fill, dtype=np.float64)
            slices.append(sep)
            seps.append(sum(s.shape[0] for s in slices[:-1]))

    return np.vstack(slices), seps


def _plot_top_stack_panel(
    fig,
    subplotspec,
    *,
    volume_xyz: np.ndarray,
    z_positions: np.ndarray,
    panel_letter: str,
    panel_title: str,
    amp_cmap: str,
    phase_cmap: str,
):
    """Plot top panel (A or B): stacked x-y amplitude and phase."""

    import matplotlib.pyplot as plt

    gs = subplotspec.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    ax_amp = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])

    depth_idx = np.arange(volume_xyz.shape[2], dtype=int)
    amp_stack, amp_sep = _build_stacked_xy_image(
        volume_xyz,
        quantity="amplitude",
        depth_indices=depth_idx,
        separator_px=2,
        blur_ky=3,
        blur_kx=3,
    )
    phase_stack, phase_sep = _build_stacked_xy_image(
        volume_xyz,
        quantity="phase",
        depth_indices=depth_idx,
        separator_px=2,
        blur_ky=3,
        blur_kx=3,
    )

    im_amp = ax_amp.imshow(amp_stack, cmap=amp_cmap, origin="upper", aspect="auto", vmin=0.0, vmax=1.0)
    im_phase = ax_phase.imshow(phase_stack, cmap=phase_cmap, origin="upper", aspect="auto", vmin=0.0, vmax=2.0 * np.pi)

    for y in amp_sep:
        ax_amp.axhline(y=y - 0.5, color="white", alpha=0.30, lw=0.5)
    for y in phase_sep:
        ax_phase.axhline(y=y - 0.5, color="white", alpha=0.30, lw=0.5)

    for ax in (ax_amp, ax_phase):
        ax.set_xticks([])
        ax.set_yticks([])

    ax_amp.text(-0.28, 1.02, panel_letter, transform=ax_amp.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_amp.set_title(panel_title, pad=3)

    cbar_amp = fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.02)
    cbar_amp.set_label("Amp.\n(a.u.)")
    cbar_amp.set_ticks([0.0, 1.0])
    cbar_phase = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.02)
    cbar_phase.set_label("Phase")
    cbar_phase.set_ticks([0.0, 2.0 * np.pi])
    cbar_phase.set_ticklabels(["0", "2π"])


def _plot_bottom_cross_section_panel(
    fig,
    subplotspec,
    *,
    volume_xyz: np.ndarray,
    z_positions: np.ndarray,
    dx: float,
    y_index: int | None,
    panel_letter: str,
    show_layer_markers: bool,
    amp_cmap: str,
    phase_cmap: str,
):
    """Plot bottom panel (C or D): x-z amplitude/phase with input-output profiles."""

    gs = subplotspec.subgridspec(
        3,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.24, 1.0, 0.24],
        wspace=0.20,
        hspace=0.06,
    )
    ax_in = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[1, 0])
    ax_out = fig.add_subplot(gs[2, 0])
    ax_phase_head = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_phase_tail = fig.add_subplot(gs[2, 1])

    if y_index is None:
        y_index = int(volume_xyz.shape[1] // 2)
    xz_complex = np.asarray(volume_xyz[:, y_index, :]).T  # (z, x)
    if show_layer_markers:
        xz_complex = _blur_complex2d(xz_complex, ky=5, kx=7)
    else:
        xz_complex = _blur_complex2d(xz_complex, ky=3, kx=5)
    xz_amp = np.abs(xz_complex)
    xz_phase = np.mod(np.angle(xz_complex), 2.0 * np.pi)
    xz_amp_disp = _normalize_for_display(xz_amp, percentile=99.8, gamma=0.72)

    N = volume_xyz.shape[0]
    x0, x1 = _extent_mm(N, dx)
    z0 = float(np.min(z_positions) * 1e3)
    z1 = float(np.max(z_positions) * 1e3)

    im_amp = ax_amp.imshow(
        xz_amp_disp,
        cmap=amp_cmap,
        origin="upper",
        aspect="auto",
        extent=[x0, x1, z0, z1],
        vmin=0.0,
        vmax=1.0,
    )
    im_phase = ax_phase.imshow(
        xz_phase,
        cmap=phase_cmap,
        origin="upper",
        aspect="auto",
        extent=[x0, x1, z0, z1],
        vmin=0.0,
        vmax=2.0 * np.pi,
    )

    input_profile = _normalize_for_display(xz_amp[0], percentile=100.0, gamma=1.0)
    output_profile = _normalize_for_display(xz_amp[-1], percentile=100.0, gamma=1.0)
    x_mm = np.linspace(x0, x1, N)
    ax_in.plot(x_mm, input_profile, color="#89b6d6", lw=1.0)
    ax_out.plot(x_mm, output_profile, color="#89b6d6", lw=1.0)

    ax_in.set_title("Input Amp.", fontsize=9, pad=1)
    ax_phase_head.set_title("Input Phase", fontsize=9, pad=1)
    ax_out.set_title("Output", fontsize=9, pad=1)

    for ax in (ax_in, ax_out):
        ax.set_xlim(x0, x1)
        ax.set_ylim(0.0, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in (ax_phase_head, ax_phase_tail):
        ax.axis("off")

    ax_amp.text(-0.30, 1.02, panel_letter, transform=ax_amp.transAxes, fontsize=14, fontweight="bold", va="top")
    ax_amp.set_xlabel("x")
    ax_amp.set_ylabel("z")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("z")

    if show_layer_markers:
        # z_positions[1:] correspond to L1..Ln.
        for idx, z in enumerate(z_positions[1:], start=1):
            z_mm = float(z * 1e3)
            ax_amp.plot([x1 - 0.7, x1 - 0.2], [z_mm, z_mm], color="white", lw=0.8)
            ax_amp.text(x1 + 0.4, z_mm, f"$L_{idx}$", va="center", fontsize=7)

    cbar_amp = fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.02)
    cbar_amp.set_label("Amp.\n(a.u.)")
    cbar_amp.set_ticks([0.0, 1.0])
    cbar_phase = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.02)
    cbar_phase.set_label("Phase")
    cbar_phase.set_ticks([0.0, 2.0 * np.pi])
    cbar_phase.set_ticklabels(["0", "2π"])


def plot_wave_propagation_figure_s6(
    d2nn_volume_xyz: np.ndarray,
    free_space_volume_xyz: np.ndarray,
    *,
    d2nn_z_positions: np.ndarray,
    free_space_z_positions: np.ndarray,
    dx: float,
    y_index: int | None = None,
    amp_cmap: str = "magma",
    phase_cmap: str = "hsv",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Create Figure S6-like 4-panel visualization (A/B/C/D)."""

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    apply_style()
    fig = plt.figure(figsize=(11.6, 12.0))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], wspace=0.18, hspace=0.20)

    _plot_top_stack_panel(
        fig,
        outer[0, 0],
        volume_xyz=d2nn_volume_xyz,
        z_positions=d2nn_z_positions,
        panel_letter="A",
        panel_title="D$^2$NN Amplitude Imaging",
        amp_cmap=amp_cmap,
        phase_cmap=phase_cmap,
    )
    _plot_top_stack_panel(
        fig,
        outer[0, 1],
        volume_xyz=free_space_volume_xyz,
        z_positions=free_space_z_positions,
        panel_letter="B",
        panel_title="Free-space Propagation",
        amp_cmap=amp_cmap,
        phase_cmap=phase_cmap,
    )
    _plot_bottom_cross_section_panel(
        fig,
        outer[1, 0],
        volume_xyz=d2nn_volume_xyz,
        z_positions=d2nn_z_positions,
        dx=dx,
        y_index=y_index,
        panel_letter="C",
        show_layer_markers=True,
        amp_cmap=amp_cmap,
        phase_cmap=phase_cmap,
    )
    _plot_bottom_cross_section_panel(
        fig,
        outer[1, 1],
        volume_xyz=free_space_volume_xyz,
        z_positions=free_space_z_positions,
        dx=dx,
        y_index=y_index,
        panel_letter="D",
        show_layer_markers=False,
        amp_cmap=amp_cmap,
        phase_cmap=phase_cmap,
    )

    # Middle dashed divider between D2NN and free-space columns.
    divider = mlines.Line2D([0.5, 0.5], [0.06, 0.96], transform=fig.transFigure, ls="--", lw=1.0, color="0.55", alpha=0.8)
    fig.add_artist(divider)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
