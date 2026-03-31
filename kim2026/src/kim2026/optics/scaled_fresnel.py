"""Arbitrary-window Fresnel propagation for zoom paths via a direct integral."""

from __future__ import annotations

import math
from typing import Any

import torch

from kim2026.optics.gaussian_beam import coordinate_axis

_OPERATOR_CACHE: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}


def _complex_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.complex128:
        return "complex128"
    return "complex64"


def _build_cached_operators(
    *,
    n: int,
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    z_m: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    key = (
        int(n),
        float(wavelength_m),
        float(source_window_m),
        float(destination_window_m),
        float(z_m),
        str(device),
        _complex_dtype_name(dtype),
    )
    cached = _OPERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    if dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        real_dtype = torch.float32

    x_in = coordinate_axis(n, source_window_m, device=device).to(real_dtype)
    x_out = coordinate_axis(n, destination_window_m, device=device).to(real_dtype)
    k = 2.0 * math.pi / float(wavelength_m)
    dx_in = float(source_window_m) / float(n)

    phase_in = (k / (2.0 * float(z_m))) * (x_in.square().unsqueeze(0) + x_in.square().unsqueeze(1))
    phase_out = (k / (2.0 * float(z_m))) * (x_out.square().unsqueeze(0) + x_out.square().unsqueeze(1))
    kernel_phase = (-2.0 * math.pi / (float(wavelength_m) * float(z_m))) * torch.outer(x_out, x_in)

    q_in = torch.exp(1j * phase_in).to(dtype)
    q_out = torch.exp(1j * phase_out).to(dtype)
    kernel = torch.exp(1j * kernel_phase).to(dtype)
    kernel_t = kernel.transpose(-1, -2).contiguous()
    prefactor = torch.tensor(
        (dx_in * dx_in) * math.e ** (1j * k * float(z_m)) / (1j * float(wavelength_m) * float(z_m)),
        dtype=dtype,
        device=device,
    )

    cached = {
        "q_in": q_in,
        "q_out": q_out,
        "kernel": kernel,
        "kernel_t": kernel_t,
        "prefactor": prefactor,
    }
    _OPERATOR_CACHE[key] = cached
    return cached


def scaled_fresnel_propagate(
    field: torch.Tensor,
    *,
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    z_m: float,
) -> torch.Tensor:
    """Propagate between planes with arbitrary windows using the Fresnel integral."""
    if field.ndim < 2 or field.shape[-1] != field.shape[-2]:
        raise ValueError("field must be square in the last two dimensions")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0")
    if source_window_m <= 0.0:
        raise ValueError("source_window_m must be > 0")
    if destination_window_m <= 0.0:
        raise ValueError("destination_window_m must be > 0")
    if z_m <= 0.0:
        raise ValueError("z_m must be > 0")

    n = int(field.shape[-1])
    original_shape = field.shape
    flattened = field.reshape(-1, n, n)
    dtype = field.dtype if field.dtype in (torch.complex64, torch.complex128) else torch.complex64
    flattened = flattened.to(dtype)
    operators = _build_cached_operators(
        n=n,
        wavelength_m=wavelength_m,
        source_window_m=source_window_m,
        destination_window_m=destination_window_m,
        z_m=z_m,
        device=flattened.device,
        dtype=dtype,
    )

    work = flattened * operators["q_in"].unsqueeze(0)
    work = torch.matmul(operators["kernel"].unsqueeze(0), work)
    work = torch.matmul(work, operators["kernel_t"].unsqueeze(0))
    work = operators["prefactor"] * operators["q_out"].unsqueeze(0) * work
    return work.reshape(original_shape)


def warmup_scaled_fresnel(
    *,
    n: int,
    wavelength_m: float,
    source_window_m: float,
    destination_window_m: float,
    z_m: float,
    iterations: int,
    device: torch.device | str,
) -> None:
    """Pre-build fixed-shape Fresnel operators and execute a dry pass."""
    if int(iterations) <= 0:
        return
    field = torch.zeros(1, n, n, dtype=torch.complex64, device=device)
    for _ in range(int(iterations)):
        _ = scaled_fresnel_propagate(
            field,
            wavelength_m=wavelength_m,
            source_window_m=source_window_m,
            destination_window_m=destination_window_m,
            z_m=z_m,
        )
