"""Phase screen synthesis for modified von Karman turbulence."""

from __future__ import annotations

import math

import torch

from kim2026.turbulence.von_karman import phase_psd


def _frequency_grid(n: int, window_m: float, *, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    dx = float(window_m) / float(n)
    freq = torch.fft.fftfreq(n, d=dx, device=device, dtype=torch.float64)
    fy, fx = torch.meshgrid(freq, freq, indexing="ij")
    return fx, fy


def generate_phase_screen(
    *,
    n: int,
    window_m: float,
    wavelength_m: float,
    cn2: float,
    path_segment_m: float,
    outer_scale_m: float,
    inner_scale_m: float,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a deterministic real-valued turbulence phase screen."""
    if float(cn2) == 0.0:
        return torch.zeros(n, n, dtype=torch.float32, device=device)

    df = 1.0 / float(window_m)

    fx, fy = _frequency_grid(n, window_m, device=device)
    psd = phase_psd(
        fx=fx,
        fy=fy,
        wavelength_m=wavelength_m,
        cn2=cn2,
        path_length_m=path_segment_m,
        outer_scale_m=outer_scale_m,
        inner_scale_m=inner_scale_m,
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    noise_real = torch.randn(n, n, generator=gen, dtype=torch.float64)
    noise_imag = torch.randn(n, n, generator=gen, dtype=torch.float64)
    if device is not None:
        noise_real = noise_real.to(device)
        noise_imag = noise_imag.to(device)
    noise = torch.complex(noise_real, noise_imag)

    spectrum = noise * torch.sqrt(psd).to(noise.dtype) * df
    phase = torch.fft.ifft2(spectrum).real * (n * n)
    phase = phase - phase.mean()
    return phase.to(torch.float32)
