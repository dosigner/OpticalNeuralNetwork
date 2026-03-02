"""Photorefractive SBN nonlinearity."""

from __future__ import annotations

import math

import torch
from torch import nn

from tao2019_fd2nn.utils.math import intensity


class SBNNonlinearity(nn.Module):
    """Intensity-dependent phase modulation (background perturbation).

      u_out = u * exp(i * phi_scale * eta/(1+eta))
      eta   = max((|u|^2 - I0) / I_sat, 0)

    If `learnable_saturation=True`, I_sat is a trainable parameter
    (log-parameterized to stay positive).  The optimizer will drive it
    toward the scale at which the SBN operates in its sensitive regime.

    If physical parameters are provided, phi_scale is computed as:
      phi_scale = (2*pi/lambda) * d * kappa * E_app
    """

    def __init__(
        self,
        *,
        phi_max: float = torch.pi,
        background_intensity: float = 0.0,
        saturation_intensity: float = 1.0,
        clamp_negative_perturbation: bool = True,
        learnable_saturation: bool = False,
        voltage_v: float | None = None,
        electrode_gap_m: float | None = None,
        e_app_v_per_m: float | None = None,
        kappa_m_per_v: float | None = None,
        thickness_m: float | None = None,
        wavelength_m: float | None = None,
    ) -> None:
        super().__init__()
        self.phi_max = float(phi_max)
        self.background_intensity = float(background_intensity)
        self.clamp_negative_perturbation = bool(clamp_negative_perturbation)
        self.learnable_saturation = bool(learnable_saturation)

        init_isat = max(float(saturation_intensity), 1e-12)
        if self.learnable_saturation:
            self.log_isat = nn.Parameter(torch.tensor(math.log(init_isat)))
        else:
            self.register_buffer("_isat_fixed", torch.tensor(init_isat))

        self.phi_scale_rad = self._resolve_phi_scale(
            phi_max=float(phi_max),
            voltage_v=voltage_v,
            electrode_gap_m=electrode_gap_m,
            e_app_v_per_m=e_app_v_per_m,
            kappa_m_per_v=kappa_m_per_v,
            thickness_m=thickness_m,
            wavelength_m=wavelength_m,
        )

    @staticmethod
    def _resolve_phi_scale(
        *,
        phi_max: float,
        voltage_v: float | None,
        electrode_gap_m: float | None,
        e_app_v_per_m: float | None,
        kappa_m_per_v: float | None,
        thickness_m: float | None,
        wavelength_m: float | None,
    ) -> float:
        if e_app_v_per_m is None and voltage_v is not None and electrode_gap_m is not None:
            gap = max(float(electrode_gap_m), 1e-30)
            e_app_v_per_m = float(voltage_v) / gap
        if (
            e_app_v_per_m is not None
            and kappa_m_per_v is not None
            and thickness_m is not None
            and wavelength_m is not None
        ):
            return (2.0 * math.pi / float(wavelength_m)) * float(thickness_m) * float(kappa_m_per_v) * float(e_app_v_per_m)
        return float(phi_max)

    @property
    def saturation_intensity(self) -> torch.Tensor:
        if self.learnable_saturation:
            return torch.exp(self.log_isat)
        return self._isat_fixed

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        I_sat = self.saturation_intensity
        eta = (intensity(field) - float(self.background_intensity)) / I_sat
        if self.clamp_negative_perturbation:
            eta = torch.relu(eta)
        delta_phi = self.phi_scale_rad * (eta / (1.0 + eta))
        return field * torch.exp(1j * delta_phi.to(field.real.dtype))
