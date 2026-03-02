"""Core D2NN model supporting real/fourier/hybrid configurations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tao2019_fd2nn.models.nonlinearity_sbn import SBNNonlinearity
from tao2019_fd2nn.models.phase_mask import PhaseMask
from tao2019_fd2nn.optics.fft2c import fft2c, ifft2c
from tao2019_fd2nn.optics.asm import asm_propagate, asm_transfer_function
from tao2019_fd2nn.optics.lens_2f import lens_2f_forward, lens_2f_inverse


@dataclass(frozen=True)
class Fd2nnConfig:
    """Scalar runtime model config parsed from spec-style YAML."""

    N: int
    dx_m: float
    wavelength_m: float
    z_layer_m: float
    z_out_m: float
    num_layers: int
    phase_max: float
    phase_constraint: str = "sigmoid"
    phase_init: str = "uniform"
    model_type: str = "fd2nn"
    na: float | None = None
    evanescent: str = "mask"
    dtype: str = "complex64"
    use_dual_2f: bool = False
    dual_2f_f1_m: float | None = None
    dual_2f_f2_m: float | None = None
    dual_2f_na1: float | None = None
    dual_2f_na2: float | None = None
    dual_2f_apply_scaling: bool = False
    hybrid_sequence: tuple[str, ...] = ()
    sbn_enabled: bool = False
    sbn_phi_max: float = float(torch.pi)
    sbn_position: str = "rear"
    sbn_background_intensity: float = 0.0
    sbn_saturation_intensity: float = 1.0
    sbn_clamp_negative_perturbation: bool = True
    sbn_learnable_saturation: bool = False
    sbn_voltage_v: float | None = None
    sbn_electrode_gap_m: float | None = None
    sbn_e_app_v_per_m: float | None = None
    sbn_kappa_m_per_v: float | None = None
    sbn_thickness_m: float | None = None
    sbn_wavelength_m: float | None = None


class Fd2nnModel(nn.Module):
    """Diffractive model with domain switching and optional SBN."""

    def __init__(self, cfg: Fd2nnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                PhaseMask(
                    N=cfg.N,
                    phase_max=cfg.phase_max,
                    constraint_mode=cfg.phase_constraint,
                    init_mode=cfg.phase_init,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.sbn = (
            SBNNonlinearity(
                phi_max=cfg.sbn_phi_max,
                background_intensity=cfg.sbn_background_intensity,
                saturation_intensity=cfg.sbn_saturation_intensity,
                clamp_negative_perturbation=cfg.sbn_clamp_negative_perturbation,
                learnable_saturation=cfg.sbn_learnable_saturation,
                voltage_v=cfg.sbn_voltage_v,
                electrode_gap_m=cfg.sbn_electrode_gap_m,
                e_app_v_per_m=cfg.sbn_e_app_v_per_m,
                kappa_m_per_v=cfg.sbn_kappa_m_per_v,
                thickness_m=cfg.sbn_thickness_m,
                wavelength_m=cfg.sbn_wavelength_m,
            )
            if cfg.sbn_enabled
            else None
        )

    def _propagate(self, field: torch.Tensor, z_m: float, *, dx_m: float, na: float | None) -> torch.Tensor:
        H = asm_transfer_function(
            N=self.cfg.N,
            dx_m=float(dx_m),
            wavelength_m=self.cfg.wavelength_m,
            z_m=z_m,
            na=na,
            evanescent=self.cfg.evanescent,
            dtype=self.cfg.dtype,
            device=field.device,
        )
        return asm_propagate(field, H)

    def _target_domain(self, layer_idx: int) -> str:
        if self.cfg.model_type == "fd2nn":
            return "fourier"
        if self.cfg.model_type == "real_d2nn":
            return "real"
        if self.cfg.model_type == "hybrid_d2nn":
            if not self.cfg.hybrid_sequence:
                return "fourier" if (layer_idx % 2 == 0) else "real"
            if layer_idx < len(self.cfg.hybrid_sequence):
                return self.cfg.hybrid_sequence[layer_idx]
            return self.cfg.hybrid_sequence[-1]
        raise ValueError(f"unsupported model_type: {self.cfg.model_type}")

    def _switch_domain(
        self,
        field: torch.Tensor,
        *,
        current: str,
        target: str,
        dx_m: float,
    ) -> tuple[torch.Tensor, str, float]:
        if current == target:
            return field, current, float(dx_m)
        if current == "real" and target == "fourier":
            if not self.cfg.use_dual_2f:
                return fft2c(field), "fourier", float(dx_m)
            out, dx_fourier = lens_2f_forward(
                field,
                dx_in_m=float(dx_m),
                wavelength_m=self.cfg.wavelength_m,
                f_m=float(self.cfg.dual_2f_f1_m or 1.0e-3),
                na=self.cfg.dual_2f_na1,
                apply_scaling=bool(self.cfg.dual_2f_apply_scaling),
            )
            return out, "fourier", float(dx_fourier)
        if current == "fourier" and target == "real":
            if not self.cfg.use_dual_2f:
                return ifft2c(field), "real", float(dx_m)
            out, dx_real = lens_2f_inverse(
                field,
                dx_fourier_m=float(dx_m),
                wavelength_m=self.cfg.wavelength_m,
                f_m=float(self.cfg.dual_2f_f2_m or self.cfg.dual_2f_f1_m or 1.0e-3),
                na=self.cfg.dual_2f_na2,
                apply_scaling=bool(self.cfg.dual_2f_apply_scaling),
            )
            return out, "real", float(dx_real)
        raise ValueError(f"invalid domain transition: {current} -> {target}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        domain = "real"
        out = field
        dx_m = float(self.cfg.dx_m)
        if self.cfg.model_type == "fd2nn" and self.cfg.use_dual_2f:
            out, domain, dx_m = self._switch_domain(out, current=domain, target="fourier", dx_m=dx_m)

        for idx, layer in enumerate(self.layers):
            target_domain = self._target_domain(idx)
            out, domain, dx_m = self._switch_domain(out, current=domain, target=target_domain, dx_m=dx_m)
            out = self._propagate(out, self.cfg.z_layer_m, dx_m=dx_m, na=self.cfg.na)
            out = layer(out)
            if self.sbn is not None and self.cfg.sbn_position == "per_layer":
                out = self.sbn(out)

        if self.sbn is not None and self.cfg.sbn_position == "rear":
            out = self.sbn(out)

        out = self._propagate(out, self.cfg.z_out_m, dx_m=dx_m, na=self.cfg.na)

        # Sensor plane is modeled in real domain.
        if domain == "fourier":
            out, domain, dx_m = self._switch_domain(out, current=domain, target="real", dx_m=dx_m)
        return out
