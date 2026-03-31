"""Simulation configuration for FSO beam propagation."""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class SimulationConfig:
    """Parameters for a single FSO turbulence propagation run."""

    # Propagation geometry
    Dz: float  # propagation distance [m]
    Cn2: float  # refractive-index structure constant [m^{-2/3}]

    # Source
    theta_div: float  # Full-angle far-field divergence [rad]

    # Observation plane
    D_roi: float  # observation computational window diameter [m]
    delta_n: float  # observation-plane grid spacing [m]
    D_aperture: Optional[float] = None  # coherence-factor verification aperture [m]

    # Simulation
    N: Optional[int] = None  # grid points (power of 2). None = auto
    n_reals: int = 20  # turbulence realizations

    # Fixed
    wvl: float = 1550e-9  # wavelength [m]
    dtype: str = "complex128"

    @property
    def k(self) -> float:
        """Wavenumber [rad/m]."""
        return 2 * math.pi / self.wvl

    @property
    def w0(self) -> float:
        """Beam waist from full-angle divergence: w0 = 2*lambda / (pi * theta_div)."""
        return 2 * self.wvl / (math.pi * self.theta_div)

    @property
    def D1(self) -> float:
        """Source aperture diameter: 4*w0 (99.97% energy)."""
        return 4 * self.w0
