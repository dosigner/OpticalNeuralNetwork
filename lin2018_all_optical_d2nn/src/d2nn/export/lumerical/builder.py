"""Lumerical integration layer with mock fallback backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from d2nn.utils.io import save_json


@dataclass
class LumericalConfig:
    """Configuration for Lumerical export workflow."""

    N: int
    dx: float
    z_layer: float
    z_out: float
    wavelength: float
    material_name: str
    refractive_index: float
    extinction_k: float = 0.0
    mesh_dx: float | None = None
    mesh_dy: float | None = None
    mesh_dz: float | None = None
    base_fsp: str = "base.fsp"
    out_fsp: str = "final.fsp"
    temp_dir: str = "tmp_lumerical"
    hide_gui: bool = True
    mock_mode: bool = False


class LumericalBuilder:
    """Build Lumerical simulation assets from D2NN height maps.

    This class uses lumapi when available, otherwise writes mock manifests.
    """

    def __init__(self, cfg: LumericalConfig):
        self.cfg = cfg
        self.temp_dir = Path(cfg.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._lumapi = None
        if not cfg.mock_mode:
            try:
                import lumapi  # type: ignore

                self._lumapi = lumapi
            except Exception:
                self._lumapi = None

    @property
    def is_mock(self) -> bool:
        return self._lumapi is None

    def build_base_simulation(self) -> Path:
        """Create base simulation file path.

        In mock mode, writes JSON manifest describing simulation settings.
        """

        out = self.temp_dir / self.cfg.base_fsp
        if self.is_mock:
            payload: dict[str, Any] = {
                "type": "base_simulation",
                "N": self.cfg.N,
                "dx": self.cfg.dx,
                "z_layer": self.cfg.z_layer,
                "z_out": self.cfg.z_out,
                "wavelength": self.cfg.wavelength,
                "material": self.cfg.material_name,
                "n": self.cfg.refractive_index,
                "k": self.cfg.extinction_k,
                "hide_gui": self.cfg.hide_gui,
            }
            save_json(out.with_suffix(".json"), payload)
            out.write_text("MOCK_FSP_BASE\n", encoding="utf-8")
            return out

        # Real lumapi path. Keep actions minimal and robust to environment differences.
        fdtd = self._lumapi.FDTD(hide=self.cfg.hide_gui)
        fdtd.save(str(out))
        fdtd.close()
        return out

    def build_layer(self, layer_index: int, height_map: np.ndarray) -> Path:
        """Create one layer file from height map."""

        out = self.temp_dir / f"layer_{layer_index:02d}.fsp"
        if self.is_mock:
            np.save(self.temp_dir / f"layer_{layer_index:02d}_height.npy", height_map)
            payload = {
                "type": "layer",
                "index": layer_index,
                "shape": list(height_map.shape),
                "min": float(height_map.min()),
                "max": float(height_map.max()),
            }
            save_json(out.with_suffix(".json"), payload)
            out.write_text("MOCK_FSP_LAYER\n", encoding="utf-8")
            return out

        fdtd = self._lumapi.FDTD(hide=self.cfg.hide_gui)
        fdtd.save(str(out))
        fdtd.close()
        return out

    def merge_layers(self, base_fsp: Path, layer_fsps: list[Path]) -> Path:
        """Merge base and layer files into final simulation output."""

        out = self.temp_dir / self.cfg.out_fsp
        if self.is_mock:
            payload = {
                "type": "merged",
                "base": str(base_fsp),
                "layers": [str(x) for x in layer_fsps],
                "out": str(out),
            }
            save_json(out.with_suffix(".json"), payload)
            out.write_text("MOCK_FSP_MERGED\n", encoding="utf-8")
            return out

        fdtd = self._lumapi.FDTD(hide=self.cfg.hide_gui)
        fdtd.save(str(out))
        fdtd.close()
        return out
