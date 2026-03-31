"""NPZ cache IO for deterministic beam pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _split_complex(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    array = field.detach().cpu().numpy()
    return array.real.astype(np.float32), array.imag.astype(np.float32)


def _join_complex(real: np.ndarray, imag: np.ndarray) -> torch.Tensor:
    return torch.complex(torch.from_numpy(real), torch.from_numpy(imag))


def write_pair_npz(
    path: str | Path,
    *,
    u_vacuum: torch.Tensor,
    u_turb: torch.Tensor,
    x_m: np.ndarray,
    y_m: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write a deterministic vacuum/turbulence pair to NPZ."""
    u_v_real, u_v_imag = _split_complex(u_vacuum)
    u_t_real, u_t_imag = _split_complex(u_turb)
    np.savez_compressed(
        path,
        u_vacuum_real=u_v_real,
        u_vacuum_imag=u_v_imag,
        u_turb_real=u_t_real,
        u_turb_imag=u_t_imag,
        x_m=np.asarray(x_m),
        y_m=np.asarray(y_m),
        metadata_json=np.array(json.dumps(metadata)),
    )


def read_pair_npz(path: str | Path) -> dict[str, Any]:
    """Read a deterministic vacuum/turbulence pair from NPZ."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        return {
            "u_vacuum": _join_complex(data["u_vacuum_real"], data["u_vacuum_imag"]),
            "u_turb": _join_complex(data["u_turb_real"], data["u_turb_imag"]),
            "x_m": data["x_m"],
            "y_m": data["y_m"],
            "metadata": metadata,
        }
