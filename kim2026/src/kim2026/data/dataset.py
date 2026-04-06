"""Datasets backed by cached NPZ field pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from kim2026.data.canonical_pupil import (
    SUPPORTED_PLANE_SELECTORS,
    is_canonical_pupil_npz,
    read_canonical_pupil_npz,
)
from kim2026.data.npz_pairs import read_pair_npz
from kim2026.optics.beam_reducer import BeamReducerPlane, apply_beam_reducer


class CachedFieldDataset(Dataset):
    """Dataset that reads deterministic cached field pairs from disk."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        manifest_path: str | Path,
        split: str,
        plane_selector: str = "stored",
        reducer_pad_factor: int = 2,
    ) -> None:
        if plane_selector not in SUPPORTED_PLANE_SELECTORS:
            raise ValueError(f"unsupported plane_selector='{plane_selector}'")
        self.cache_dir = Path(cache_dir)
        self.plane_selector = plane_selector
        self.reducer_pad_factor = int(reducer_pad_factor)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        all_entries = [self.cache_dir / name for name in manifest.get(split, [])]
        missing = [p for p in all_entries if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(all_entries)} cache files missing for split '{split}'. "
                f"First missing: {missing[0]}. Is cache generation still running?"
            )
        self.entries = all_entries
        self._canonical_pupil = bool(all_entries) and is_canonical_pupil_npz(all_entries[0])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._canonical_pupil:
            record = read_canonical_pupil_npz(self.entries[index])
            if self.plane_selector in ("stored", "pupil"):
                return {
                    "u_vacuum": record["u_vacuum_pupil"].to(dtype=record["u_vacuum_pupil"].dtype),
                    "u_turb": record["u_turb_pupil"].to(dtype=record["u_turb_pupil"].dtype),
                    "metadata": record["metadata"],
                }
            if self.plane_selector == "reduced_ideal":
                metadata = dict(record["metadata"])
                input_plane = BeamReducerPlane(
                    window_m=float(metadata["receiver_window_m"]),
                    n=int(record["u_vacuum_pupil"].shape[-1]),
                    aperture_diameter_m=float(metadata["telescope_diameter_m"]),
                )
                output_plane = BeamReducerPlane(
                    window_m=float(metadata["reducer_output_window_m"]),
                    n=int(record["u_vacuum_pupil"].shape[-1]),
                    aperture_diameter_m=float(metadata["reducer_output_window_m"]),
                )
                reduced_vacuum = apply_beam_reducer(
                    record["u_vacuum_pupil"],
                    input_plane=input_plane,
                    output_plane=output_plane,
                    pad_factor=self.reducer_pad_factor,
                )
                reduced_turb = apply_beam_reducer(
                    record["u_turb_pupil"],
                    input_plane=input_plane,
                    output_plane=output_plane,
                    pad_factor=self.reducer_pad_factor,
                )
                metadata["plane"] = "reduced_ideal"
                metadata["reducer_pad_factor"] = self.reducer_pad_factor
                return {
                    "u_vacuum": reduced_vacuum.to(dtype=reduced_vacuum.dtype),
                    "u_turb": reduced_turb.to(dtype=reduced_turb.dtype),
                    "metadata": metadata,
                }
            raise ValueError(f"unsupported plane_selector='{self.plane_selector}'")

        record = read_pair_npz(self.entries[index])
        if self.plane_selector == "pupil":
            raise ValueError("legacy reduced-plane datasets cannot be loaded with plane_selector='pupil'")
        return {
            "u_vacuum": record["u_vacuum"].to(dtype=record["u_vacuum"].dtype),
            "u_turb": record["u_turb"].to(dtype=record["u_turb"].dtype),
            "metadata": record["metadata"],
        }
