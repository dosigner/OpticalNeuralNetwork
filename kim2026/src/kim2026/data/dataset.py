"""Datasets backed by cached NPZ field pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from kim2026.data.npz_pairs import read_pair_npz


class CachedFieldDataset(Dataset):
    """Dataset that reads deterministic cached field pairs from disk."""

    def __init__(self, *, cache_dir: str | Path, manifest_path: str | Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
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

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = read_pair_npz(self.entries[index])
        return {
            "u_vacuum": record["u_vacuum"].to(dtype=record["u_vacuum"].dtype),
            "u_turb": record["u_turb"].to(dtype=record["u_turb"].dtype),
            "metadata": record["metadata"],
        }
