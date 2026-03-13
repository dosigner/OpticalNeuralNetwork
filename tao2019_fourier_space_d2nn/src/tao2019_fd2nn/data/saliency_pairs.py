"""Folder-based saliency pair dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tao2019_fd2nn.data.preprocess import resize_and_pad_square, to_complex_field

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class SaliencyPairsDataset(Dataset):
    """Read aligned image/mask files from directories."""

    def __init__(
        self,
        *,
        image_dir: str | Path,
        mask_dir: str | Path,
        N: int,
        object_size: int,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.N = int(N)
        self.object_size = int(object_size)
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"image_dir not found: {self.image_dir}\n"
                "Expected paired saliency folders:\n"
                "  <root>/images/*.png (or jpg/tif)\n"
                "  <root>/masks/*.png (same filenames)\n"
                "Set explicit paths in YAML with data.image_dir and data.mask_dir."
            )
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"mask_dir not found: {self.mask_dir}\n"
                "Expected paired saliency folders:\n"
                "  <root>/images/*.png (or jpg/tif)\n"
                "  <root>/masks/*.png (same filenames)\n"
                "Set explicit paths in YAML with data.image_dir and data.mask_dir."
            )
        self.image_files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in _EXTS])
        if not self.image_files:
            raise RuntimeError(f"no images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_gray(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[index]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            # Try other extensions (e.g., image=.jpg, mask=.png)
            found = None
            for ext in _EXTS:
                candidate = self.mask_dir / (img_path.stem + ext)
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                raise FileNotFoundError(f"missing mask for image: {img_path.name}")
            mask_path = found

        image = self._load_gray(img_path)
        mask = self._load_gray(mask_path)

        amp = resize_and_pad_square(image, out_size=self.N, object_size=self.object_size)
        tgt = resize_and_pad_square(mask, out_size=self.N, object_size=self.object_size).clamp(0.0, 1.0)
        return to_complex_field(amp), tgt
