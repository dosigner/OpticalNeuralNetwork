"""Folder-based imaging dataset wrapper."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from d2nn.data.preprocess import resize_to_square
from d2nn.physics.apertures import center_pad_2d


class ImageFolderFieldDataset(Dataset):
    """Read grayscale images from a folder and convert to complex fields.

    Input images are used both as illumination object and target intensity.
    The object is amplitude-only (phase=0), matching imaging-lens setup.
    """

    def __init__(self, root: str | Path, *, N: int, object_size: int):
        try:
            from torchvision import datasets, transforms
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("torchvision is required for folder dataset") from exc

        tfm = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        self.ds = datasets.ImageFolder(root=str(root), transform=tfm)
        self.N = N
        self.object_size = object_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.ds[idx]
        image = x.squeeze(0)

        object_plane = resize_to_square(image, self.object_size)
        object_plane = center_pad_2d(object_plane, target_N=self.N).clamp(0.0, 1.0)

        field = torch.complex(object_plane, torch.zeros_like(object_plane)).to(torch.complex64)
        target = object_plane
        return field, target
