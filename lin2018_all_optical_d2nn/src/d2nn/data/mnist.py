"""MNIST dataset wrappers for D2NN amplitude encoding."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from d2nn.data.preprocess import to_input_field


class MNISTFieldDataset(Dataset):
    """MNIST -> complex field dataset."""

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        download: bool,
        N: int,
        object_size: int,
        binarize: bool = True,
    ):
        try:
            from torchvision import datasets, transforms
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("torchvision is required for MNIST dataset") from exc

        self.ds = datasets.MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.N = N
        self.object_size = object_size
        self.binarize = binarize

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.ds[idx]
        image = x.squeeze(0)
        field = to_input_field(
            image,
            encoding="amplitude",
            N=self.N,
            object_size=self.object_size,
            binarize=self.binarize,
        )
        return field, int(y)
