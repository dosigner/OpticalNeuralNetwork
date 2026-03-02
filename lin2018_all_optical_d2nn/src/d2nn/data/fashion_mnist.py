"""Fashion-MNIST dataset wrappers for D2NN phase encoding."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from d2nn.data.preprocess import to_input_field


class FashionMNISTFieldDataset(Dataset):
    """Fashion-MNIST -> complex field dataset using phase input encoding."""

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        download: bool,
        N: int,
        object_size: int,
        phase_max: float = 2.0 * torch.pi,
    ):
        try:
            from torchvision import datasets, transforms
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("torchvision is required for FashionMNIST dataset") from exc

        self.ds = datasets.FashionMNIST(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.N = N
        self.object_size = object_size
        self.phase_max = phase_max

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.ds[idx]
        image = x.squeeze(0)
        field = to_input_field(
            image,
            encoding="phase",
            N=self.N,
            object_size=self.object_size,
            phase_max=self.phase_max,
        )
        return field, int(y)
