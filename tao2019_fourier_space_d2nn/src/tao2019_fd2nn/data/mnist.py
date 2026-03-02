"""MNIST amplitude-encoded dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from tao2019_fd2nn.data.preprocess import resize_and_pad_square, to_complex_field


class MnistAmplitudeDataset(Dataset):
    """MNIST digits -> complex fields for optical simulation."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        N: int,
        object_size: int,
        binarize: bool = True,
    ) -> None:
        self.base = datasets.MNIST(root=root, train=train, download=download)
        self.N = int(N)
        self.object_size = int(object_size)
        self.binarize = bool(binarize)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.base[index]
        arr = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        if self.binarize:
            arr = (arr > 0.5).to(torch.float32)
        amp = resize_and_pad_square(arr, out_size=self.N, object_size=self.object_size)
        field = to_complex_field(amp)
        return field, torch.tensor(label, dtype=torch.long)
