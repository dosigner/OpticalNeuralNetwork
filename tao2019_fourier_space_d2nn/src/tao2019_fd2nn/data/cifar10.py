"""CIFAR10-derived saliency training dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from tao2019_fd2nn.data.preprocess import resize_and_pad_square, to_complex_field
from tao2019_fd2nn.data.saliency_gt import SaliencyGtBuilder


class Cifar10SaliencyDataset(Dataset):
    """Map CIFAR image to (field, saliency target)."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        N: int,
        object_size: int,
        foreground_class: int = 3,
        gt_source: str = "class_conditioned_intensity",
        gt_params: dict[str, object] | None = None,
    ) -> None:
        self.base = datasets.CIFAR10(root=root, train=train, download=download)
        self.N = int(N)
        self.object_size = int(object_size)
        self.foreground_class = int(foreground_class)
        source = "class_conditioned_intensity" if gt_source == "dataset" else str(gt_source)
        self.gt_builder = SaliencyGtBuilder(
            source=source,
            params=(gt_params or {}),
            foreground_class=self.foreground_class,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self.base[index]
        arr = np.asarray(img).astype(np.float32) / 255.0
        gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        gray_t = torch.from_numpy(gray)
        amp = resize_and_pad_square(gray_t, out_size=self.N, object_size=self.object_size)
        field = to_complex_field(amp)
        target_np = self.gt_builder.build(image=arr, label=int(label))
        target_t = torch.from_numpy(target_np).to(torch.float32)
        target = resize_and_pad_square(target_t, out_size=self.N, object_size=self.object_size).clamp(0.0, 1.0)
        return field, target
