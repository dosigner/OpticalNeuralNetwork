"""MNIST dataset with amplitude preprocessing for D2NN simulation."""

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision.transforms import InterpolationMode


class MNISTAmplitude(torch.utils.data.Dataset):
    """MNIST images preprocessed as amplitude inputs for optical D2NN.

    Pipeline per image:
        1. Load 28x28 grayscale → ToTensor (already [0,1])
        2. Bilinear resize to resize_to x resize_to
        3. Zero-pad (centered) to final_size x final_size
        4. Build binary support mask (amplitude > 0)
        5. Return {"amplitude", "mask", "label"}
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize_to: int = 160,
        final_size: int = 240,
        download: bool = True,
    ):
        super().__init__()
        self.resize_to = resize_to
        self.final_size = final_size

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # (1, 28, 28), [0, 1]
            torchvision.transforms.Resize(
                (resize_to, resize_to),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ])

        if split in ("train", "val"):
            full_train = torchvision.datasets.MNIST(
                root=root, train=True, download=download, transform=transform
            )
            if split == "train":
                self.dataset = torch.utils.data.Subset(full_train, range(50000))
            else:
                self.dataset = torch.utils.data.Subset(full_train, range(50000, 60000))
        elif split == "test":
            self.dataset = torchvision.datasets.MNIST(
                root=root, train=False, download=download, transform=transform
            )
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

        # Compute symmetric padding
        pad_total = final_size - resize_to
        self.pad_each = pad_total // 2  # 40 for default sizes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.dataset[idx]  # (1, resize_to, resize_to)

        # Zero-pad to final_size x final_size (pad order: left, right, top, bottom)
        p = self.pad_each
        amplitude = F.pad(image, (p, p, p, p), mode="constant", value=0.0)

        mask = (amplitude > 0).float()

        return {"amplitude": amplitude, "mask": mask, "label": label}
