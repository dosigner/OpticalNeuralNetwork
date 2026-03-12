"""Tests for the MNIST / data preprocessing pipeline.

All tests avoid downloading real MNIST data — they mock what is needed.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import pytest

from luo2022_d2nn.data.mnist import MNISTAmplitude
from luo2022_d2nn.data.resolution_targets import generate_grating_target
from luo2022_d2nn.data.masks import make_support_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_28x28(value: float = 1.0) -> torch.Tensor:
    """Return a (1, 28, 28) tensor filled with *value*."""
    return torch.full((1, 28, 28), value)


# ---------------------------------------------------------------------------
# MNIST resize / pad tests (unit-level, no dataset download)
# ---------------------------------------------------------------------------

class TestMNISTResize:
    def test_mnist_resize_shape(self):
        """Bilinear resize from 28×28 → 160×160."""
        img = _fake_28x28(0.5)
        resizer = T.Resize(
            (160, 160),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized = resizer(img)
        assert resized.shape == (1, 160, 160)

    def test_mnist_padding_shape(self):
        """Zero-pad from 160×160 → 240×240 with 40 px border."""
        img = torch.rand(1, 160, 160)
        padded = F.pad(img, (40, 40, 40, 40), mode="constant", value=0.0)
        assert padded.shape == (1, 240, 240)
        # Verify the border is zero
        assert padded[0, :40, :].sum() == 0.0
        assert padded[0, 200:, :].sum() == 0.0
        assert padded[0, :, :40].sum() == 0.0
        assert padded[0, :, 200:].sum() == 0.0

    def test_mnist_mask_positive_pixels(self):
        """Mask should be 1 where amplitude > 0, 0 elsewhere."""
        amp = torch.tensor([[[0.0, 0.5], [0.3, 0.0]]])
        expected = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        mask = (amp > 0).float()
        assert torch.equal(mask, expected)


# ---------------------------------------------------------------------------
# Full MNISTAmplitude dataset with mocked MNIST
# ---------------------------------------------------------------------------

class _FakeMNIST(torch.utils.data.Dataset):
    """Stand-in for torchvision.datasets.MNIST so we never download."""

    def __init__(self, *, length: int = 60000, transform=None):
        self.length = length
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Produce a fake 28×28 PIL-like tensor (before transform) or raw
        # Actually, torchvision MNIST returns (PIL.Image, label).
        # The transform chain starts with ToTensor, which expects a PIL image
        # or ndarray.  We'll produce a numpy uint8 array so ToTensor works.
        import numpy as np
        img_np = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        from PIL import Image
        img_pil = Image.fromarray(img_np, mode="L")
        label = idx % 10
        if self.transform is not None:
            img_pil = self.transform(img_pil)
        return img_pil, label


class TestMNISTAmplitudeDataset:
    """Test the full dataset wrapper with a mocked MNIST backend."""

    @pytest.fixture()
    def dataset(self, monkeypatch):
        """Patch torchvision.datasets.MNIST and return a small MNISTAmplitude."""
        import torchvision.datasets

        def _fake_mnist_init(self_inner, root, train=True, download=True, transform=None):
            # Replace internals with our fake
            length = 60000 if train else 10000
            fake = _FakeMNIST(length=length, transform=transform)
            self_inner._fake = fake
            self_inner.data = list(range(length))  # just for len()
            self_inner.targets = [i % 10 for i in range(length)]
            self_inner.transform = transform

        def _fake_getitem(self_inner, idx):
            return self_inner._fake[idx]

        def _fake_len(self_inner):
            return len(self_inner._fake)

        monkeypatch.setattr(torchvision.datasets.MNIST, "__init__", _fake_mnist_init)
        monkeypatch.setattr(torchvision.datasets.MNIST, "__getitem__", _fake_getitem)
        monkeypatch.setattr(torchvision.datasets.MNIST, "__len__", _fake_len)

        return MNISTAmplitude(root="/tmp/fake", split="train", resize_to=160, final_size=240, download=False)

    def test_output_shape(self, dataset):
        sample = dataset[0]
        assert sample["amplitude"].shape == (1, 240, 240)
        assert sample["mask"].shape == (1, 240, 240)

    def test_amplitude_range(self, dataset):
        sample = dataset[0]
        assert sample["amplitude"].min() >= 0.0
        assert sample["amplitude"].max() <= 1.0

    def test_mask_binary(self, dataset):
        sample = dataset[0]
        unique = torch.unique(sample["mask"])
        for v in unique:
            assert v.item() in (0.0, 1.0)

    def test_label_type(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["label"], int)

    def test_train_split_length(self, dataset):
        assert len(dataset) == 50000


# ---------------------------------------------------------------------------
# Grating target tests
# ---------------------------------------------------------------------------

class TestGratingTarget:
    def test_grating_target_shape(self):
        target = generate_grating_target(10.8, dx_mm=0.3)
        assert target.shape == (1, 240, 240)

    def test_grating_target_binary(self):
        target = generate_grating_target(10.8, dx_mm=0.3)
        unique_vals = torch.unique(target)
        for v in unique_vals:
            assert v.item() in (0.0, 1.0)

    def test_grating_target_has_bars(self):
        target = generate_grating_target(10.8, dx_mm=0.3)
        # The centre 160×160 region should contain some 1s
        center = target[0, 40:200, 40:200]
        assert center.sum() > 0


# ---------------------------------------------------------------------------
# Support mask tests
# ---------------------------------------------------------------------------

class TestSupportMask:
    def test_support_mask_strategy(self):
        amp = torch.tensor([[[0.0, 0.7, 0.0], [0.3, 0.0, 0.9]]])
        mask = make_support_mask(amp, strategy="positive_pixels")
        expected = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]])
        assert torch.equal(mask, expected)
        assert mask.shape == amp.shape

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown mask strategy"):
            make_support_mask(torch.zeros(1, 2, 2), strategy="magic")
