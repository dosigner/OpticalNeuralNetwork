"""Cell pathology saliency adapter."""

from __future__ import annotations

from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset


class CellGdcSaliencyDataset(SaliencyPairsDataset):
    """Cell pathology image/mask wrapper."""
