"""ECSSD saliency adapter."""

from __future__ import annotations

from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset


class EcssdSaliencyDataset(SaliencyPairsDataset):
    """ECSSD wrapper (expects image_dir/mask_dir structure)."""
