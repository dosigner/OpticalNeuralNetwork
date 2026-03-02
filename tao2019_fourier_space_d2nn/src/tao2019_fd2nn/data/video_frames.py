"""Video-frame saliency adapter."""

from __future__ import annotations

from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset


class VideoFramesSaliencyDataset(SaliencyPairsDataset):
    """Video frame/mask wrapper."""
