"""Data adapters for classification and saliency tasks."""

from tao2019_fd2nn.data.cell_gdc import CellGdcSaliencyDataset
from tao2019_fd2nn.data.davis import DavisSaliencyDataset
from tao2019_fd2nn.data.ecssd import EcssdSaliencyDataset
from tao2019_fd2nn.data.mnist import MnistAmplitudeDataset
from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset
from tao2019_fd2nn.data.video_frames import VideoFramesSaliencyDataset

__all__ = [
    "MnistAmplitudeDataset",
    "SaliencyPairsDataset",
    "CellGdcSaliencyDataset",
    "DavisSaliencyDataset",
    "EcssdSaliencyDataset",
    "VideoFramesSaliencyDataset",
]
