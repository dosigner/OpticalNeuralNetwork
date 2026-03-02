"""Data adapters for classification and saliency tasks."""

from tao2019_fd2nn.data.cifar10 import Cifar10SaliencyDataset
from tao2019_fd2nn.data.cell_gdc import CellGdcSaliencyDataset
from tao2019_fd2nn.data.davis import DavisSaliencyDataset
from tao2019_fd2nn.data.ecssd import EcssdSaliencyDataset
from tao2019_fd2nn.data.mnist import MnistAmplitudeDataset
from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset
from tao2019_fd2nn.data.saliency_gt import SaliencyGtBuilder
from tao2019_fd2nn.data.video_frames import VideoFramesSaliencyDataset

__all__ = [
    "MnistAmplitudeDataset",
    "Cifar10SaliencyDataset",
    "SaliencyPairsDataset",
    "CellGdcSaliencyDataset",
    "DavisSaliencyDataset",
    "EcssdSaliencyDataset",
    "VideoFramesSaliencyDataset",
    "SaliencyGtBuilder",
]
