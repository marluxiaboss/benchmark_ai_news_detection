import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import Dataset

from tqdm import tqdm

from watermark.auto_watermark import AutoWatermark


class Detector(ABC):
    @abstractmethod
    def detect(
        self, texts: list[str], batch_size: int, detection_threshold: float = 0.0
    ) -> tuple[list[int], list[float], list[int]]:
        """
        Detect the watermark in the texts.

        Parameters:
        ----------
            texts: list[str]
                The texts to detect the watermark in
            batch_size: int
                The batch size
            detection_threshold: float
                The threshold to use for the detection

        Returns:
        -------
            tuple[list[int], list[float], list[int]]
                The predictions, the logits, and the predictions at the threshold
        """
        pass
