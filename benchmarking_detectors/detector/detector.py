import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import Dataset

from tqdm import tqdm

from watermark.auto_watermark import AutoWatermark


class Detector(ABC):
    @abstractmethod
    def detect(self, text: str) -> tuple[list[int], list[float], list[int]]:
        pass
    

    
        
        
        
        
        
        
    