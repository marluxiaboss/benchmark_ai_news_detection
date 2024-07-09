import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import Dataset
from .detector import Detector
from watermark.auto_watermark import AutoWatermark

class WatermarkDetector(Detector):
    
    def __init__(self, watermarking_scheme: AutoWatermark, detection_threshold: float) -> None:
        """
        Detector class based on a watermarking scheme.
        
        Parameters:
            watermarking_scheme: AutoWatermark
                The watermarking scheme to use for detection (see https://github.com/THU-BPM/MarkLLM for the source of AutoWatermark).
                Caveat: it should be the exact same watermarking scheme that was used for generating the watermarked texts.
            detection_threshold: float
                The threshold to use for detection
        """
        
        self.watermarking_scheme = watermarking_scheme
        self.detection_threshold = detection_threshold
        
    def detect(self, texts: list[str], batch_size: int, detection_threshold: int) -> tuple[list[int], list[float], list[int]]:
        """
        Detect the if the texts given as input are watermarked (label 1) or not (label 0).
        """
        
        preds = []
        preds_at_threshold = []
        logits_pos_class = []

        for text in texts:
            res_dict = self.watermarking_scheme.detect_watermark(text)
            z_score = res_dict["score"]
            pred = int(res_dict["is_watermarked"])
            pred_at_threshold = int(z_score > detection_threshold)
            
            preds.append(pred)
            preds_at_threshold.append(pred_at_threshold)
            logits_pos_class.append(z_score)
        
        return preds, logits_pos_class, preds_at_threshold