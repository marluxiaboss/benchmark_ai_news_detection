import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import Dataset
from .detector import Detector
from watermark.auto_watermark import AutoWatermark

class WatermarkDetector(Detector):
    
    def __init__(self, watermarking_scheme: AutoWatermark, detection_threshold: float):
        self.watermarking_scheme = watermarking_scheme
        self.detection_threshold = detection_threshold
        
    def detect(self, texts: list, batch_size: int, detection_threshold: int) -> list:
        
        preds = []
        preds_at_threshold = []
        logits_pos_class = []

        for text in texts:
            res_dict = self.watermarking_scheme.detect_watermark(text)
            z_score = res_dict["score"]
            pred = int(res_dict["is_watermarked"])
            pred_at_threshold = int(z_score > self.detection_threshold)
            
            preds.append(pred)
            preds_at_threshold.append(pred_at_threshold)
            logits_pos_class.append(z_score)
        
        return preds, logits_pos_class, preds_at_threshold