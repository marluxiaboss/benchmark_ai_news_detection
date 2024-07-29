import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from abc import ABC, abstractmethod
from datasets import Dataset
from tqdm import tqdm
from .detector import Detector


class BertDetector(Detector):
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: str) -> None:
        """
        Detector class based on a trained BERT-type model such as RoBERTa, DistilBERT and Electra.
        
        Parameters:
            model: torch.nn.Module
                The model to use for detection
            tokenizer: PreTrainedTokenizerBase
                The tokenizer to use for tokenization
            device: str
                The device to use for the model
        """
        
        # we assume here that the model is already trained
        # maybe only provide the path and let this class handle the loading
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.eval()
        
    def detect(self, texts: list[str], batch_size: int, detection_threshold: float=0.0) -> tuple[list[int], list[float], list[int]]:
        """
        Detect the if the texts given as input are AI-generated (label 1) or human-written (label 0).
        Returns the predicted lables with argmax, the logits of the positive class and the predicted labels with the given detection threshold
        instead of the argmax.
        
        Parameters:
            texts: list[str]
                The texts to detect
            batch_size: int
                The batch size to use for detection
            detection_threshold: float
                The threshold to use for detection. Default is 0.0.
        
        Returns:
            tuple[list[int], list[float], list[int]]
                The predicted labels with argmax, the logits of the positive class and the predicted labels with the given detection threshold
                instead of the argmax.
        """
            
        # tokenized texts and create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)
        
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.remove_columns(["text"])
        dataset.set_format("torch")
        
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds = []
        preds_at_threshold = []
        logits_pos_class = []

        for i, batch in enumerate(tqdm(test_loader, desc="Detecting...")):

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            pos_class_logits = logits[:, 1]
            prob_pos_class_logits = torch.softmax(pos_class_logits, dim=-1)
            
            # predict the class with the highest probability, without applying a threshold
            preds_batch = torch.argmax(logits, dim=-1)
            
            # apply detection threshold, i.e. predict 1 if the probability of the positive class is higher than the threshold
            preds_batch_thresholds = (prob_pos_class_logits > detection_threshold).int()
                
            preds.extend(preds_batch.tolist())
            preds_at_threshold.extend(preds_batch_thresholds.tolist())
            logits_pos_class.extend(prob_pos_class_logits.tolist())
            
        return preds, logits_pos_class, preds_at_threshold
    