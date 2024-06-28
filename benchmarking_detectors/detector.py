import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import Dataset

class Detector(ABC):
    @abstractmethod
    def detect(self, text: str) -> bool:
        pass
    
class BertDetector(Detector):
    def __init__(self, model, tokenizer, device):
        
        # we assume here that the model is already trained
        # maybe only provide the path and let this class handle the loading
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.eval()
        
    def detect(self, texts: list, detection_threshold, batch_size) -> list:
                
        # tokenized texts and create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], return_tensors="pt")
        
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.remove_columns(["text"])
        dataset.set_format("torch")

        
        test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        preds = []
        logits_pos_class = []

        for batch in test_loader:

            input_ids = batch["input_ids"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
            logits = outputs.logits
            
            pos_class_logits = logits[:, 1]
            if detection_threshold is None:
                preds_batch = torch.argmax(logits, dim=-1)
            else:
    
                # apply detection threshold, i.e. predict 1 if the probability of the positive class is higher than the threshold
                preds_batch = (pos_class_logits > detection_threshold).int()
                
            preds.extend(preds_batch.tolist())
            logits_pos_class.extend(pos_class_logits.tolist())
            
            return preds, logits_pos_class
    