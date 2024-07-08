
from transformers import (ElectraForSequenceClassification, ElectraTokenizer,
    AutoConfig, AutoModelForCausalLM, AutoTokenizer)
import torch
from .bert_detector import BertDetector
from .fast_detect_gpt import FastDetectGPT


class DetectorLoader:
    
    def __init__(self, detector_name, device,
                 weights_checkpoint=None, local_weights=False) -> None:
        
        self.detector_name = detector_name
        self.device = device
        self.weights_checkpoint = weights_checkpoint
        self.local_weights = local_weights
        
    def load(self):
        
        detector_name = self.detector_name
        device = self.device
        
        match detector_name:
            
            case "electra_large":
                assert (self.local_weights and self.weights_checkpoint is not None), "This detector requires a weights checkpoint"
                
                detector_path = "google/electra-large-discriminator"
                config = AutoConfig.from_pretrained(detector_path)
                detector_model = ElectraForSequenceClassification(config)
                detector_tokenizer = ElectraTokenizer.from_pretrained(detector_path)
                
                model_path = self.weights_checkpoint
                detector_model.load_state_dict(torch.load(model_path))
                detector_model.to(device)
                
                detector = BertDetector(detector_model, detector_tokenizer, device)

            case "fast_detect_gpt":
                
                # TODO: add more config options for fast_detect_gpt
                
                ref_model_path = "openai-community/gpt2"
                ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype="auto").to(device)
                ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_path, trust_remote_code=True, padding_side="left")

                # special for gpt2
                ref_tokenizer.pad_token = ref_tokenizer.eos_token
                ref_tokenizer.padding_side = 'left'

                scoring_model = ref_model
                scoring_tokenizer = ref_tokenizer

                detector = FastDetectGPT(ref_model, scoring_model, ref_tokenizer, scoring_tokenizer, device)
            
            case _:
                raise ValueError(f"Detector {detector_name} not supported yet")
        
        return detector