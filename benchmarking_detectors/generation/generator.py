
from tqdm import tqdm

import torch
from torch import nn
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM,
        LogitsProcessor, LogitsProcessorList)

class LLMGenerator(nn.Module):
    def __init__(self, model, model_config):
        super().__init__()

        # gpt should already be trained
        self.generator = model
        self.tokenizer = model_config.tokenizer
        self.device = model_config.device
        self.gen_params = model_config.gen_params

    def forward(self, samples: list, batch_size: int = 1, watermarking_scheme=None):
        
        # TODO: optimize it with dataset/dataloader
        outputs_list = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating text"):
            
            batch_samples = samples[i:i+batch_size]
            encoding = self.tokenizer.batch_encode_plus(
                batch_samples, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(self.device)

            with torch.no_grad():
                if watermarking_scheme is not None:
                    output_ids = self.generator.generate(
                        input_ids, pad_token_id=self.tokenizer.pad_token_id, 
                        logits_processor=LogitsProcessorList([watermarking_scheme]), **self.gen_params)
                else:     
                    output_ids = self.generator.generate(
                        input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_params)

            # decode the generated text
            decoded_outputs = self.tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:])
                
            outputs_list.extend(decoded_outputs)
            
        # remove special tokens from the generated text
        special_tokens = self.tokenizer.additional_special_tokens + \
            [self.tokenizer.pad_token] + [self.tokenizer.eos_token]
            
        for i, sample in enumerate(samples):
            output = outputs_list[i]
            for special_token in special_tokens:
                output = output.replace(special_token, "")
            outputs_list[i] = output
        
        return outputs_list
    