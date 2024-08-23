
from tqdm import tqdm

import torch
from torch import nn
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM,
        LogitsProcessor, LogitsProcessorList)
from utils.configs import ModelConfig
from watermark.auto_watermark import AutoWatermark
from typing import Optional

class LLMGenerator(nn.Module):
    def __init__(self, model: AutoModelForCausalLM, model_config: ModelConfig) -> None:
        """
        Class for generating text using a model from Huggingface.
        
        Parameters:
            model: AutoModelForCausalLM
                The pretrained language model (Transformers) to be used for text generation.
            model_config: ModelConfig
                The configuration of the model.
        """
        
        super().__init__()

        # gpt should already be trained
        self.generator = model
        self.tokenizer = model_config.tokenizer
        self.device = model_config.device
        self.gen_params = model_config.gen_params

    def forward(self, samples: list, batch_size: int=1, watermarking_scheme: Optional[AutoWatermark]=None) -> list[str]:
        """
        Takes a list of input contexts and generates text using the model.
        
        Parameters:
            samples: list
                A list of input contexts for text generation.
            batch_size: int
                The batch size to use for generation.
            watermarking_scheme: LogitsProcessor
                The watermarking scheme to use for generation.
        """
        
        outputs_list = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating text"):
            
            batch_samples = samples[i:i+batch_size]
            encoding = self.tokenizer.batch_encode_plus(
                batch_samples, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(self.device)

            with torch.no_grad():
                if watermarking_scheme is not None:
                    
                    # if the watermarking scheme has a logits processor, use it
                    if hasattr(watermarking_scheme, "logits_processor"):
                        output_ids = self.generator.generate(
                            input_ids, pad_token_id=self.tokenizer.pad_token_id, 
                            logits_processor=LogitsProcessorList([watermarking_scheme.logits_processor]), **self.gen_params
                        )
                        
                    # otherwise, use the generate method from the watermarking scheme
                    else:
                        output_ids = watermarking_scheme.generate(
                            input_ids
                        )
                        
                else:     
                    output_ids = self.generator.generate(
                        input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_params
                    )

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
                output = output.strip()
            outputs_list[i] = output
        
        return outputs_list
    
    
    def forward_step_by_step(self, samples: list, batch_size: int=1, watermarking_scheme: Optional[AutoWatermark]=None) -> list[str]:
        """
        Takes a list of input contexts and generates text using the model.
        
        Parameters:
            samples: list
                A list of input contexts for text generation.
            batch_size: int
                The batch size to use for generation.
            watermarking_scheme: LogitsProcessor
                The watermarking scheme to use for generation.
        """
        
        outputs_list = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating text"):
            
            batch_samples = samples[i:i+batch_size]
            encoding = self.tokenizer.batch_encode_plus(
                batch_samples, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(self.device)

            with torch.no_grad():
                if watermarking_scheme is not None:
                    
                    # if the watermarking scheme has a logits processor, use it
                    if hasattr(watermarking_scheme, "logits_processor"):
                        output_ids = self.generator.generate(
                            input_ids, pad_token_id=self.tokenizer.pad_token_id, 
                            logits_processor=LogitsProcessorList([watermarking_scheme.logits_processor]), **self.gen_params
                        )
                        
                    # otherwise, use the generate method from the watermarking scheme
                    else:
                        output_ids = watermarking_scheme.generate(
                            input_ids
                        )
                        
                else:     
                    output_ids = self.generator.generate(
                        input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_params
                    )

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
                output = output.strip()
            outputs_list[i] = output
        
        return outputs_list
    
    
    def forward_debug(self, samples: list, batch_size: int=1, watermarking_scheme: Optional[AutoWatermark]=None) -> list[str]:
        """
        Takes a list of input contexts and generates text using the model.
        
        Parameters:
            samples: list
                A list of input contexts for text generation.
            batch_size: int
                The batch size to use for generation.
            watermarking_scheme: LogitsProcessor
                The watermarking scheme to use for generation.
        """
        
        outputs_list = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating text"):
            
            batch_samples = samples[i:i+batch_size]
            encoding = self.tokenizer.batch_encode_plus(
                batch_samples, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(self.device)

            with torch.no_grad():
                if watermarking_scheme is not None:
                    
                    # if the watermarking scheme has a logits processor, use it
                    if hasattr(watermarking_scheme, "logits_processor"):
                        output = self.generator.generate(
                            input_ids, pad_token_id=self.tokenizer.pad_token_id, 
                            logits_processor=LogitsProcessorList([watermarking_scheme.logits_processor]),
                             return_dict_in_generate=True, output_scores=True, output_logits=True, **self.gen_params
                        )
                        
                    # otherwise, use the generate method from the watermarking scheme
                    else:
                        output = watermarking_scheme.generate(
                            input_ids
                        )
                        
                else:     
                    output = self.generator.generate(
                        input_ids, pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True, output_logits=True,
                        output_scores=True, **self.gen_params
                    )
            
            output_ids = output.sequences
            raw_logits = output.logits
            processed_logits = output.scores
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
                output = output.strip()
            outputs_list[i] = output
        
        return outputs_list, raw_logits, processed_logits
    
    
    