
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
            decoded_outputs_before = self.tokenizer.batch_decode(
                output_ids[:, :])
            print(f"decoded_outputs_before: {decoded_outputs_before}")
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
    
    """
    def forward_step_by_step(self, samples: list, batch_size: int=1, watermarking_scheme: Optional[AutoWatermark]=None,
            compute_entropy, entropy_model) -> list[str]:
        
        outputs_list = []
        generation_config = self.gen_params
        warpers = []
        
        
        
        if generation_config.max_new_tokens is not None:
            warpers.append(MaxNewTokensLogitsWarper(generation_config.max_new_tokens))
        if generation_config.min_tokens is not None:
            warpers.append(MinNewTokensLengthLogitsProcessor(generation_config.min_tokens))
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating text"):
            
            # Generate tokens
            for i in range(self.config.sequence_length):
                with torch.no_grad():
                    if past:
                        output = self.config.generation_model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                    else:
                        output = self.config.generation_model(inputs)
                        
                scores = output.scores
                
                # apply the
                
                # Get probabilities
                probs = torch.nn.functional.softmax(output.logits[:,-1, :self.config.vocab_size], dim=-1).cpu()
                
                # Generate r1, r2,..., rk
                self.utils.seed_rng(inputs[0])
                random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
                
                # Sample token to add watermark
                token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

                # Update inputs
                inputs = torch.cat([inputs, token], dim=-1)

                # Update past
                past = output.past_key_values

                # Update attention mask
                attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

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
    
    """
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
    
    
    