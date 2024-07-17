# ============================================
# exp.py
# Description: Implementation of EXP algorithm
# ============================================

import torch
from math import log
from functools import partial
from ..base import BaseWatermark
from utils.configs import ModelConfig
from utils.utils import create_directory_for_file, load_config_file

from transformers import LogitsProcessor, LogitsProcessorList


class EXPConfig:
    """Config class for EXP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: dict, gen_model, model_config: ModelConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """

        self.prefix_length = algorithm_config['prefix_length']
        self.hash_key = algorithm_config['hash_key']
        self.threshold = algorithm_config['threshold']
        self.sequence_length = algorithm_config['sequence_length']

        self.generation_model = gen_model
        self.generation_tokenizer = model_config.tokenizer
        self.vocab_size = self.generation_tokenizer.vocab_size
        self.device = model_config.device
        self.gen_kwargs = model_config.gen_params


class EXPUtils:
    """Utility class for EXP algorithm, contains helper functions."""

    def __init__(self, config: EXPConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP utility class.

            Parameters:
                config (EXPConfig): Configuration for the EXP algorithm.
        """
        self.config = config
        self.rng = torch.Generator()

    def seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last `prefix_length` tokens of the input."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
    
    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample a token from the vocabulary using the exponential sampling method."""
        return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
    
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value/(value + 1)
    

class EXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: dict, gen_model, transformers_config: ModelConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = EXPConfig(algorithm_config, gen_model, transformers_config)
        self.utils = EXPUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the EXP algorithm."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        
        # Initialize
        inputs = encoded_prompt
        attn = torch.ones_like(encoded_prompt)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)
            
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
            
        
        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        return watermarked_text    
    
    
    def generate_watermarked_text_modified_old(self, prompts: list[str], *args, **kwargs) -> str:
        """Generate watermarked text using the EXP algorithm."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompts, return_tensors="pt",
                                                    add_special_tokens=True, padding=True, truncation=True).to(self.config.device)
        
        # Initialize
        inputs = encoded_prompt["input_ids"]
        attn = torch.ones_like(inputs)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)

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
        
        watermarked_tokens = inputs.detach().cpu()
        watermarked_text = self.config.generation_tokenizer.batch_decode(watermarked_tokens, skip_special_tokens=True)

        return watermarked_text    


    def generate(self, input_ids: list, *args, **kwargs) -> str:
        """Generate watermarked text using the EXP algorithm."""

        # Initialize
        inputs = input_ids
        attn = torch.ones_like(inputs)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)

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
        
        watermarked_tokens = inputs.detach().cpu()

        return watermarked_tokens   
     

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Calculate the number of tokens to score, excluding the prefix
        num_scored = len(encoded_text) - self.config.prefix_length
        total_score = 0

        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed RNG with the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)

            # Calculate score for the current token
            r = random_numbers[encoded_text[i]]
            total_score += log(1 / (1 - r))

        # Compute the average score across all scored tokens
        score = total_score / num_scored if num_scored > 0 else 0

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = score > self.config.threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": score}
        else:
            return (is_watermarked, score)
        
        
    