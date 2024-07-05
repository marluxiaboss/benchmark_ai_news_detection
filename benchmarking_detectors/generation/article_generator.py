from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os
import pandas as pd
import copy
from tqdm import tqdm

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_from_disk, concatenate_datasets, Dataset

from abc import ABC, abstractmethod

from watermark.auto_watermark import AutoWatermark
from utils.gen_utils import transform_chat_template_with_prompt

class ArticleGenerator(ABC):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, max_sample_len, watermarking_scheme=None):
        
        # Generator LLM
        self.gen_model = gen_model
        self.gen_prompt_config = gen_prompt_config
        self.gen_model_config = gen_config
        self.max_sample_len = max_sample_len
        self.watermarking_scheme = watermarking_scheme
        
        self.attack_name = ""
        self.watermarking_scheme_name = ""
        self.gen_name = gen_config.model_name
        
        
    def generate_text(self, prefixes, batch_size=1):
        
        # assumption: all attacks will generate text

        gen_model = self.gen_model

        # apply the chat template with the prompt
        system_prompt = self.gen_prompt_config.system_prompt
        user_prompt = self.gen_prompt_config.user_prompt
        gen_tokenizer = self.gen_model_config.tokenizer
        use_chat_template = self.gen_model_config.use_chat_template
        template_type = self.gen_model_config.chat_template_type
        
        # apply the chat template with the prompt
        prefixes_with_prompt = [transform_chat_template_with_prompt(
            prefix, user_prompt, gen_tokenizer,
            use_chat_template, template_type, system_prompt, forced_prefix=prefix) for prefix in prefixes]

        # generate articles
        fake_articles = []
        fake_articles = gen_model(prefixes_with_prompt, batch_size=batch_size, watermarking_scheme=self.watermarking_scheme)
            
        # add the prefix back to the generated text since generation cuts the first "input_size" tokens from the input
        # if we force the prefix in the generation, it is counted in the "input_size" tokens
        # We have to be careful though because sometimes fake articles starts with a space, sometimes not
        prefixes = [prefix.strip() for prefix in prefixes]
        fake_articles = [fake_article.strip() for fake_article in fake_articles]
        fake_articles = [f"{prefixes[i]} {fake_articles[i]}" for i in range(len(fake_articles))]
        
        # cut to max_sample_len
        fake_articles = [text[:self.max_sample_len] for text in fake_articles]
        
        return fake_articles
    
    
    def set_attack_name(self, attack_name):
        self.attack_name = attack_name
        
    def set_watermarking_scheme_name(self, watermarking_scheme_name):
        self.watermarking_scheme_name = watermarking_scheme_name
    
    @abstractmethod 
    def generate_adversarial_text(self, prefixes, batch_size=1):
        """
        This is the adversarial version of text generation. 
        All attack should generate text at some point. Either generate text in a specific way or modify the generated text.
        """
        pass
    