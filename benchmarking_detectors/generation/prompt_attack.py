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
from utils import transform_chat_template_with_prompt
from .article_generator import ArticleGenerator


class PromptAttack(ArticleGenerator):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, adversarial_prompt_config,
                max_sample_len, watermarking_scheme=None):
        
        super().__init__(gen_model, gen_config, gen_prompt_config, max_sample_len, watermarking_scheme)

        # Set adversarial prompts
        self.adversarial_prompt_config = adversarial_prompt_config
        
        self.attack_name = "prompt_attack"
    
    def generate_adversarial_text(self, prefixes, batch_size=1):
        
        # Create adversarial prompt configuration
        self.gen_prompt_config = self.adversarial_prompt_config
        
        # generate text
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)
        
        # cut to max_sample_len
        fake_articles = [text[:self.max_sample_len] for text in fake_articles]
        
        return fake_articles