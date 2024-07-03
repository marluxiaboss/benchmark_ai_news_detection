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
from .article_generator import ArticleGenerator

class GenParamsAttack(ArticleGenerator):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, adversarial_gen_params,
                  max_sample_len, watermarking_scheme=None):
        
        super().__init__(gen_model, gen_config, gen_prompt_config, max_sample_len, watermarking_scheme)

        self.adversarial_gen_params = adversarial_gen_params
        
        self.attack_name = "gen_parameters_attack"
    
    def generate_adversarial_text(self, prefixes, batch_size=1):
    
        # Change specific generation parameters compared to base model
        for key, value in self.adversarial_gen_params.items():
            self.gen_model_config.gen_params[key] = value

        # generate text
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)
        
        # cut to max_sample_len
        fake_articles = [text[:self.max_sample_len] for text in fake_articles]
        
        return fake_articles