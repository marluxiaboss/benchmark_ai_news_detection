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


class ParaphrasingAttack(ArticleGenerator):
    
    def paraphrase(self, texts, nb_paraphrasing=1, batch_size=1) -> list:
        pass
    
class PromptParaphrasingAttack(ArticleGenerator):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, paraphraser_model,
                 paraphraser_config, paraphraser_prompt_config, max_sample_len, watermarking_scheme=None):
        
        super().__init__(gen_model, gen_config, gen_prompt_config, max_sample_len, watermarking_scheme)
                
        # Paraphraser LLM
        self.paraphraser_model = paraphraser_model
        self.paraphraser_prompt_config = paraphraser_prompt_config
        self.model_config = paraphraser_config
        
        self.attack_name = "paraphrasing_attack"
    
    def paraphrase(self, texts, nb_paraphrasing=1, batch_size=1) -> list:
        
        # Get all the parameters
        model_config = self.model_config
        tokenizer = model_config.tokenizer
        use_chat_template = model_config.use_chat_template
        template_type = model_config.chat_template_type
        system_paraphrasing_prompt = self.paraphraser_prompt_config.system_prompt
        user_paraphrasing_prompt = self.paraphraser_prompt_config.user_prompt
        
        fake_articles = texts

        # generate articles
        for i in range(nb_paraphrasing):
            
            #user_paraphrasing_prompts = [f"INPUT: {fake_text}" for fake_text in fake_articles]
        
            prefixes_with_prompt = [transform_chat_template_with_prompt(
                fake_article, user_paraphrasing_prompt, tokenizer,
                use_chat_template, template_type, system_paraphrasing_prompt, forced_prefix="OUTPUT:")
                for fake_article in fake_articles]
            
            fake_articles = []
            
            # generate the articles
            for i in range(0, len(prefixes_with_prompt), batch_size):
                samples = prefixes_with_prompt[i:i+batch_size]
                outputs = self.paraphraser_model(samples)
                fake_articles.extend(outputs)
                    
        return fake_articles
    
    def generate_adversarial_text(self, prefixes, batch_size=1):

        # generate news articles in a "normal" way
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)
        
        # paraphrase the texts
        paraphrased_fake_articles = self.paraphrase(fake_articles, batch_size=batch_size, nb_paraphrasing=1)
        
        # cut to max_sample_len
        paraphrased_fake_articles = [text[:self.max_sample_len] for text in paraphrased_fake_articles]
        
        return paraphrased_fake_articles