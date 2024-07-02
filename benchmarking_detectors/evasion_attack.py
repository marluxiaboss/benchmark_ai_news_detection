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
import nltk.data
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_from_disk, concatenate_datasets, Dataset

from abc import ABC, abstractmethod

from watermark.auto_watermark import AutoWatermark
from utils import transform_chat_template_with_prompt

class ArticleGenerator(ABC):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, watermarking_scheme=None):
        
        # Generator LLM
        self.gen_model = gen_model
        self.gen_prompt_config = gen_prompt_config
        self.gen_model_config = gen_config
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
        fake_articles = [f"{prefixes[i]} {fake_articles[i]}" for i in range(len(fake_articles))]
        
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
    
class ParaphrasingAttack(ArticleGenerator):
    
    def paraphrase(self, texts, nb_paraphrasing=1, batch_size=1) -> list:
        pass
    
class PromptParaphrasingAttack(ArticleGenerator):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, paraphraser_model,
                 paraphraser_config, paraphraser_prompt_config, watermarking_scheme=None):
        
        super().__init__(gen_model, gen_config, gen_prompt_config, watermarking_scheme)
                
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
        
        return paraphrased_fake_articles
    
class PromptAttack(ArticleGenerator):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, adversarial_prompt_config,
                watermarking_scheme=None):
        
        super().__init__(gen_model, gen_config, gen_prompt_config, watermarking_scheme)

        # Set adversarial prompts
        self.adversarial_prompt_config = adversarial_prompt_config
        
        self.attack_name = "prompt_attack"
    
    def generate_adversarial_text(self, prefixes, batch_size=1):
        
        # Create adversarial prompt configuration
        self.gen_prompt_config = self.adversarial_prompt_config
        
        # generate text
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)
        
        return fake_articles
    
class GenParamsAttack(ArticleGenerator):
    
    def __init__(self, gen_model, gen_config, gen_prompt_config, adversarial_gen_params,
                  watermarking_scheme=None):
        
        super().__init__(gen_model, gen_config, gen_prompt_config, watermarking_scheme)

        self.adversarial_gen_params = adversarial_gen_params
        
        self.attack_name = "gen_parameters_attack"
    
    def generate_adversarial_text(self, prefixes, batch_size=1):
    
        # Change specific generation parameters compared to base model
        for key, value in self.adversarial_gen_params.items():
            self.gen_model_config.gen_params[key] = value

        # generate text
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)
        
        return fake_articles

        