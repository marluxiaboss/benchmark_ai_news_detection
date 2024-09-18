from abc import ABC, abstractmethod
from dataclasses import dataclass

import argparse
import os
import pandas as pd
import copy
from tqdm import tqdm

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

from ..watermark.auto_watermark import AutoWatermark
from ..utils.gen_utils import transform_chat_template_with_prompt
from .article_generator import ArticleGenerator
from ..utils.configs import ModelConfig, PromptConfig
from .generator import LLMGenerator


class GenParamsAttack(ArticleGenerator):
    def __init__(
        self,
        gen_model: LLMGenerator,
        gen_config: ModelConfig,
        gen_prompt_config: PromptConfig,
        adversarial_gen_params: dict,
        max_sample_len: int,
        watermarking_scheme: AutoWatermark = None,
    ) -> None:
        """
        Class for generating text using a model from Huggingface with adversarial generation parameters.
        This class can also be used to generate text with specific generation parameters in an non-adversarial way.
        Here, we add the adversarial_gen_params parameter to make it explicit that we are changing the generation parameters.

        Parameters:
        ----------
            gen_model: LLMGenerator
                The pretrained language model (Transformers) to be used for text generation.
            gen_config: ModelConfig
                The configuration of the model.
            gen_prompt_config: PromptConfig
                The configuration of the prompt.
            adversarial_gen_params: dict
                The adversarial generation parameters to use for generation.
            max_sample_len: int
                The maximum length of the generated text.
            watermarking_scheme: AutoWatermark
                The optional watermarking scheme to use for generation. Default is None.
        """

        super().__init__(
            gen_model, gen_config, gen_prompt_config, max_sample_len, watermarking_scheme
        )

        self.adversarial_gen_params = adversarial_gen_params

        self.attack_name = "gen_parameters_attack"

    def generate_adversarial_text(self, prefixes: list[str], batch_size: int = 1) -> list[str]:
        """
        Generate text with adversarial generation parameters.

        Parameters:
        ----------
            prefixes: list
                A list of input contexts for text generation.
            batch_size: int
                The batch size to use for generation.

        Returns:
        -------
            fake_articles: list
                A list of generated text.
        """

        # Change specific generation parameters compared to base model
        for key, value in self.adversarial_gen_params.items():
            self.gen_model_config.gen_params[key] = value

        # generate text
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)

        # cut to max_sample_len
        fake_articles = [text[: self.max_sample_len] for text in fake_articles]

        return fake_articles
