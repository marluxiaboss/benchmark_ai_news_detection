from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
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

from utils.gen_utils import transform_chat_template_with_prompt
from .article_generator import ArticleGenerator
from watermark.auto_watermark import AutoWatermark
from utils.configs import ModelConfig, PromptConfig
from .generator import LLMGenerator


# TODO: Should we keep this in case we use a different model for paraphrasing?
class ParaphrasingAttack(ArticleGenerator):
    def paraphrase(self, texts, nb_paraphrasing=1, batch_size=1) -> list:
        pass


class PromptParaphrasingAttack(ArticleGenerator):
    def __init__(
        self,
        gen_model: LLMGenerator,
        gen_config: ModelConfig,
        gen_prompt_config: PromptConfig,
        paraphraser_model: LLMGenerator,
        paraphraser_config: ModelConfig,
        paraphraser_prompt_config: PromptConfig,
        max_sample_len: int,
        watermarking_scheme: AutoWatermark = None,
    ) -> None:
        """
        Class for generating text using a model from Huggingface with paraphrasing based on prompting a model to paraphrase the generated text.

        Parameters:
        ----------
            gen_model: LLMGenerator
                The pretrained language model (Transformers) to be used for the initial text generation.
            gen_config: ModelConfig
                The configuration of the model.
            gen_prompt_config: PromptConfig
                The configuration of the prompt.
            paraphraser_model: LLMGenerator
                The pretrained language model (Transformers) to be used for paraphrasing.
            paraphraser_config: ModelConfig
                The configuration of the paraphraser model.
            paraphraser_prompt_config: PromptConfig
                The configuration of the prompt for the paraphraser model.
            max_sample_len: int
                The maximum length of the generated text.
            watermarking_scheme: AutoWatermark
                The optional watermarking scheme to use for the initial generation (not the paraphrasing!). Default is None.$
        """

        super().__init__(
            gen_model, gen_config, gen_prompt_config, max_sample_len, watermarking_scheme
        )

        # Paraphraser LLM
        self.paraphraser_model = paraphraser_model
        self.paraphraser_prompt_config = paraphraser_prompt_config
        self.model_config = paraphraser_config

        self.attack_name = "paraphrasing_attack"

    def paraphrase(
        self, texts: list[str], nb_paraphrasing: int = 1, batch_size: int = 1
    ) -> list[str]:
        """
        Paraphrasing function used after the initial text generation.

        Parameters:
        ----------
            texts: list
                Initial generated texts to be paraphrased.
            nb_paraphrasing: int
                Number of recursive paraphrasing to be done.
            batch_size: int
                The batch size to use for generation.

        Returns:
        -------
            list
                A list of paraphrased generated texts.
        """

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

            # user_paraphrasing_prompts = [f"INPUT: {fake_text}" for fake_text in fake_articles]

            prefixes_with_prompt = [
                transform_chat_template_with_prompt(
                    fake_article,
                    user_paraphrasing_prompt,
                    tokenizer,
                    use_chat_template,
                    template_type,
                    system_paraphrasing_prompt,
                    forced_prefix="OUTPUT:",
                )
                for fake_article in fake_articles
            ]

            fake_articles = []

            # generate text with paraphrasing propmt withou watermarking
            fake_articles = self.paraphraser_model(
                prefixes_with_prompt, batch_size=batch_size, watermarking_scheme=None
            )

        return fake_articles

    def generate_adversarial_text(self, prefixes: list[str], batch_size: int = 1) -> list[str]:
        """
        Generate text with paraphrasing.

        Parameters:
        ----------
            prefixes: list
                A list of input contexts for text generation.
            batch_size: int
                The batch size to use for generation.

        Returns:
        -------
            list
                A list of generated text.
        """

        # generate news articles in a "normal" way
        fake_articles = self.generate_text(prefixes, batch_size=batch_size)

        # paraphrase the texts
        paraphrased_fake_articles = self.paraphrase(
            fake_articles, batch_size=batch_size, nb_paraphrasing=1
        )

        # cut to max_sample_len
        paraphrased_fake_articles = [
            text[: self.max_sample_len] for text in paraphrased_fake_articles
        ]

        return paraphrased_fake_articles
