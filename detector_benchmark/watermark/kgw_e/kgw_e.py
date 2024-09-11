# ============================================
# kgw.py
# Description: Implementation of KGW algorithm
# ============================================

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from math import sqrt
from functools import partial
from base import BaseWatermark
from utils.configs import ModelConfig
from utils.utils import create_directory_for_file, load_config_file

from transformers import LogitsProcessor, LogitsProcessorList


class KGW_EConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: dict, gen_model, model_config: ModelConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW configuration.

            Parameters:
                algorithm_config (dict): Configuration for the KGW algorithm.	
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        config_dict = algorithm_config

        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        self.nb_docs = config_dict['nb_docs']
        self.embedding_batch_size = config_dict['embedding_batch_size']
        
        self.generation_model = gen_model
        self.generation_tokenizer = model_config.tokenizer
        self.vocab_size = self.generation_tokenizer.vocab_size
        self.device = model_config.device
        self.gen_kwargs = model_config.gen_params


class KGW_EUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: KGW_EConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.embedding_model = self.init_embedding_model()
        self.embeddings_corpus = self.create_embedding_corpus(self.config.nb_docs)
        
    
    
    def init_embedding_model(self) -> SentenceTransformer:
        """Initialize the SentenceTransformer model."""
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        return model
    
    def create_embedding_corpus(self, nb_docs: int = 10000) -> list:
        dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
        
        # take the nb_docs documents from the dataset randomly
        dataset = dataset.shuffle(seed=42)
        corpus = dataset.take(nb_docs)
        list_corpus = list(corpus)
        list_corpus = [x['text'] for x in list_corpus]
        
        # create embeddings for the corpus
        # if too slow, we can use a larger batch size
        print("Creating embeddings for the corpus...")
        list_corpus_embeddings = self.embedding_model.encode(list_corpus, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=self.config.embedding_batch_size)
        return list_corpus_embeddings
        
        
    def get_seed(self, input_ids: torch.LongTensor) -> int:
        

        # create a seed using self.config.prefix_length tokens (usually 100 or all the previous tokens if less than 100)
        #if len(input_ids) < self.config.prefix_length:
        #    context_tokens = input_ids
        #else:
        #    context_tokens = input_ids[-self.config.prefix_length:]
        context_tokens = input_ids
        
        decoded_context = self.config.generation_tokenizer.decode(context_tokens, skip_special_tokens=True)
        
        decoded_context_split = decoded_context.split("\n")
        
        # remove the system, user tokens by taking everything after the assistant token
        if "assistant" in decoded_context_split:
            decoded_context = " ".join(decoded_context_split[decoded_context_split.index("assistant")+1:]).strip()
        elif "<|assistant|>" in decoded_context_split:
            decoded_context = " ".join(decoded_context_split[decoded_context_split.index("<|assistant|>")+1:]).strip()
        else:
            decoded_context = decoded_context
            
        #print("decoded_context: ", decoded_context)
        test_sentence_embedding = self.embedding_model.encode(decoded_context, normalize_embeddings=True, show_progress_bar=False, device="cuda")
        cosine_scores = cosine_similarity([test_sentence_embedding], self.embeddings_corpus)
        most_similar_idx = np.argmax(cosine_scores) 
        seed = most_similar_idx % self.config.vocab_size
        #print("seed: ", seed)
        return seed
                
    
    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the RNG with the last min_prefix_len tokens of the input_ids."""
        #time_result = 1
        #for i in range(0, self.config.prefix_length):
        #    time_result *= input_ids[-1 - i].item()
        #prev_token = time_result % self.config.vocab_size
        seed = self.get_seed(input_ids)
        self.rng.manual_seed(int(self.config.hash_key * seed))
        return
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids
    
    def _compute_z_score(self, observed_count: int , T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z
    
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            num_tokens_scored = len(input_ids)
        #    raise ValueError(
        #        (
        #            f"Must have at least {1} token to score after "
        #            f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
        #        )
        #    )

        green_token_count = 0
        #green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        green_token_flags = []

        #for idx in range(self.config.prefix_length, len(input_ids)):
        for idx in range(len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGW_ELogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: KGW_EConfig, utils: KGW_EUtils, *args, **kwargs) -> None:
        """
            Initialize the KGW logits processor.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
                utils (KGWUtils): Utility class for the KGW algorithm.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        #if input_ids.shape[-1] < self.config.prefix_length:
        #    return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores
    

class KGW_E(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: dict, gen_model, transformers_config: ModelConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (dict): Configuration for the KGW algorithm.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = KGW_EConfig(algorithm_config, gen_model, transformers_config)
        self.utils = KGW_EUtils(self.config)
        self.logits_processor = KGW_ELogitsProcessor(self.config, self.utils)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def generate(self, encoded_prompts: list, *args, **kwargs) -> str:
        """Generate watermarked text. Takes a list of encoded prompts as input, like transformers model.generate."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompts)
        
        # Decode
        #watermarked_texts = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)
        watermarked_tokens = encoded_watermarked_text
        
        return watermarked_tokens
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)