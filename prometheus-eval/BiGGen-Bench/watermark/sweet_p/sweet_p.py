# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================
# sweet.py
# Description: Implementation of SWEET algorithm
# ==============================================
import torch
import numpy as np
from math import sqrt

from functools import partial
from ..base import BaseWatermark
from transformers import LogitsProcessor, LogitsProcessorList



class SWEET_PConfig:
    """Config class for SWEET algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: dict, gen_model, model_config: ModelConfig, *args, **kwargs) -> None:
        """
            Initialize the SWEET configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        
        config_dict = algorithm_config
        
        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        self.entropy_threshold = config_dict['entropy_threshold']
        self.cut_off_method = config_dict['cut_off_method']
        self.prob_ratio = config_dict['prob_ratio']
        self.top_p = config_dict['top_p']
        
        self.generation_model = gen_model
        self.generation_tokenizer = model_config.tokenizer
        self.vocab_size = self.generation_tokenizer.vocab_size
        self.device = model_config.device
        self.gen_kwargs = model_config.gen_params


class SWEET_PUtils:
    """Utility class for SWEET algorithm, contains helper functions."""

    def __init__(self, config: SWEET_PConfig, *args, **kwargs):
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last prefix_length tokens of the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size] 
        return greenlist_ids
    
    def calculate_entropy(self, model, tokenized_text: torch.Tensor):
        """Calculate entropy for each token in the tokenized_text."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
            entropy = entropy[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]

    def _compute_z_score(self, observed_count: int, T: int) -> float: 
        """Compute z-score for the observed count of green tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z

    def score_sequence(self, input_ids: torch.Tensor, entropy_list: list[float]) -> tuple[float, list[int], list[int]]:
        """Score the input_ids based on the greenlist and entropy."""
        num_tokens_scored = (len(input_ids) - self.config.prefix_length - 
                             len([e for e in entropy_list[self.config.prefix_length:] if e <= self.config.entropy_threshold]))
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                )
            )

        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            if entropy_list[idx] > self.config.entropy_threshold:
                weights.append(1)
            else:
                weights.append(0)

        # calculate number of green tokens where weight is 1
        green_token_count = sum([1 for i in range(len(green_token_flags)) if green_token_flags[i] == 1 and weights[i] == 1])
        print(f"Green token count: {green_token_count}")
        print(f"Num tokens scored: {num_tokens_scored}")
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags, weights


class SWEET_PLogitsProcessor(LogitsProcessor):
    """Logits processor for SWEET algorithm, contains the logic to bias the logits."""

    def __init__(self, config: SWEET_PConfig, utils: SWEET_PUtils, *args, **kwargs) -> None:
        """
            Initialize the SWEET logits processor.

            Parameters:
                config (SWEETConfig): Configuration for the SWEET algorithm.
                utils (SWEETUtils): Utility class for the SWEET algorithm.
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
    
    def __call__old(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        
        cut_off_method = self.config.cut_off_method
        prob_ratio = self.config.prob_ratio
        top_p = self.config.top_p
        
        # need to cast to float, otherwise can throw an error
        print(f"Entropy threshold: {self.config.entropy_threshold}")
        entropy_threshold = self.config.entropy_threshold
        gamma = self.config.gamma
        bias = self.config.delta
        
        # compute the prob of tokens and find highest prob tokens
        def softmax(x):
            f = np.exp(x - np.max(x))  # shift values
            return f / f.sum(axis=0)
        
        
        original_scores = scores
        softmaxed_logits = torch.softmax(scores, dim=-1)
        scores_array = scores.cpu().numpy()
        scores_array = scores_array.reshape(-1)
        softmaxed_logits = softmaxed_logits.cpu().numpy()
        # check if there are nan values and replace them with 0
        #softmaxed_logits = np.nan_to_num(softmaxed_logits)
        filtered_out_logits = softmaxed_logits[softmaxed_logits > 0]
        #print(f"Filtered out logits: {filtered_out_logits}")
        entropy = -np.sum(filtered_out_logits * np.log(filtered_out_logits))
        #print(f"Entropy: {entropy}")
        #print(f"Entropy: {entropy}")
        #print(f"Max entropy: {np.log(len(softmaxed_logits))}")
        
        # bias the logits only if the entropy is above the threshold
        # issue: depends on the vocabulary size!
        if entropy > entropy_threshold:
            
            # compute the prob of tokens and find highest prob tokens
            prob_of_tokens = softmaxed_logits[0]
            highest_prob = np.max(prob_of_tokens)
            
            if cut_off_method == "ratio":
                
                # filter out tokens with prob smaller than prob_ratio * highest_prob
                filtered_logits_indices = list(np.where(prob_of_tokens > prob_ratio * highest_prob)[0])
            
            elif cut_off_method == "top_p":
                dict_probs = {i: prob_of_tokens[i] for i in range(len(prob_of_tokens))}
                sorted_probs = sorted(dict_probs.items(), key=lambda x: x[1], reverse=True)
                cum_prob = 0
                filtered_logits_indices = []
                for i, (index, prob) in enumerate(sorted_probs):
                    cum_prob += prob
                    if cum_prob < top_p:
                        filtered_logits_indices.append(index)
                    else:
                        break
                        
            else:
                raise ValueError("Cut off method not recognized")
            

            
            # FIX: boost only green tokens!
            #green_tokens = get_list_of_green_tokens()
        	#updated_probs = [1 / unif if in green tokens else 0 for token in tokens]
            
            #green_tokens = ...
            #updated_probs = ...
            batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

            for b_idx in range(input_ids.shape[0]):
                greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
                batched_greenlist_ids[b_idx] = greenlist_ids

            green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

            # issue with batch here
            green_tokens_mask = green_tokens_mask[0].cpu().numpy()
            
            #print("filtered_logits_indices before: ", filtered_logits_indices)
            # filter out non-green tokens from filtered_logit_indices
            filtered_logits_indices = [idx for idx in filtered_logits_indices if green_tokens_mask[idx] == True ]
            #print("filtered_logits_indices after: ", filtered_logits_indices)
            
            if len(filtered_logits_indices) != 0:
                print("Green token boosted!")
            
                # choose nb_of_tokens_to_bias tokens randomly among the filtered out tokens but indices should be in the original logits
                # i.e. probability of token should be 0 if it is not in the filtered out tokens
                #uniform_prob = 1 / len(filtered_logits_indices)
                #print(f"Uniform prob: {uniform_prob}")
                #probs  = [uniform_prob if i in filtered_logits_indices else 0 for i in range(len(logits))]
                
                
                #probs = np.array([uniform_prob if i in filtered_logits_indices else 0 for i in range(len(scores_array))])
                
                # apply bias to gamma fraction of the logits randomly
                #nb_of_tokens_to_bias = int(gamma * len(filtered_logits_indices))
                
                #sum_probs = np.sum(probs)
                #print(f"Sum probs: {sum_probs}")
                #print(f"Uniform prob: {uniform_prob}")
                #print(f"Len scores: {len(scores_array)}")
                #print(f"Len filtered logits: {len(filtered_logits_indices)}")
                #print(f"Filtered logits: {filtered_logits_indices}")
                    
                #indices = np.random.choice(range(len(scores_array)), nb_of_tokens_to_bias, replace=False, p=probs)
                #print(f"Indices: {indices}")
                indices = filtered_logits_indices
                
                mask = np.zeros_like(scores_array)
                mask[indices] = 1
                
                # 1 for positions in filtered_logits_indices, 0 otherwise
                boosted_tokens_mask = [True if i in filtered_logits_indices else False for i in range(len(scores_array))]
                
                green_tokens_orig_scores = scores_array[boosted_tokens_mask]
                
                #print(f"Green tokens orig scores: {green_tokens_orig_scores}")
                #print(f"Orig probs: {softmaxed_logits[0][boosted_tokens_mask]}")
                
                scores = scores_array + mask * bias
                green_tokens_updated_scores = scores[boosted_tokens_mask]
                
                #print(f"Green tokens updated scores: {green_tokens_updated_scores}")
                #print(f"Updated probs: {softmax(scores)[boosted_tokens_mask]}")
                
            else:
                print("Warning: no tokens boosted! due to not green tokens in the filtered logits")
        else:
            print("Warning: no tokens boosted! due to entropy below threshold")
        scores = torch.tensor(scores).to(self.config.device)
        scores = scores.view(original_scores.shape)
        return scores


    def find_probability_mask(self, raw_probs: torch.FloatTensor) -> torch.BoolTensor:
        
        cut_off_method = self.config.cut_off_method
        prob_ratio = self.config.prob_ratio
        top_p = self.config.top_p
        
        # raw_probs of shape (batch_size, vocab_size)
        highest_prob = torch.max(raw_probs, dim=-1)[0]
        
        if cut_off_method == "ratio":
            
            # filter out tokens with prob smaller than prob_ratio * highest_prob
            #filtered_logits_indices = list(np.where(prob_of_tokens > prob_ratio * highest_prob)[0])
            prob_threshold = torch.tensor(prob_ratio * highest_prob).to(self.config.device).reshape(-1, 1)
            prob_mask = torch.where(raw_probs > prob_threshold, True, False)

        else:
            raise ValueError("Cut off method not recognized")
        
        return prob_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # get entropy
        raw_probs = torch.softmax(scores, dim=-1)  
        ent = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy_mask = (ent > self.config.entropy_threshold).view(-1, 1)
        probability_mask = self.find_probability_mask(raw_probs)
        
        green_tokens_mask = green_tokens_mask * entropy_mask * probability_mask
        
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores
    

class SWEET_P(BaseWatermark):
    """Top-level class for SWEET algorithm."""

    def __init__(self, algorithm_config: dict, gen_model, transformers_config: ModelConfig, *args, **kwargs) -> None:
        """
            Initialize the SWEET algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = SWEET_PConfig(algorithm_config, gen_model, transformers_config)
        self.utils = SWEET_PUtils(self.config)
        self.logits_processor = SWEET_PLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
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

        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # compute z_score
        z_score, _, _ = self.utils.score_sequence(encoded_text, entropy_list)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
