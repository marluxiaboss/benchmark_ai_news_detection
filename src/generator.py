import torch
from torch import nn
from datetime import datetime

import nltk.data
nltk.download('punkt')

from tqdm import tqdm


class LLMGenerator(nn.Module):
    def __init__(self, gpt_model, tokenizer, gen_params=None, device=None):
        super().__init__()

        # gpt should already be trained
        self.gpt = gpt_model
        self.tokenizer = tokenizer
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.gen_params = gen_params 

    def forward(self, samples, max_new_tokens=None, min_new_tokens=None, watermarking_config=None):

        max_length = self.gen_params["max_length"] if max_new_tokens is None else max_new_tokens
        self.gen_params["max_length"] = max_length
        self.gen_params["min_new_tokens"] = min_new_tokens
        encoding = self.tokenizer.batch_encode_plus(
            samples, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = encoding['input_ids'].to(self.device)

        print("self gen params: ", self.gen_params)
        # generate text using the gpt model
        with torch.no_grad():
            output_ids = self.gpt.generate(
                input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_params)

        # decode the generated text
        # decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
        decoded_outputs = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:])

        # remove special tokens from the generated text
        special_tokens = self.tokenizer.additional_special_tokens + \
            [self.tokenizer.pad_token] + [self.tokenizer.eos_token]
        for i, sample in enumerate(samples):
            decoded_output = decoded_outputs[i]
            for special_token in special_tokens:
                decoded_output = decoded_output.replace(special_token, "")
            decoded_outputs[i] = decoded_output

        return decoded_outputs

    def forward_with_distribution(self, sample):
        
        samples = [sample]
        
        # sample is only one text, this function does not support batch generation
        max_length = self.gen_params["max_length"]
        encoding = self.tokenizer.batch_encode_plus(
            samples, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = encoding['input_ids'].to(self.device)

        # generate text using the gpt model
        with torch.no_grad():
            #output_ids = self.gpt.generate(
            #    input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_params)
            output = self.gpt.generate(
                input_ids, pad_token_id=self.tokenizer.pad_token_id, **self.gen_params, return_dict_in_generate=True,
                output_logits=True)
            
        output_ids = output.sequences

        # decode the generated text
        # decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
        decoded_outputs = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:])
        
        # remove special tokens from the generated text
        special_tokens = self.tokenizer.additional_special_tokens + \
            [self.tokenizer.pad_token] + [self.tokenizer.eos_token]
        for i, sample in enumerate(samples):
            decoded_output = decoded_outputs[i]
            for special_token in special_tokens:
                decoded_output = decoded_output.replace(special_token, "")
            decoded_outputs[i] = decoded_output

        # get the distribution of the generated tokens with format list dict {token: probability}
        # i.e. the list should have the same length as the decoded_outputs
        distributions = []
        scores = output.logits
        
        new_scores = []
    
        # apply softmax to the scores
        for i in range(len(scores)):
            new_scores.append(torch.nn.functional.softmax(scores[i], dim=-1))
            
        # compute all the decoded tokens once
        token_vocab = self.tokenizer.get_vocab()
        
        # reverse the dictionary
        token_vocab = {v: k for k, v in token_vocab.items()}
        
        # sort the dictionary by key
        token_vocab = dict(sorted(token_vocab.items(), key=lambda item: item[0]))
        token_list_ordered = list(token_vocab.items())

        for i in range(len(new_scores)):
            
            distribution = {}
            
            distribution_list = new_scores[i][0].tolist()
            distribution = dict(zip(token_list_ordered, distribution_list))
            distributions.append(distribution)
        
        # sort all the distributions by probability
        top_p = self.gen_params["top_p"]
        new_distributions = []
        for i in range(len(distributions)):
            distribution = dict(sorted(distributions[i].items(), key=lambda item: item[1], reverse=True))
            
            # only keep tokens until we get probability = top_p
            top_p_sum = 0
            new_distribution = {}
            for key, value in distribution.items():
                new_distribution[key] = value
                top_p_sum += value
                if top_p_sum >= top_p:
                    break
            new_distributions.append(new_distribution)
            
            
        distributions = new_distributions

        #for i in range(50):
        #    print("token: ", list(distribution_0.keys())[i], " prob: ", list(distribution_0.values())[i])
        
        
        
        return decoded_outputs, distributions
