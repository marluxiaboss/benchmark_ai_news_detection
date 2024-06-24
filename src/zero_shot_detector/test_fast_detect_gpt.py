# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import sys
import time
import glob
import argparse
import json
from datetime import datetime
import jsonlines

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve


from tqdm import tqdm

SRC_PATH = ["src"]
for module_path in SRC_PATH:
    if module_path not in sys.path:
        sys.path.append(module_path)
from utils import *




### MODEL FILE ###
def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


### FAST DETECT GPT FILE ###
def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args=None, ref_path=None):
        if args is None:
            ref_path = ref_path
        else:
            ref_path = args.ref_path
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)


def load_test_dataset(dataset_path, use_eval_set=False):

    dataset = load_from_disk(dataset_path)
    try:
        if use_eval_set:
            dataset_test = dataset["valid"]
        else:
            dataset_test = dataset["test"]
    except KeyError:
        dataset_test = dataset

    return dataset_test

def tokenize_dataset(tokenizer, dataset):

    if tokenizer is None:
        raise ValueError("Tokenizer not set")
    
    def tokenize_text(x, tokenizer):
        return tokenizer(x["text"], truncation=True, padding="max_length", return_tensors="pt")
    
    dataset = dataset.map(lambda x: tokenize_text(x, tokenizer), batched=True)

    return dataset

### MAIN FILE ###
def run(args):

    # create experiment folder
    base_path = "saved_training_logs_experiment_2/fast_detect_gpt"
    #experiment_path = f"{base_path}/{datetime.now().strftime('%d_%m_%H%M')}"
    experiment_path = f"{base_path}"

    dataset_name = args.dataset_path.split("/")[-1]

    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()

    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)

    dataset = load_test_dataset(args.dataset_path, args.use_eval_set)

    # maybe need to tokenize the dataset twice, once for each tokenizer
    if args.reference_model_name != args.scoring_model_name:
        pass
    else:
        tokenized_dataset = tokenize_dataset(scoring_tokenizer, dataset)

    # iterate over the dataset and do detection on each sample

    # create dataloader
    batch_size = 1

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    preds = []
    probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Performing detection on dataset..."):
            text = batch["text"]

            tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]

            for i in range(batch_size):
                crit = criterion_fn(logits_ref[i:i+1], logits_score[i:i+1], labels[i:i+1])
                prob = prob_estimator.crit_to_prob(crit)

                pred = 1 if prob > 0.5 else 0
                preds.append(pred)
                probs.append(prob)

    # calculate accuracy
    preds = np.array(preds)
    labels = np.array(dataset["label"])
    acc = np.mean(preds == labels)
    print(f'Accuracy: {acc * 100:.2f}%')
    
    # calculate roc auc score
    probs = np.array(probs)
    
    nb_pos_labels = np.sum(labels)
    nb_neg_labels = len(labels) - nb_pos_labels
        
    if nb_pos_labels == 0 or nb_neg_labels == 0:
        print("Only one class detected, cannot compute ROC AUC")
        roc_auc = 0
        fpr = np.zeros(1)
        tpr = np.zeros(1)
        thresholds = np.zeros(1)
        #roc_auc = roc_auc_score(labels, probs)
        #fpr, tpr, thresholds = roc_curve(labels, probs)
        
    else:
        roc_auc = roc_auc_score(labels, probs)
        fpr, tpr, thresholds = roc_curve(labels, probs)

    results = compute_bootstrap_metrics(preds, labels)
    
    print("Test metrics:")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f'ROC AUC: {roc_auc * 100:.2f}%')
    print(f"fpr: {fpr}")
    print(f"tpr: {tpr}")
    print(f"thresholds: {thresholds}")
    
    results["roc_auc"] = roc_auc
    results["fpr_at_thresholds"] = fpr.tolist()
    results["tpr_at_thresholds"] = tpr.tolist()
    results["thresholds"] = thresholds.tolist()
    
    if args.classifier_threshold is not None:
        preds_at_threshold = np.array(probs > args.classifier_threshold, dtype=int)
        results_at_threshold = compute_bootstrap_metrics(preds_at_threshold, labels)
        print(f"Test metrics at threshold {args.classifier_threshold}:")
        
        for key, value in results_at_threshold.items():
            print(f"{key}: {value}")
        
        results["given_threshold"] = args.classifier_threshold
        for key, value in results_at_threshold.items():
            results[f"{key}_at_given_threshold"] = value

        # define where to save the results
        if args.use_eval_set:
            
            if not os.path.isdir(f"{experiment_path}/eval"):
                os.makedirs(f"{experiment_path}/eval")
            
            json_res_file_path = f"{experiment_path}/eval/eval_metrics_{dataset_name}.json"
            
        else:
            if args.classifier_threshold is not None:
                if not os.path.isdir(f"{experiment_path}/test_at_threshold"):
                    os.makedirs(f"{experiment_path}/test_at_threshold")
                    
                json_res_file_path = f"{experiment_path}/test_at_threshold/test_metrics_{dataset_name}.json"
                
            else:
                if not os.path.isdir(f"{experiment_path}/test"):
                    os.makedirs(f"{experiment_path}/test")
            
                json_res_file_path = f"{experiment_path}/test/test_metrics_{dataset_name}.json"
                
        with open(json_res_file_path, "w") as f:
            f.write(json.dumps(results, indent=4))

    # results for random prediction
    random_preds = np.random.randint(0, 2, len(labels))
    random_acc = np.mean(random_preds == labels)
    print(f'Random prediction accuracy: {random_acc * 100:.2f}%')


def predict_on_dataset(dataset):
    reference_model_name = "gpt-neo-2.7B"
    scoring_model_name = "gpt-neo-2.7B"
    dataset_path = "xsum"
    #dataset = "xsum"
    ref_path = "src/zero_shot_detector/local_infer_ref"
    device = "cuda"
    cache_dir = "../cache"
    
    # create experiment folder

    # load model
    scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
    scoring_model = load_model(scoring_model_name,device,cache_dir)
    scoring_model.eval()
    if reference_model_name != scoring_model_name:
        reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)
        reference_model = load_model(reference_model_name, device, cache_dir)
        reference_model.eval()

    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(ref_path=ref_path)

    #dataset = load_test_dataset(args.dataset_path)

    # maybe need to tokenize the dataset twice, once for each tokenizer
    if reference_model_name != scoring_model_name:
        pass
    else:
        tokenized_dataset = tokenize_dataset(scoring_tokenizer, dataset)

    # iterate over the dataset and do detection on each sample

    # create dataloader
    batch_size = 1
    
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    preds = []
    probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Performing detection on dataset..."):
            text = batch["text"]

            tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            labels = tokenized.input_ids[:, 1:]
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if reference_model_name == scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]

            for i in range(batch_size):
                crit = criterion_fn(logits_ref[i:i+1], logits_score[i:i+1], labels[i:i+1])
                prob = prob_estimator.crit_to_prob(crit)

                pred = 1 if prob > 0.5 else 0
                
                probs.append(prob)
                preds.append(pred)

    # calculate accuracy
    preds = np.array(preds)
    labels = np.array(dataset["label"])
    
    # calculate roc auc score
    probs = np.array(probs)
    
    return preds, labels
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset_path', type=str, default="xsum")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="src/zero_shot_detector/local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument("--classifier_threshold", type=float, default=None, help="Threshold for the classifier")
    parser.add_argument("--use_eval_set", action=argparse.BooleanOptionalAction, help="Whether to use the eval set for training", default=False)
    args = parser.parse_args()

    run(args)



