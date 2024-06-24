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
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve

# for caching
from diskcache import Cache
from tqdm import tqdm
import hashlib

SRC_PATH = ["src"]
for module_path in SRC_PATH:
    if module_path not in sys.path:
        sys.path.append(module_path)
from utils import *



def load_test_dataset(dataset_path):

    dataset = load_from_disk(dataset_path)
    try:
        dataset_test = dataset["test"]
    except KeyError:
        dataset_test = dataset

    return dataset_test

def predict_gpt_zero(text, api_key):
    
    url = "https://api.gptzero.me/v2/predict/text"
    payload = {
        "document": text,
        "version": "2024-04-04",
        "multilingual": False
    }
    headers = {
        "Accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key
    }
    
    while True:
        try:
            time.sleep(1)  # 1 request per 10 minutes for free access
            response = requests.post(url, json=payload, headers=headers)
            print(response.json())
            #return response.json()['documents'][0]['completely_generated_prob']
            
            # try to access document
            response_doc = response.json()['documents'][0]
            return response.json()
        except Exception as ex:
            print(ex)

def generate_cache_key(args):
    args_str = [f"{key}={value}" for key, value in vars(args).items() if value is not None]
    args_str = "_".join(args_str)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()
    return args_hash



### MAIN FILE ###
def run(args):
    
    api_key = ""
    
    if args.use_api_key:
        api_key = os.environ.get("GPT_ZERO_API_KEY")

    # create experiment folder
    base_path = "saved_training_logs_experiment_2/gpt_zero"
    experiment_path = f"{base_path}"
    dataset_name = args.dataset_path.split("/")[-1]

    # load model
    dataset = load_test_dataset(args.dataset_path)
    dataset = dataset.select(range(args.sample_size))

    # iterate over the dataset
    preds = []
    probs = []
    
    # setup caching
    cache = Cache("cache_dir", size_limit=2e10)
    cache.reset("cull_limit", 0)
    
    # if reset_cache is set, clear the cache
    if args.reset_cache:
        print("Resetting cache")
        cache.clear()
    
    # if cache_key is not in cache, set it to empty string to avoid key error
    if "cache_key" not in cache:
        cache["cache_key"] = ""
        
    # Generate an hash of the arguments to use check if the cache is still valid
    cache_key = generate_cache_key(args)
    
    if cache["cache_key"] != cache_key:
        print("Cache key has changed, resetting cache")
        
        # reset the cache
        cache.clear()
        cache["cache_key"] = cache_key
    
    set_used = "eval" if args.use_eval_set else "test"
    for elem in tqdm(dataset, desc=f"Predicting labels for {set_used} set"):
        
        # check if the result is cached
        if elem["text"] in cache:
            result = cache[elem["text"]]
            pred = result["pred"]
            prob = result["prob"]
            preds.append(pred)
            probs.append(prob)
            continue
        
        text = elem["text"]
        
        pred_json = predict_gpt_zero(text, api_key=api_key)
        pred_json_doc = pred_json["documents"][0]
        pred_class = pred_json_doc["predicted_class"]
        
        if pred_class == "human":
            pred = 0
            
        elif pred_class == "ai":
            pred = 1
            
        elif pred_class == "mixed":
            
            pred_score_ai = pred_json_doc["class_probabilities"]["ai"]
            pred_score_human = pred_json_doc["class_probabilities"]["human"]
            pred = 1 if pred_score_ai > pred_score_human else 0
            
        else:
            raise ValueError("Unknown class")
        
        #pred = 0
        preds.append(pred)
        
        # record probability for positive class
        prob = pred_json_doc["class_probabilities"]["ai"]
        #prob = 0.5
        probs.append(prob)
        
        
        # cache the result
        cache[elem["text"]] = {
            "pred": pred,
            "prob": prob
        }
        
        #time.sleep(1)
        
    # close the cache, maybe use a with statement, but it's not in the documentation
    cache.close()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None, required=True)
    parser.add_argument('--sample_size', type=int, default=100, required=True)
    parser.add_argument('--classifier_threshold', type=float, default=None)
    parser.add_argument('--use_eval_set', action='store_true', default=False)
    parser.add_argument('--reset_cache', action='store_true', default=False)
    parser.add_argument('--use_api_key', action='store_true', default=False)
    args = parser.parse_args()

    run(args)



