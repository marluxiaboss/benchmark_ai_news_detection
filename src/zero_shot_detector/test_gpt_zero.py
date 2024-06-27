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



def predict_gpt_zero(text, api_key, debug_mode=False):
    
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
            # 1 request per 10 minutes for free access
            
            # 0.06 should correspond to 1000 requests per 1 minute
            #time.sleep(0.06)  
            response = requests.post(url, json=payload, headers=headers)
            
            if debug_mode:
                print(response.json())

            #return response.json()['documents'][0]['completely_generated_prob']
            
            # try to access document
            response_doc = response.json()['documents'][0]
            return response.json()
        except Exception as ex:
            print("Issue with prediction, skipping (see error below):")
            print("response: ", response)
            print(ex)
            
            # better to skip since no point in retrying
            return None


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
    dataset = load_test_dataset(args.dataset_path, args.use_eval_set)
    
    if args.sample_size is not None:
        dataset = dataset.select(range(args.sample_size))

    # iterate over the dataset
    pred_res_list = []

    set_used = "eval" if args.use_eval_set else "test"
    for elem in tqdm(dataset, desc=f"Predicting labels for {set_used} set"):  
        text = elem["text"]
        
        pred_json = predict_gpt_zero(text, api_key=api_key, debug_mode=args.debug_mode)
        
        # if prediction failed
        if pred_json is None:
            pred_res = {}
            pred_res["text"] = text
            pred_res["pred"] = None
            pred_res["prob"] = None
            
            pred_res_list.append(pred_res)
        
        else:
        
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
                
                # if mixed is higher prob than human and ai, set to 1
                if (pred_json_doc["class_probabilities"]["mixed"] > pred_score_ai and
                    pred_json_doc["class_probabilities"]["mixed"] > pred_score_human):
                    pred = 1
                
            else:
                raise ValueError("Unknown class")
            
            # record probability for positive class (mixed considered as positive class)
            prob = pred_json_doc["class_probabilities"]["ai"] + pred_json_doc["class_probabilities"]["mixed"]
            
            # create prediction res dict to tie results to text
            pred_res = {}
            
            pred_res["text"] = text
            pred_res["pred"] = pred
            pred_res["prob"] = prob
            
            pred_res_list.append(pred_res)
        
    # create label_text dict to tie labels to text and join with results
    labels_text = [{"text": elem["text"], "label": elem["label"]} for elem in dataset]
    
    # alligned preds, probs and labels such that they are in the same order even if some predictions failed
    preds = []
    probs = []
    labels = []
    
    # skip none pred/prob results
    for i in range(len(labels_text)):
        
        labels_text_i = labels_text[i]
        pred_res_i = pred_res_list[i]
        
        if pred_res_i["pred"] is not None:
            preds.append(pred_res_i["pred"])
            probs.append(pred_res_i["prob"])
            labels.append(labels_text_i["label"])
    
    # calculate accuracy
    preds = np.array(preds)
    labels = np.array(labels)
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
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--classifier_threshold', type=float, default=None)
    parser.add_argument('--use_eval_set', action='store_true', default=False)
    parser.add_argument('--use_api_key', action='store_true', default=False)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    args = parser.parse_args()

    run(args)



