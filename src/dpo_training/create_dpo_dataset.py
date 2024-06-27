from datasets import concatenate_datasets, load_from_disk, DatasetDict, Dataset, load_dataset
import argparse
import os
import json
import pandas as pd
import numpy as np


from transformers import AutoTokenizer

import sys
SRC_PATH = ["src"]
for module_path in SRC_PATH:
    if module_path not in sys.path:
        sys.path.append(module_path)
from utils import *


from zero_shot_detector.test_fast_detect_gpt import predict_on_dataset


def apply_chat_template_and_prepare_for_dpo(
    sample, tokenizer):
    """
    Taken from https://github.com/argilla-io/notus/blob/main/v1/fine-tune/run_dpo.py
    """
    
    prompt = [
        {
            "role": "system",
            # Maybe content = "You are a helpful assistant."
            "content": "",
        },
        {
            "role": "user",
            "content": sample["prompt"],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": sample["chosen"],
        }
    ]
    rejected = [
        {
            "role": "assistant",
            "content": sample["rejected"],
        }
    ]
    
    sample["prompt"] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    sample["chosen"] = tokenizer.apply_chat_template(chosen, tokenize=False)
    sample["rejected"] = tokenizer.apply_chat_template(rejected, tokenize=False)
    
    return {
        "prompt": sample["prompt"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
    }
    
    
if __name__ == "__main__":
    
    # e.g. python src/format_out_of_domain_dataset.py --save_path=fake_true_datasets/xsum_true_only_test --original_dataset_path=EdinburghNLP/xsum --dataset_name=xsum_only_true --take_samples=10000 --orig_dataset_type=huggingface
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to save the dataset", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to the original json dataset", required=True)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", required=False, default=-1)
    parser.add_argument("--model_name", type=str, help="Name of the model that will be used for DPO, used for the tokenizer to have the chat template", required=True, default="zephyr")
    args = parser.parse_args()

    
    # plan:
    # 1) Load fake news articles dataset (10k samples)
    # 2) Run FastDetectGPT to create labels for the datasetl. 0 = true -> accepted, 1 = fake -> rejected
    # 3) Format into DPO format
    
    
    # 0) Load the tokenizer for the chat template
    model_name = args.model_name
    
    if model_name == "zephyr":
        gen_path = "HuggingFaceH4/zephyr-7b-beta"
        tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
    
    # 1) Load fake news articles dataset (10k samples)
    dataset_path = args.dataset_path

    # load the dataset, we only take the test split
    dataset_full = load_from_disk(dataset_path)

    #if args.test_only:
    #    dataset = dataset_full["test"]
    #else:
    dataset = dataset_full["train"]
        
    if args.take_samples > 0:
        dataset = dataset.select(range(args.take_samples))
        
    # 2) Run FastDetectGPT to create labels for the datasetl. 0 = true -> accepted, 1 = fake -> rejected
    preds, labels = predict_on_dataset(dataset)
    
    # set labels to random 0 or 1 for testing purposes
    #preds = np.random.randint(0, 2, len(dataset))
    #labels = np.random.randint(0, 2, len(dataset))
    
    print("preds", preds)
    print("labels", labels)
        
    # 3) Format into DPO format    
    instructions = []
    chosen_responses = []
    rejected_responses = []
    prefix_length = 10
    prompt = "Continue to write this news article:"
    
    previous_prefix = ""
    previous_previous_prefix = ""
        
    # iterate over the dataset and create the DPO format
    for i, sample in enumerate(dataset):
        
        # don't consider the last sample if dataset has odd length
        if len(dataset) % 2 == 1 and i == len(dataset) - 1:
            break
        
        prefix = " ".join(sample["text"].split()[:prefix_length])
        
        
        # if the current prefix is different than the previous one and the previous one is different than the one before, we skip
        # it would break the pairing
        # we wait fo i > 2 because otherwise previous_prefix and previous_previous_prefix are not defined
        
        if i > 2 and (prefix != previous_prefix and previous_prefix != previous_previous_prefix):
            
            previous_previous_prefix = previous_prefix
            previous_prefix = prefix
            continue
        else:
            if prefix != previous_prefix:
                prefix_with_prompt = f"{prompt} {prefix}"
                instructions.append(prefix_with_prompt)
            
            label = sample["label"]

            if label == 0:
                chosen_responses.append(sample["text"])
                
            if label == 1:
                rejected_responses.append(sample["text"])
            
            previous_previous_prefix = previous_prefix
            previous_prefix = prefix


    print("len instructions", len(instructions))
    print("len chosen_responses", len(chosen_responses))
    print("len rejected_responses", len(rejected_responses))

    dpo_dict = {"prompt": instructions, "chosen": chosen_responses, "rejected": rejected_responses}
    
    # split into train, valid, test
    train_size = int(0.8 * len(dpo_dict["prompt"]))
    valid_size = int(0.1 * len(dpo_dict["prompt"]))
    test_size = int(0.1 * len(dpo_dict["prompt"]))
    
    train_indices = np.random.choice(len(dpo_dict["prompt"]), train_size, replace=False)
    valid_indices = np.random.choice(list(set(range(len(dpo_dict["prompt"]))) - set(train_indices)), valid_size, replace=False)
    test_indices = list(set(range(len(dpo_dict["prompt"]))) - set(train_indices) - set(valid_indices))
    
    dpo_dict_train = {"prompt": [dpo_dict["prompt"][i] for i in train_indices], "chosen": [dpo_dict["chosen"][i] for i in train_indices],
                      "rejected": [dpo_dict["rejected"][i] for i in train_indices]}
    dpo_dict_valid = {"prompt": [dpo_dict["prompt"][i] for i in valid_indices], "chosen": [dpo_dict["chosen"][i] for i in valid_indices],
                      "rejected": [dpo_dict["rejected"][i] for i in valid_indices]}
    dpo_dict_test = {"prompt": [dpo_dict["prompt"][i] for i in test_indices], "chosen": [dpo_dict["chosen"][i] for i in test_indices],
                     "rejected": [dpo_dict["rejected"][i] for i in test_indices]}

    dpo_dataset_dict = DatasetDict({"train": Dataset.from_dict(dpo_dict_train), "valid": Dataset.from_dict(dpo_dict_valid), "test": Dataset.from_dict(dpo_dict_test)})
    dpo_dataset_dict_chat = dpo_dataset_dict.map(lambda x: apply_chat_template_and_prepare_for_dpo(x, tokenizer))
    
    # transform to "prompt": [...], "chosen": [...], "rejected": [...] format
    
    #dpo_dataset_dict_chat = dpo_dataset_dict_chat.map(lambda x: [{"prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"]}])
    
    # save the dataset
    dpo_dataset_dict_chat.save_to_disk(args.save_path)
    
    dpo_dataset_train_pd = pd.DataFrame(dpo_dataset_dict_chat["train"])
    dpo_dataset_valid_pd = pd.DataFrame(dpo_dataset_dict_chat["valid"])
    dpo_dataset_test_pd = pd.DataFrame(dpo_dataset_dict_chat["test"])
    
    # save the dataset to json too
    dpo_dataset_train_pd.to_json(f"{args.save_path}_train.json", indent=4)
    dpo_dataset_valid_pd.to_json(f"{args.save_path}_valid.json", indent=4)
    dpo_dataset_test_pd.to_json(f"{args.save_path}_test.json", indent=4)