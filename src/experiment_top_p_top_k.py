from datasets import concatenate_datasets, load_from_disk, DatasetDict
import argparse
import os
import pandas as pd
import copy
from tqdm import tqdm

import torch
import nltk.data
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, concatenate_datasets, Dataset

from abc import ABC, abstractmethod

from model_loader import load_generator

import matplotlib.pyplot as plt
import altair as alt
from vega_datasets import data

class ArticleGenerator:

    """
    Generates news article given a prefix, a model and an optional prompt
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_articles(self, prefixes_with_prompt: list, prefixes: list, batch_size=4) -> list:

        articles = []
        distributions_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(prefixes_with_prompt), batch_size)):
                if i + batch_size > len(prefixes_with_prompt):
                    samples = prefixes_with_prompt[i:]
                else:
                    samples = prefixes_with_prompt[i:i+batch_size]
                    
                for j in range(len(samples)):
                    
                    # distribution is a list of dict containing the probabilities distribution of each generated token
                    outputs, distributions = self.model.forward_with_distribution(samples[j])
                    distributions_list.extend(distributions)
                    
                    
                    
        # count nb of retained tokens per distribution
        nb_retained_tokens = [len(distribution) for distribution in distributions_list]
        
        return articles, nb_retained_tokens, distributions_list
    
def transform_to_chat_template_with_prompt(prefix, prompt, tokenizer, use_chat_template=False, template_type=None):

    if prefix != "":
        text_instruction = f"{prompt} {prefix}"
    else:
        text_instruction = prompt
        
    if use_chat_template:
        match template_type:
            case "system_user":
                messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{text_instruction}"},
                ]
            case "user":
                messages = [
                {"role": "user", "content": f"{text_instruction}"},
                ]
            case _:
                raise ValueError("Template type not supported")

        text_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # force prefix on the generated response
        text_template = f"{text_template}\n{prefix}"

    else:
        text_template = text_instruction

    return text_template

def regroup_pairs(merged_dataset, seed=42):
    """
    Regroup pairs of true and fake responses two by two so that they are in the same batch and in the same split
    """

    def fix_ids(dataset):
        """
        Fix the ids of the dataset
        """
        fake_responses_dataset = dataset.filter(lambda x: x["label"] == 1)
        true_responses_dataset = dataset.filter(lambda x: x["label"] == 0)

        fake_responses_text = fake_responses_dataset["text"]
        true_responses_text = true_responses_dataset["text"]

        correct_text_ordering_fake = []
        correct_text_ordering_true = []

        for i, _ in enumerate(fake_responses_text):

            fake_response = fake_responses_text[i]

            # find the prefix in true_dataset
            prefix = " ".join(fake_response.split()[:10])

            for j, _ in enumerate(true_responses_text):
                if " ".join(true_responses_text[j].split()[:10]) == prefix:
                    #correct_ids_fake_dataset.append(true_reponses_labels[i])
                    correct_text_ordering_true.append(j)
                    correct_text_ordering_fake.append(i)
                    break   

        # reorganize the fake responses according to the correct order
        fake_responses_dataset = fake_responses_dataset.select(correct_text_ordering_fake)

        # remove true_responses without a corresponding fake response
        true_responses_dataset = true_responses_dataset.select(correct_text_ordering_true)

        # add an id column to fake and true responses datasets
        fake_responses_dataset = fake_responses_dataset.add_column("id", list(range(len(fake_responses_dataset))))
        true_responses_dataset = true_responses_dataset.add_column("id", list(range(len(true_responses_dataset))))
                                                                   
        dataset = concatenate_datasets([true_responses_dataset, fake_responses_dataset])

        # shuffle the dataset to mix between true and fake responses within pairs and sort by id to have the correct order again
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.sort("id")
        dataset = dataset.remove_columns("id")

        return dataset
    
    # shuffle the dataset
    merged_dataset = merged_dataset.shuffle(seed=seed)

    # ids may be incorrect for label 1, we need to fix them
    merged_dataset = fix_ids(merged_dataset)

    return merged_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--article_generator", type=str, help="Generator used to generate the articles, it should be a chat model", default="zephyr")
    parser.add_argument("--test_only", action="store_true", help="If set, only the test split will be used")
    parser.add_argument("--take_samples", type=int, default=-1, help="If set, only take a subset of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--prompt", type=str, default="", help="")

    args = parser.parse_args()


    dataset_path = args.dataset_path

    # load the dataset, we only take the test split
    dataset_full = load_from_disk(dataset_path)

    if args.test_only:
        dataset = dataset_full["test"]

        if args.take_samples > 0:
            dataset = dataset.select(range(args.take_samples))
    else:
        dataset = dataset_full

    # only keep fake samples
    fake_dataset = dataset.filter(lambda x: x["label"] == 1)

    generator = args.article_generator
    print(f"Using article generator with {generator}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, use_chat_template, template_type = load_generator(generator, device,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        top_k=args.top_k
    )
    article_generator = ArticleGenerator(model, tokenizer, device)

    # take the prefixes from the dataset
    dataset_list = fake_dataset["text"]
    prefix_len = 10
    prefixes = [" ".join(text.split()[:prefix_len]) for text in dataset_list]

    # apply the chat template with the prompt
    prefixes_with_prompt = [transform_to_chat_template_with_prompt(prefix, args.prompt, tokenizer, use_chat_template, template_type) for prefix in prefixes]

    # generate articles
    articles, nb_retained_tokens, distributions_list = article_generator.generate_articles(prefixes_with_prompt, prefixes, batch_size=args.batch_size)
    
    # find max retained tokens index
    max_retained_tokens_index = nb_retained_tokens.index(max(nb_retained_tokens))
    print("distribution of the max retained tokens:", distributions_list[max_retained_tokens_index])
    
    # remove trivial cases where the number of retained tokens is 2 or less
    
    # plot the distribution of the number of retained tokens
    #plt.hist(nb_retained_tokens, bins=50)
    #plt.xlabel("Number of retained tokens")
    #plt.ylabel("Number of samples")
    #plt.title("Distribution of the number of retained tokens")
    #plt.savefig("nb_retained_tokens.png")
    
    # create a dataframe with 1 column number of retained tokens
    df_retained_tokens = pd.DataFrame(nb_retained_tokens, columns=["nb_retained_tokens"])
    
    # save the dataframe to a json file
    filename = f"nb_retained_tokens_top_p_{args.top_p}"
    df_retained_tokens.to_json(f"{filename}.json")
    
    #alt.Chart(df).mark_bar().encode(
    #    x=alt.X("nb_retained_tokens:Q", bin=True),
    #    y='count()',
    #)
    
    
