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

from generator import LLMGenerator
from model_loader import load_generator
    
class ArticleGenerator:

    """
    Generates news article given a prefix, a model and an optional prompt
    
    Parameters:
    model : LLMGenerator
        The model used to generate the articles
    tokenizer : AutoTokenizer
        The tokenizer used to tokenize the text
    device : str
        The device used for the model
    
    """

    def __init__(self, model: LLMGenerator, tokenizer: AutoTokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_articles(self, prefixes_with_prompt: list, prefixes: list, batch_size: int = 4) -> list:
        """
        Generate articles given a list of prefixes and a list of prefixes with prompt
        
        Parameters:
        prefixes_with_prompt : list
            List of prefixes with prompt for the generation
        prefixes : list
            Same list as prefixes_with_prompt but without the prompt
        batch_size : int, optional
            Batch size for the generation, by default 4
            
        Returns:
        list
            List of generated articles
        """

        articles = []
        with torch.no_grad():
            for i in tqdm(range(0, len(prefixes_with_prompt), batch_size)):
                if i + batch_size > len(prefixes_with_prompt):
                    samples = prefixes_with_prompt[i:]
                else:
                    samples = prefixes_with_prompt[i:i+batch_size]
                    
                # increase max_length to take into account the length of the prompt,
                # since we use the same type of prompt for all samples, it should be on average the same length
                first_sample = samples[0]
                first_sample_tokenized_length = len(self.tokenizer(first_sample)["input_ids"])
                
                # TODO: should be a parameter
                min_new_tokens = 200
                max_length = min_new_tokens + first_sample_tokenized_length 
                outputs = self.model(samples, max_new_tokens=max_length, min_new_tokens=min_new_tokens)
                res = [text.replace("\n", "") for text in outputs]
                final_res = []
                for j, _ in enumerate(res):
                    # some models add already a space at the beginning of the text
                    if res[j][0] == " ":
                        final_res.append(f"{prefixes[j + i]}{res[j]}")
                    else:
                        final_res.append(f"{prefixes[j + i]} {res[j]}")
                articles.extend(final_res)
                #res = [f"{prefixes[j + i]} {res[j]}" for j in range(len(res))]

        return articles
    
def transform_chat_template_with_prompt(prefix: str, prompt: str, tokenizer: AutoTokenizer,
                                        use_chat_template: bool = False, template_type: str = None, system_prompt: str = "") -> str:
    
    """
    Transform a prefix with a prompt into a chat template
    
    Parameters:
    prefix : str
        The prefix to use
    prompt : str
        The prompt to use
    tokenizer : AutoTokenizer
        The tokenizer to use
    use_chat_template : bool, optional
        Whether to use a chat template, by default False
    template_type : str, optional
        The type of template to use, by default None
    system_prompt : str, optional
        The system prompt to use, by default ""
        
    Returns:
    str
        The transformed prefix
    """
        

    if prefix != "":
        text_instruction = f"{prompt} {prefix}"
    else:
        text_instruction = prompt
        
    if use_chat_template:
        if system_prompt == "":
            sys_prompt = "You are a helpful assistant."
        else:
            sys_prompt = system_prompt
        match template_type:
            case "system_user":
                messages = [
                {"role": "system", "content": f"{sys_prompt}"},
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


def regroup_pairs(merged_dataset: Dataset, seed=42) -> Dataset:
    """
    Regroup pairs of true and fake responses two by two so that they are in the same batch and in the same split.
    
    Parameters:
    merged_dataset : Dataset
        The dataset to regroup
    seed : int, optional
        The seed to use for the shuffling, by default 42
        
    """

    def fix_ids(dataset: Dataset):
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

    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--dataset_name_suffix", type=str, help="Suffix to add to the dataset name", default="new")
    parser.add_argument("--test_only", help="Whether to only keep the test split", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_article_generator", help="Whether to use the article generator", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--article_generator", type=str, help="Generator used to generate the articles, it should be a chat model", default="zephyr")
    parser.add_argument("--system_prompt", type=str, help="Prompt to use for the system in the chat template", default="")
    parser.add_argument("--prompt", type=str, help="Prompt to use for the article generator", default="")
    parser.add_argument("--use_llm_paraphraser", help="Whether to use the article generator model for paraphrasing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--nb_paraphrasing", type=int, help="Number of nested paraphrasing to do", default=1)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", default=-1)
    parser.add_argument("--batch_size", type=int, help="Batch size for the paraphrasing", default=4)
    parser.add_argument("--temperature", type=float, help="Temperature for the generation, default one if not set", default=0.8)
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty for the generation", default=1.0)
    parser.add_argument("--checkpoint_path", type=str, help="Whether to load the model from a checkpoint (Only supported for Zephyr)", default=None)

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
    
    # We should not be able to both generate articles and paraphrase in one run
    if args.use_article_generator and args.use_llm_paraphraser:
        raise ValueError("You cannot set both use_article_generator and use_llm_paraphraser to True")
    
    # whether to paraphrase using the provided LLM.
    # in this case, the fake articles are already generated
    if args.use_llm_paraphraser:
        
        generator = args.article_generator
        print(f"Using article generator with {generator}")

        # load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, use_chat_template, template_type = load_generator(generator, device, temperature=args.temperature, repetition_penalty=args.repetition_penalty)
        article_generator = ArticleGenerator(model, tokenizer, device)

        # take the prefixes from the dataset
        dataset_list = fake_dataset["text"]
        prefix_len = 10
        prefixes = [" ".join(text.split()[:prefix_len]) for text in dataset_list]
        
        system_paraphrasing_prompt = ("You are a paraphraser. You are given an input passage ‘INPUT’. You should paraphrase ‘INPUT’ to print ‘OUTPUT’."
"‘OUTPUT’ shoud be diverse and different as much as possible from ‘INPUT’ and should not copy any part verbatim from ‘INPUT’."
"‘OUTPUT’ should preserve the meaning and content of ’INPUT’ while maintaining text quality and grammar."
"‘OUTPUT’ should not be much longer than ‘INPUT’. You should print ‘OUTPUT’ and nothing else so that its easy for me to parse.")
        
        user_paraphrasing_prompts = [f"INPUT: {fake_text}" for fake_text in dataset_list]
        
        # apply the chat template with the prompt
        prefixes_with_prompt = [transform_chat_template_with_prompt(prefix, paraphrase_prompt, tokenizer, use_chat_template, template_type, system_paraphrasing_prompt) for prefix, paraphrase_prompt in zip(prefixes, user_paraphrasing_prompts)]
        
        # generate articles
        for i in range(args.nb_paraphrasing):
            fake_articles = article_generator.generate_articles(prefixes_with_prompt, prefixes, batch_size=args.batch_size)
            prefixes = [" ".join(text.split()[:prefix_len]) for text in fake_articles]
            prefixes_with_prompt = [transform_chat_template_with_prompt(prefix, paraphrase_prompt, tokenizer, use_chat_template, template_type, system_paraphrasing_prompt) for prefix, paraphrase_prompt in zip(prefixes, user_paraphrasing_prompts)]

        true_dataset = dataset.filter(lambda x: x["label"] == 0)
        
        # cut the articles to a certain length (in number of characters)
        article_len = 500
        
        # find indices of articles that are less than article_len characters
        indices_short_articles = [i for i, article in enumerate(fake_articles) if len(article) < article_len]
        print("Percentage of short articles: ", len(indices_short_articles) / len(fake_articles) * 100)
        
        # remove short articles from the fake dataset and the corresponding true articles
        #fake_articles = [article for i, article in enumerate(fake_articles) if i not in indices_short_articles]
        #true_dataset = true_dataset.select([i for i in range(len(true_dataset)) if i not in indices_short_articles])
        
        fake_dataset = Dataset.from_dict({"text": [text[:article_len] for text in fake_articles], "label": [1] * len(fake_articles)})
        true_dataset = Dataset.from_dict({"text": true_dataset["text"], "label": [0] * len(true_dataset["text"])})

        # regroup the pairs to re-create the dataset as it was before
        dataset = concatenate_datasets([true_dataset, fake_dataset])
        dataset = regroup_pairs(dataset)
    
    # if this is set to true, paraphrasing should be set to false
    if args.use_article_generator:
        generator = args.article_generator
        print(f"Using article generator with {generator}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model and set the generation parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, use_chat_template, template_type = load_generator(generator, device, temperature=args.temperature, repetition_penalty=args.repetition_penalty, checkpoint_path=args.checkpoint_path)
        article_generator = ArticleGenerator(model, tokenizer, device)

        # take the prefixes from the dataset
        dataset_list = fake_dataset["text"]
        prefix_len = 10
        prefixes = [" ".join(text.split()[:prefix_len]) for text in dataset_list]

        # apply the chat template with the prompt
        prefixes_with_prompt = [transform_chat_template_with_prompt(prefix, args.prompt, tokenizer, use_chat_template, template_type, args.system_prompt) for prefix in prefixes]

        # generate articles
        fake_articles = article_generator.generate_articles(prefixes_with_prompt, prefixes, batch_size=args.batch_size)

        true_dataset = dataset.filter(lambda x: x["label"] == 0)
        
        article_len = 500
        # find indices of articles that are less than article_len characters
        indices_short_articles = [i for i, article in enumerate(fake_articles) if len(article) < article_len]
        print("Percentage of short articles: ", len(indices_short_articles) / len(fake_articles) * 100)
        
        # remove short articles from the fake dataset and the corresponding true articles
        #fake_articles = [article for i, article in enumerate(fake_articles) if i not in indices_short_articles]
        #true_dataset = true_dataset.select([i for i in range(len(true_dataset)) if i not in indices_short_articles])
        
        
        fake_dataset = Dataset.from_dict({"text": [text[:article_len] for text in fake_articles], "label": [1] * len(fake_articles)})
        true_dataset = Dataset.from_dict({"text": true_dataset["text"], "label": [0] * len(true_dataset["text"])})

        # regroup the pairs to re-create the dataset as it was before
        dataset = concatenate_datasets([true_dataset, fake_dataset])
        dataset = regroup_pairs(dataset)

        
    # save the results to a subfolder of the original dataset
    modified_dataset_folder_base = dataset_path.split("/")[0] + "/modified_datasets"
    dataset_name = dataset_path.split("/")[-1]

    if args.test_only:
        dataset = DatasetDict({"test": dataset})
        
    dataset.save_to_disk(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}")

    if args.test_only:
        df_test = pd.DataFrame(dataset)
        #df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))
        df_test.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_test.json", force_ascii=False, indent=4)
    # load to pandas train split
    else:
        df_train = pd.DataFrame(dataset['train'])
        df_eval = pd.DataFrame(dataset['valid'])

        # transform text to list by splitting on \n
        #df_train["text"] = df_train["text"].apply(lambda x: x.split("\n"))
        #df_eval["text"] = df_eval["text"].apply(lambda x: x.split("\n"))

        # dump to json
        df_train.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_train.json", force_ascii=False, indent=4)
        df_eval.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_eval.json", force_ascii=False, indent=4)

        df_test = pd.DataFrame(dataset['test'])
        df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))
        df_test.to_json(f"{modified_dataset_folder_base}/{dataset_name}_{args.dataset_name_suffix}_test.json", force_ascii=False, indent=4)



    



