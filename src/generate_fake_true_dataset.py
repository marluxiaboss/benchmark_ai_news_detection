from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets, disable_progress_bar, enable_progress_bar
import numpy as np
import pandas
import argparse
from tqdm import tqdm
import os
import torch
import pandas as pd
import json

from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer)

from generator import LLMGenerator
from model_loader import load_generator

def format_news_dataset(true_dataset: Dataset, prefix_cutoff: int = 10, max_response_length_char: int = 500) -> DatasetDict:
    """
    Function used to format the news dataset, specifically the cnn_dailymail dataset
    TODO: Should be its own script
    
    Parameters:
    true_dataset : Dataset
        The dataset to format
    prefix_cutoff : int, optional
        The number of words to keep in the instruction, by default 10
    max_response_length_char : int, optional
        The maximum length of the response in characters, by default 500
        
    Returns:
    DatasetDict
        The formatted dataset
    """

    def remove_bloat(sample):
        filtered_text = sample["article"]
        nb_separator = filtered_text.count("--")
        if nb_separator > 0:
            filtered_text = filtered_text.split("--", 1)[1].strip()

        # heurstic to handle cases where the instruction contains an input of this type:
        # By . Jill Reilly . PUBLISHED: . 08:21 EST, 6 December 2012 . | . UPDATED: . 16:19 EST, 6
        if "EST," in filtered_text.split():
            split_est = filtered_text.split("EST,")
            count_est = len(split_est)
            filtered_text = split_est[count_est-1].split()[4:]
            filtered_text = " ".join(filtered_text)
        return {"article": filtered_text}

    def format_news(sample):
        
        sample["instruction"] = " ".join(sample["article"].split()[:prefix_cutoff])
        sample["context"] = ""
        sample["response"] = " ".join(sample["article"].split()[prefix_cutoff:])
        
        # cut response to max_response_length_char, even if it cuts a word
        sample["response"] = sample["response"][:max_response_length_char]
        sample["category"] = "news"
        return sample

    true_dataset = true_dataset.map(remove_bloat)
    true_dataset = true_dataset.map(format_news)
    true_dataset = true_dataset.remove_columns(["article"])
    true_dataset = true_dataset.remove_columns(["highlights"])

    true_dataset = DatasetDict({"train": true_dataset})
    return true_dataset

def create_train_from_dataset(dataset: Dataset) -> DatasetDict:
    """
    Create a train split from a dataset
    
    Parameters:
    dataset : Dataset
        The dataset to create the train split from
    
    Returns:
    DatasetDict
        The dataset with the train split
    
    """

    dataset_dict = DatasetDict()
    dataset_dict["train"] = dataset

    return dataset_dict

def create_random_subset(dataset: Dataset, n: int = 10, seed: int = 42) -> Dataset:
    """
    Create a random subset of the dataset
    
    Parameters:
    dataset : Dataset
        The dataset to create the random subset from
    n : int, optional
        The size of the subset, by default 10
    seed : int, optional
        The seed for the random number generator, by default 42
    
    Returns:
    Dataset
        The random subset of the dataset
    """
    if n > len(dataset):
        n = len(dataset)
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), n, replace=False)
    subset = dataset.select(indices)
    return subset

def filter_duplicates(dataset: Dataset, column: str) -> Dataset:
    """
    Filter out the duplicates in the dataset
    
    Parameters:
    dataset : Dataset
        The dataset to filter
    column : str 
        The column to filter duplicates on
    
    Returns:
    Dataset
        The dataset without duplicates
    """
    
    dataset_df = pd.DataFrame(dataset)
    len_before_discard = dataset_df.shape[0]

    dataset_df = dataset_df.drop_duplicates(subset=[column])

    len_after_discard = dataset_df.shape[0]
    print(f"Percent of data discarded after removing duplicate {column}: {100*(1 - len_after_discard/len_before_discard):.2f}%")

    return Dataset.from_pandas(dataset_df)

def process_true_dataset(true_dataset: Dataset) -> DatasetDict:
    """
    Process the true dataset by creating the necessary columns and selecting a size according to
    the fake dataset size.
    
    Parameters:
    true_dataset : Dataset
        The dataset of true samples (from the original dataset)
    
    Returns:
    Dataset
        The processed true dataset
    """

    true_dataset = true_dataset.select_columns(["response", "instruction", "context", "id"])
    true_dataset = true_dataset.rename_column("response", "text")

    # create label = 0 for true responses and label = 1 for fake responses
    true_dataset = true_dataset.map(lambda x: {"label": 0})

    # drop duplicates on instruction by transforming to pandas dataframe
    true_dataset_df = pd.DataFrame(true_dataset["train"])
    len_before_discard = len(true_dataset_df)
    true_dataset_df = true_dataset_df.drop_duplicates(subset=["instruction"])
    len_after_discard = len(true_dataset_df)
    print(f"Percent of data discarded after removing duplicate instructs from true_dataset_df: {100*(1 - len_after_discard/len_before_discard):.2f}%")

    # transform back to dataset
    true_dataset = Dataset.from_pandas(true_dataset_df)
    true_dataset = create_train_from_dataset(true_dataset)

    return true_dataset


def generate_fake_responses(generator : LLMGenerator, true_dataset : Dataset, gen_tokenizer : AutoTokenizer,
                            max_new_tokens : int, batch_size: int = 2, use_chat_template: bool = False,
                             template_type: str = None, prompt: str = "") -> list:
    """
    Traverse dataset and generate responses for each instruction.
    
    Parameters:
    generator : LLMGenerator
        The generator model
    true_dataset : Dataset
        The dataset of true samples
    gen_tokenizer : AutoTokenizer
        The tokenizer
    max_new_tokens : int
        The maximum number of tokens to generate
    batch_size : int, optional
        The batch size for generation, by default 2
    use_chat_template : bool, optional
        Whether to use the chat template, by default False
    template_type : str, optional
        The type of template to use, by default None
    prompt : str, optional
        The prompt to use for generation, by default ""
    
    Returns:
    list
        The list of fake responses
    """
    
    fake_responses = []

    # save to which instructions we have generated a reponse, used when loading from cache (old feature, not used anymore)
    instructions = []
    
    # transform into chat template
    def transform_chat_template(sample, use_chat_template=False):

        if use_chat_template:
            if sample["context"] != "":
                text_instruction = f"Context: {sample['context']} \n {prompt} {sample['instruction']}"
            else:
                text_instruction = f"{prompt} {sample['instruction']}"
            
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

            text_template = gen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # force prefix on the generated response
            text_template = f"{text_template}\n{sample["instruction"]}"
        else:
            text_instruction = f"{prompt} {sample["instruction"]}"
            text_template = text_instruction
        return {"text_template": text_template}
    
    true_dataset = true_dataset.map(lambda x: transform_chat_template(x, use_chat_template=use_chat_template))
    true_dataset_list = true_dataset["text_template"]
    
    for i in tqdm(range(0, len(true_dataset_list), batch_size), desc="Generating fake responses"):
        batch = true_dataset_list[i:i+batch_size]
        responses = generator(batch, max_new_tokens=max_new_tokens)
        fake_responses.extend(responses)

    return fake_responses, instructions

def filter_instruction(sample: dict):
    """
    Filter out the instruction from the generated response
    Note: only works if special tokens are not removed
    
    Parameters:
    sample : dict
        The sample to filter
        
    Returns:
    dict
        The sample with the instruction removed
    """
    
    # removing everything between system and user (including system and user)
    response_without_instruction = sample["generated_response"]

    # replace any number of newlines with a " "
    response_without_instruction = " ".join(response_without_instruction.split())

    return {"generated_response": response_without_instruction}

def generate_fake_dataset(true_dataset : DatasetDict, generator : LLMGenerator, gen_tokenizer: AutoTokenizer, max_nb_tokens_input: int = 100,
                           max_new_tokens: int = 100, batch_size: int = 2, use_chat_template: bool = False, template_type: str = None, prompt="") -> DatasetDict:
    """
    Generate a fake dataset from a true dataset by creating instructions from the true dataset and generating fake responses.
    
    Parameters:
    true_dataset : DatasetDict
        The dataset of true samples
    generator : LLMGenerator
        The generator model
    gen_tokenizer : AutoTokenizer
        The tokenizer
    max_nb_tokens_input : int, optional
        The maximum number of tokens for the input, by default 100
    max_new_tokens : int, optional
        The maximum number of tokens to generate, by default 100
    batch_size : int, optional
        The batch size for generation, by default 2
    use_chat_template : bool, optional
        Whether to use the chat template, by default False
    template_type : str, optional
        The type of chat template to use, by default None
    prompt : str, optional
        The prompt to use for generation, by default ""
        
    Returns:
    DatasetDict
        The fake dataset
    """
        
    
    # discard instructions that are more than max_nb_tokens_input tokens
    max_nb_tokens_input = max_nb_tokens_input
    train_subset = true_dataset["train"]
    
    fake_responses_train, instructions = generate_fake_responses(generator, train_subset, gen_tokenizer, max_new_tokens=max_new_tokens, batch_size=batch_size, use_chat_template=use_chat_template, template_type=template_type, prompt=prompt)

    print("len fake_responses_train: ", len(fake_responses_train))
    if instructions:
        train_subset = train_subset.filter(lambda x: x["instruction"] in instructions)

    fake_responses_train = Dataset.from_dict({"generated_response": fake_responses_train, "instruction": train_subset["instruction"],
                    "context": train_subset["context"], "true_response": train_subset["text"], "id": train_subset["id"]})
    
    # transform to pandas dataframe
    fake_responses_train_df = pd.DataFrame(fake_responses_train)

    # remove rows with duplicate instructions from df
    len_before_discard = len(fake_responses_train_df)
    fake_responses_train_df = fake_responses_train_df.drop_duplicates(subset=["instruction"])
    len_after_discard = len(fake_responses_train_df)
    print(f"Percent of data discarded after removing duplicate instructs from fake_responses_train_df: {100*(1 - len_after_discard/len_before_discard):.2f}%")

    # remove rows with duplicate responses from df
    len_before_discard = len(fake_responses_train_df)
    fake_responses_train_df = fake_responses_train_df.drop_duplicates(subset=["generated_response"])
    len_after_discard = len(fake_responses_train_df)
    print(f"Percent of data discarded after removing duplicate responses from fake_responses_train_df: {100*(1 - len_after_discard/len_before_discard):.2f}%")

    # transform back to dataset
    fake_responses_train = Dataset.from_pandas(fake_responses_train_df)

    # add ids again
    #fake_responses_train = fake_responses_train.add_column(ids, "id")

    fake_dataset = DatasetDict()
    fake_dataset["train"] = fake_responses_train

    return fake_dataset

def process_fake_dataset(fake_dataset: Dataset) -> Dataset:
    """
    Process the fake dataset by creating the necessary columns and selecting a size according to
    the true dataset size.
    
    Parameters:
    fake_dataset : Dataset
        The dataset of fake samples
        
    Returns:
    Dataset
        The processed fake dataset
    """

    # filter out instruction from generated_response
    fake_dataset = fake_dataset.map(lambda x: filter_instruction(x))

    # only select the fake samples
    fake_dataset = fake_dataset.map(lambda x: {"label": 1})

    # remove true_response from fake_dataset
    fake_dataset = fake_dataset.remove_columns(["true_response"])

    # rename and select columns
    fake_dataset = fake_dataset.rename_column("generated_response", "text")
    fake_dataset = fake_dataset.select_columns(["text", "label", "instruction", "context", "id"])

    return fake_dataset

def merge_true_fake_dataset(true_dataset: DatasetDict, fake_dataset: DatasetDict) -> DatasetDict:
    """
    Merge the true and fake datasets
    
    Parameters:
    true_dataset : DatasetDict
        The true dataset
        
    fake_dataset : DatasetDict
        The fake dataset
    
    Returns:
    DatasetDict
        The merged dataset
    """
    
    merged = concatenate_datasets([true_dataset["train"], fake_dataset["train"]])
    merged_dataset = DatasetDict()
    merged_dataset["train"] = merged

    return merged_dataset
def regroup_pairs(merged_dataset: DatasetDict, seed: int = 42) -> DatasetDict:
    """
    Regroup pairs of true and fake responses two by two so that they are in the same batch and in the same split.
    
    Parameters:
    merged_dataset : DatasetDict
        The merged dataset
    seed : int, optional
        The seed for the random number generator, by default 42
        
    Returns:
    DatasetDict
        The regrouped dataset
    
    """

    def fix_ids(dataset):
        """
        Fix the ids of the dataset so that the samples with same instruction have the same id
        """
        fake_responses_dataset = dataset.filter(lambda x: x["label"] == 1)["train"]
        true_responses_dataset = dataset.filter(lambda x: x["label"] == 0)["train"]

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
                    correct_text_ordering_true.append(j)
                    correct_text_ordering_fake.append(i)
                    break   

        # reorganize the fake responses according to the correct order
        fake_responses_dataset = fake_responses_dataset.select(correct_text_ordering_fake)

        # remove true_responses without a corresponding fake response
        true_responses_dataset = true_responses_dataset.select(correct_text_ordering_true)

        # sort both datasets by id to allign them, otherwise concat doesn't work
        true_responses_dataset = true_responses_dataset.sort("id")
        fake_responses_dataset = fake_responses_dataset.sort("id")

        dataset = concatenate_datasets([true_responses_dataset, fake_responses_dataset])
        dataset = create_train_from_dataset(dataset)

        # shuffle the dataset again to mix the true and fake responses
        dataset = dataset.shuffle(seed=seed)

        return dataset
    
    # shuffle the dataset
    merged_dataset = merged_dataset.shuffle(seed=seed)

    # ids may be incorrect for label 1, we need to fix them
    merged_dataset = fix_ids(merged_dataset)

    # sort the dataset by id
    merged_dataset = merged_dataset.sort("id")

    # remove id column
    merged_dataset = merged_dataset.remove_columns(["id"])
    print("merged_dataset: ", merged_dataset)

    return merged_dataset


def split_merged_dataset_random(merged_dataset: DatasetDict, eval_size: float = 0.1, test_size: float = 0.1) -> DatasetDict:
    """
    Create a train, eval, test split from the merged dataset
    taken from https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/6
    
    Parameters:
    merged_dataset : DatasetDict
        The merged dataset
    eval_size : float, optional
        The size of the evaluation set, by default 0.1
    test_size : float, optional
        The size of the test set, by default 0.1
        
    Returns:
    DatasetDict
        The split dataset
    """

    merged_dataset_train_test_valid = merged_dataset["train"].train_test_split(test_size=test_size + eval_size)

    test_valid = merged_dataset_train_test_valid['test'].train_test_split(test_size=test_size / (test_size + eval_size))

    merged_dataset = DatasetDict({
    'train': merged_dataset_train_test_valid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

    print("Train size:", len(merged_dataset['train']))
    print("Eval size:", len(test_valid['train']))
    print("Test size:", len(test_valid['test']))

    return merged_dataset

def split_merged_dataset(merged_dataset: DatasetDict, eval_size: float = 0.1, test_size: float = 0.1) -> DatasetDict:
    """
    Same as above, but assumes that the dataset is already shuffled.
    
    Parameters:
    merged_dataset : DatasetDict
        The merged dataset
    eval_size : float, optional
        The size of the evaluation set, by default 0.1
    test_size : float, optional
        The size of the test set, by default 0.1
        
    Returns:
    DatasetDict
        The split dataset
    """

    train_size = len(merged_dataset["train"])
    eval_size = int(train_size * eval_size)
    test_size = int(train_size * test_size)

    merged_dataset = DatasetDict({
    'train': merged_dataset["train"].select(range(train_size - eval_size - test_size)),
    'valid': merged_dataset["train"].select(range(train_size - eval_size - test_size, train_size - test_size)),
    'test': merged_dataset["train"].select(range(train_size - test_size, train_size))})

    print("Train size:", len(merged_dataset['train']))
    print("Eval size:", len(merged_dataset['valid']))
    print("Test size:", len(merged_dataset['test']))

    return merged_dataset

def balance_dataset(dataset: Dataset, create_train: bool = True) -> DatasetDict:
    """
    Rebalance the dataset so that the number of samples with label 0 and 1 are the same.
    
    Parameters:
    dataset : Dataset
        The dataset to balance
    create_train : bool, optional
        Whether to create a train split, by default True
        
    Returns:
    DatasetDict
        The balanced dataset
    """
    label_0 = dataset.filter(lambda x: x["label"] == 0)
    label_1 = dataset.filter(lambda x: x["label"] == 1)
    nb_label_0 = len(label_0["text"])
    nb_label_1 = len(label_1["text"])

    if nb_label_0 > nb_label_1:
        label_0 = label_0.select(range(nb_label_1))

    elif nb_label_1 > nb_label_0:
        label_1 = label_1.select(range(nb_label_0))
    
    merged_dataset = concatenate_datasets([label_0, label_1])
    if create_train:
        merged_dataset = create_train_from_dataset(merged_dataset)
    
    return merged_dataset

def format_merged_dataset(merged_dataset: DatasetDict, max_repsonse_length_char : int = 500) -> DatasetDict:
    """
    Format the text into a template.
    
    Parameters:
    merged_dataset : DatasetDict
        The merged dataset
    max_repsonse_length_char : int, optional
        The maximum length of the response in characters, by default 500
    
    Returns:
    DatasetDict
        The formatted dataset
    """

    def format_text(sample):

        text = sample["text"]
        if sample["label"] == 0:
            modified_text = sample["context"] + "" + sample["instruction"] + " " + text
        elif sample["label"] == 1:
            modified_text = sample["context"] + "" + sample["instruction"] + " " + text
        else:
            raise ValueError("Label not supported")

        return {"text": modified_text}
        
    merged_dataset = merged_dataset.map(format_text)

    # if max_repsonse_length_char > 0, cut the response to max_repsonse_length_char characters, even if it cuts a word
    if max_repsonse_length_char > 0:
        
        merged_dataset = merged_dataset.map(lambda x: {"text": x["text"][:max_repsonse_length_char]})
        # discard the samples that are not max_repsonse_length_char
        len_before_discard = len(merged_dataset["train"])
        merged_dataset = merged_dataset.filter(lambda x: len(x["text"]) == max_repsonse_length_char)
        len_after_discard = len(merged_dataset["train"])
        print(f"Percent of data discarded after filtering out input not equal to max_repsonse_length_char: {100*(1 - len_after_discard/len_before_discard):.2f}%")

        # check if the number of samples with label 0 and 1 are the same, if not, discard the extra samples
        label_0 = merged_dataset.filter(lambda x: x["label"] == 0)["train"]
        label_1 = merged_dataset.filter(lambda x: x["label"] == 1)["train"]
        nb_label_0 = len(label_0["text"])
        nb_label_1 = len(label_1["text"])
        print("Number of samples with label 0:", nb_label_0)
        print("Number of samples with label 1:", nb_label_1)

        merged_dataset = balance_dataset(merged_dataset["train"], create_train=True)

        # check the balancing again
        nb_label_0 = len(merged_dataset.filter(lambda x: x["label"] == 0)["train"]["text"])
        nb_label_1 = len(merged_dataset.filter(lambda x: x["label"] == 1)["train"]["text"])

        print("Number of samples with label 0:", nb_label_0)
        print("Number of samples with label 1:", nb_label_1)

    # only keep text and label
    merged_dataset = merged_dataset.select_columns(["text", "label", "id"])

    return merged_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--true_dataset_path", type=str, help="Path to the true dataset (hugginface dataset path)", default="cnn_dailymail")
    parser.add_argument("--fake_dataset_size", type=int, help="Size of the fake dataset", default=10)
    parser.add_argument("--max_nb_tokens_input", type=int, help="Max number of tokens for input", default=100)
    parser.add_argument("--generator", type=str, help="Generator model name between 'qwen', 'phi', 'gemma', 'mistral', 'gpt2'", default="gpt2")
    parser.add_argument("--device", type=str, help="Device to use for the generator", default="cuda")
    parser.add_argument("--validation_size", type=float, help="Size of the validation set", default=0.1)
    parser.add_argument("--test_size", type=float, help="Size of the test set", default=0.1)
    parser.add_argument("--max_new_tokens", type=int, help="Max length of the generated response", default=200)
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("--batch_size", type=int, help="Batch size for generation", default=2)
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment", default="test_experiment")
    parser.add_argument("--access_token", type=str, help="Huggingface access token used for Llama and Gemma", default="")
    parser.add_argument("--max_response_length", type=int, help="Max length of the response in characters", default=500)
    parser.add_argument("--prefix_cutoff", type=int, help="Number of words to keep in the instruction", default=10)
    parser.add_argument("--prompt", type=str, help="Prompt to use for generation, placed before the prefix", default="")
    parser.add_argument("--repetition_penalty", type=float, help="Controls repetition penalty parameter of generation", default=1.0)
    parser.add_argument("--temperature", type=float, help="Controls temperature parameter of generation", default=0.8)
    parser.add_argument("--top_p", type=float, help="Controls top_p parameter of generation", default=0.8)
    args = parser.parse_args()

    # check if the dataset already exists for the given experiment name
    if (os.path.exists(f"true_dataset_{args.experiment_name}")
        or os.path.exists(f"fake_dataset_{args.experiment_name}")
        or os.path.exists(f"fake_true_dataset_{args.experiment_name}")):
        print("Warning: dataset already exists for the given experiment name, do you want to overwrite it?")
        #   answer = input("[y/n]: ")
        #if answer != "y":
        #    exit()

    # set default parameters for generation
    default_gen_params = {
        "max_length": 200,
        "max_new_tokens": None,
        "temperature": 0.8,
        "top_p": 0.8,
        "repetition_penalty": 1,
        "do_sample": True,
        "min_new_tokens": 100
    }
    # TODO: add checks for test_size and validation_size, max_length and max_nb_tokens_input

    # load generator
    generator, gen_tokenizer, use_chat_template, template_type = load_generator(args.generator, args.device, args.access_token)
    
    gen_params = default_gen_params

    gen_params["repetition_penalty"] = args.repetition_penalty
    gen_params["temperature"] = args.temperature
    gen_params["top_p"] = args.top_p

    if args.true_dataset_path == "databricks/databricks-dolly-15k":
        # load true dataset
        true_dataset = load_dataset(args.true_dataset_path)

    elif args.true_dataset_path == 'cnn_dailymail':
        
        # load true dataset from disk
        true_dataset = load_dataset(args.true_dataset_path, "3.0.0")["train"]

        if 2 * args.fake_dataset_size < len(true_dataset):
            # make the dataset smaller in most cases to speed up the processing
            true_dataset = true_dataset.select(range(2 * args.fake_dataset_size))

        # format dataset to have the same columns as the other datasets: "instruction", "context", "response", "category"
        true_dataset = format_news_dataset(true_dataset, prefix_cutoff=args.prefix_cutoff, max_response_length_char=args.max_response_length)    
        true_dataset = filter_duplicates(true_dataset["train"], "instruction") 
        true_dataset = true_dataset.select(range(args.fake_dataset_size))

        true_dataset = create_train_from_dataset(true_dataset)

    else:
        raise ValueError("Dataset not supported")

    # process true dataset
    true_dataset = process_true_dataset(true_dataset)

    # generate fake dataset
    fake_dataset = generate_fake_dataset(true_dataset, generator, gen_tokenizer, args.max_nb_tokens_input, args.max_new_tokens,
                                          args.batch_size, use_chat_template=use_chat_template, template_type=template_type, prompt=args.prompt)

    # process fake dataset
    fake_dataset = process_fake_dataset(fake_dataset)

    # merge true and fake dataset
    merged_dataset = merge_true_fake_dataset(true_dataset, fake_dataset)

    # format merged dataset into a template
    merged_dataset = format_merged_dataset(merged_dataset, args.max_response_length)

    # group pairs of true and fake responses two by two so that they are in the same batch and in the same split
    merged_dataset = regroup_pairs(merged_dataset)

    nb_label_0 = len(merged_dataset["train"].filter(lambda x: x["label"] == 0)["text"])
    nb_label_1 = len(merged_dataset["train"].filter(lambda x: x["label"] == 1)["text"])
    print("Number of samples with label 0:", nb_label_0)
    print("Number of samples with label 1:", nb_label_1)

    # split merged dataset into train, eval, test
    merged_dataset = split_merged_dataset(merged_dataset, eval_size=args.validation_size, test_size=args.test_size)

    # check if folder "fake_true_datasets" exists
    if not os.path.exists("fake_true_datasets"):
        os.makedirs("fake_true_datasets")

    merged_dataset.save_to_disk(f"./fake_true_datasets/fake_true_dataset_{args.experiment_name}")

    # load to pandas train split
    df_train = pd.DataFrame(merged_dataset['train'])
    df_eval = pd.DataFrame(merged_dataset['valid'])
    df_test = pd.DataFrame(merged_dataset['test'])

    # transform text to list by splitting on \n
    df_train["text"] = df_train["text"].apply(lambda x: x.split("\n"))
    df_eval["text"] = df_eval["text"].apply(lambda x: x.split("\n"))
    df_test["text"] = df_test["text"].apply(lambda x: x.split("\n"))

    # dump to json
    df_train.to_json(f"./fake_true_datasets/fake_true_dataset_{args.experiment_name}_train.json", force_ascii=False, indent=4)
    df_eval.to_json(f"./fake_true_datasets/fake_true_dataset_{args.experiment_name}_eval.json", force_ascii=False, indent=4)
    df_test.to_json(f"./fake_true_datasets/fake_true_dataset_{args.experiment_name}_test.json", force_ascii=False, indent=4)






    

