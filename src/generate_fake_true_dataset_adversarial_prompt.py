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
    parser.add_argument("--load_from_cache", type=str, help="Load mutiple datasets chunk from cache", default="False")
    parser.add_argument("--prompt", type=str, help="Prompt to use for generation, placed before the prefix", default="")
    parser.add_argument("--repetition_penalty", type=float, help="Controls repetition penalty parameter of generation", default=1.0)
    parser.add_argument("--temperature", type=float, help="Controls temperature parameter of generation", default=0.8)
    parser.add_argument("--top_p", type=float, help="Controls top_p parameter of generation", default=0.8)
    args = parser.parse_args()

    # set default parameters for generation
    default_gen_params = {
        "max_length": 200,
        "max_new_tokens": None,
        "temperature": 0.8,
        "top_p": 0.8,
        "repetition_penalty": 1,
        "do_sample": True
    }
    # TODO: add checks for test_size and validation_size, max_length and max_nb_tokens_input

    # load generator
    if args.generator == "qwen_chat":
        gen_path = "Qwen/Qwen1.5-0.5B-Chat"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = True
        template_type ="system_user"

    elif args.generator == "qwen_0.5b":
        gen_path = "Qwen/Qwen1.5-0.5B"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None        

    elif args.generator == "gpt2":
        gen_path = "openai-community/gpt2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")

        gen_params = default_gen_params
        gen_params["repetition_penalty"] = 2.0
        
        # special for gpt2
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None
    elif args.generator == "gemma_2b_chat":
        gen_path = "google/gemma-2b-it"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=args.access_token, torch_dtype=torch.bfloat16).to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=args.access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, args.device, gen_params=default_gen_params)

        #template for chat
        use_chat_template = True
        template_type ="user"

    elif args.generator == "gemma_2b":
        gen_path = "google/gemma-2b"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=args.access_token, torch_dtype=torch.bfloat16).to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=args.access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, args.device, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None  

    elif args.generator == "phi":
        gen_path = "microsoft/phi-2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.float16).to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path)
        generator = LLMGenerator(gen_model, gen_tokenizer, args.device, gen_params=default_gen_params)

        # special for phi
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        #template for chat
        use_chat_template = False
        template_type = None  

    elif args.generator == "mistral":
        gen_path = "mistralai/Mistral-7B-v0.1"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        # special for mistral
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        #template for chat
        use_chat_template = False
        template_type = None  

    elif args.generator == "zephyr":
        gen_path = "HuggingFaceH4/zephyr-7b-beta"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(args.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        # special for mistral
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        #template for chat
        use_chat_template = True
        template_type ="user"


    else:
        # no other generator is supported for now
        raise ValueError("Generator not supported")
    
    gen_params = default_gen_params

    gen_params["repetition_penalty"] = args.repetition_penalty
    gen_params["temperature"] = args.temperature
    gen_params["top_p"] = args.top_p

    # 1) Load the fake true dataset

    # 2) Select the prefixes from the fake samples

    # 3) Generate the responses with the generator using the prefixes

    # process true dataset
    true_dataset = process_true_dataset(true_dataset, args.fake_dataset_size, args.seed)

    # generate fake dataset
    #fake_dataset = generate_fake_dataset(true_dataset, args.fake_dataset_size, generator, gen_tokenizer, args.max_nb_tokens_input, args.max_new_tokens, args.seed, args.batch_size, use_chat_template=use_chat_template, template_type=template_type)
    fake_dataset = generate_fake_dataset(true_dataset, args.fake_dataset_size, generator, gen_tokenizer, args.max_nb_tokens_input, args.max_new_tokens, args.seed,
                                          args.batch_size, use_chat_template=use_chat_template, template_type=template_type, load_from_cache=args.load_from_cache, prompt=args.prompt)
    
    #true_dataset.save_to_disk(f"true_dataset_{args.experiment_name}")

    # process fake dataset
    fake_dataset = process_fake_dataset(fake_dataset, gen_tokenizer, args.max_response_length)
    #fake_dataset.save_to_disk(f"fake_dataset_{args.experiment_name}")

    # merge true and fake dataset
    merged_dataset = merge_true_fake_dataset(true_dataset, fake_dataset, args.seed)

    # format merged dataset into a template
    merged_dataset = format_merged_dataset(merged_dataset, use_chat_template, args.max_response_length)

    # group pairs of true and fake responses two by two so that they are in the same batch and in the same split
    merged_dataset = regroup_pairs(merged_dataset)

    # balance the dataset again
    #merged_dataset = balance_dataset(merged_dataset["train"], create_train=True)


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






    

