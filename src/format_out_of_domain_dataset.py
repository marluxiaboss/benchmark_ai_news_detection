from datasets import concatenate_datasets, load_from_disk, DatasetDict, Dataset, load_dataset
import argparse
import os
import json
import pandas as pd

if __name__ == "__main__":
    
    # e.g. python src/format_out_of_domain_dataset.py --save_path=fake_true_datasets/xsum_true_only_test --original_dataset_path=EdinburghNLP/xsum --dataset_name=xsum_only_true --take_samples=10000 --orig_dataset_type=huggingface
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to save the dataset", required=True)
    parser.add_argument("--original_dataset_path", type=str, help="Path to the original json dataset", required=True)
    parser.add_argument("--orig_dataset_type", type=str, help="Type of the original dataset (between huggingface and json)", required=True)
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", required=False, default=None)
    args = parser.parse_args()


    # load original json dataset, make it into a dataframe, transform into correct shape, and save it
    match args.orig_dataset_type:
        
        case "huggingface":
            dataset = load_dataset(args.original_dataset_path)
            dataset_test = dataset["test"]
            dataset_df = dataset_test.to_pandas()
            
        case "json":
            orig_json_dataset_path = args.original_dataset_path
            dataset_df = pd.read_json(orig_json_dataset_path)
            
        case _:
            raise ValueError("orig_dataset_type must be either huggingface or json")
    
    if args.take_samples is not None:
        
        if not dataset_df.shape[0] < args.take_samples:
            dataset_df = dataset_df.sample(args.take_samples, random_state=42)
        else:
            print("Warning: take_samples is larger than the dataset size. Taking the whole dataset.")

    fake_true_samples = []

    
    # for https://github.com/baoguangsheng/fast-detect-gpt/blob/main/exp_gpt3to4/data/pubmed_gpt-4.raw_data.json
    if args.dataset_name == "pubmed_fake_true":
        for index, row in dataset_df.iterrows():
            fake_true_samples.append({"text": row["original"], "label": 0})
            fake_true_samples.append({"text": row["sampled"], "label": 1})
            
    if args.dataset_name == "pubmed_only_true":
        for index, row in dataset_df.iterrows():
            print("row: original", row["original"])
            fake_true_samples.append({"text": row["original"], "label": 0})
            
    # for https://huggingface.co/datasets/EdinburghNLP/xsum
    if args.dataset_name == "xsum_only_true":
        for index, row in dataset_df.iterrows():
            fake_true_samples.append({"text": row["document"], "label": 0})
            
    if args.dataset_name == "cnn_dailymail_only_true":
        for index, row in dataset_df.iterrows():
            if row["label"] == 0:
                fake_true_samples.append({"text": row["text"][0], "label": 0})

    fake_true_dataset_df = pd.DataFrame(fake_true_samples)
    
    # keep only the first 500 characters of the text
    fake_true_dataset_df["text"] = fake_true_dataset_df["text"].apply(lambda x: x[:500])
    
    
    size_before = len(fake_true_dataset_df)
    # fiter out samples with < 500 characters
    fake_true_dataset_df = fake_true_dataset_df[fake_true_dataset_df["text"].apply(lambda x: len(x) >= 500)]
    
    print(f"Filtered out percentage of samples: {100*(size_before - len(fake_true_dataset_df))/size_before}%")
    print("Size of the dataset: ", len(fake_true_dataset_df))
    
    # print first rows of the dataset to verify the validity
    print(fake_true_dataset_df.head())
    
    # transform back the pandas dataframe into a huggingface dataset and save it
    fake_true_dataset = DatasetDict({"test": Dataset.from_pandas(fake_true_dataset_df)})
    fake_true_dataset.save_to_disk(f"{args.save_path}")

    # save the json version of the dataset
    fake_true_dataset_df.to_json(f"{args.save_path}.json", force_ascii=False, indent=4)