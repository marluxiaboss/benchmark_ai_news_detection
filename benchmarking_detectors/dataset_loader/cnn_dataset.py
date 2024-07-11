from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, disable_progress_bar, enable_progress_bar
import pandas as pd
import numpy as np

from .dataset_loader_utils import *
from .fake_true_dataset import FakeTruePairsDataLoader
    
    
class CNNDataLoader(FakeTruePairsDataLoader):
    
    def __init__(self, dataset_size: int, hf_dataset_path:str ="abisee/cnn_dailymail", text_field: str="article",
                prefix_size: int=10, max_sample_len: int=500,
                train_fraction: float=0.8, eval_fraction: float=0.1, test_fraction: float=0.1,
                seed: int=42) -> None:
        
        """
        Class used to load the cnn_dailymail from Huggingface and create the fake-true pairs dataset format used for the benchmarking.
        
        Parameters:
            dataset_size: int
                The number of samples to load from the dataset. Note: we will have at the end 2*dataset_size samples in the dataset
                since we create a fake sample for each true sample (maybe less due to filtering duplicates).
            hf_dataset_path: str
                The path to the Huggingface dataset. Default is "abisee/cnn_dailymail".
            text_field: str
                The name of the field containing the text of interest. Default is "article".
            prefix_size: int
                The number of words to use as prefix. Default is 10.
            max_sample_len: int
                The maximum length of the text in characters. Default is 500.
            seed: int
                The seed to use for reproducibility. Default is 42.
        """
        self.dataset_size = dataset_size
        self.text_field = text_field
        self.prefix_size = prefix_size
        self.hf_dataset_path = hf_dataset_path
        self.max_sample_len = max_sample_len
        self.seed = seed
        
        # size of the splits
        self.train_fraction = train_fraction
        self.eval_fraction = eval_fraction
        self.test_fraction = test_fraction
        
        self.dataset_name = "cnn_dailymail"
    
    def regroup_pairs(self, dataset_true: Dataset, dataset_fake: Dataset) -> Dataset:
        """
        Merge the two datasets by regrouping the pairs of human and AI samples with the same prefix.
        The first element of the pair is chosen randomly among the true and fake samples.
        
        Parameters:
            dataset_true: Dataset
                The dataset containing the true samples.
            dataset_fake: Dataset
                The dataset containing the fake samples.
        
        Returns:
            Dataset
                The merged dataset.
        """

        merged_dataset = []
        for i in range(len(dataset_true)):
            
            # choose randomly the first element of the pair
            random_first = np.random.choice([0, 1])
            
            if random_first == 0:
                merged_dataset.append({"label": dataset_true[i]["label"], self.text_field: dataset_true[i][self.text_field], "prefix": dataset_true[i]["prefix"]})
                merged_dataset.append({"label": dataset_fake[i]["label"], self.text_field: dataset_fake[i][self.text_field], "prefix": dataset_fake[i]["prefix"]})
                
            else:
                merged_dataset.append({"label": dataset_fake[i]["label"], self.text_field: dataset_fake[i][self.text_field], "prefix": dataset_fake[i]["prefix"]})
                merged_dataset.append({"label": dataset_true[i]["label"], self.text_field: dataset_true[i][self.text_field], "prefix": dataset_true[i]["prefix"]})
                
        dataset = Dataset.from_pandas(pd.DataFrame(merged_dataset))
        
        return dataset
    
    def clean_dataset(self, dataset: Dataset) -> Dataset:
        """
        Clean the dataset by removing bloat from the text field.
        
        Parameters:
            dataset: Dataset
                The dataset to clean.
        
        Returns:
            Dataset
                The cleaned dataset.
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
                
            # Heuristic for cases like "By . Charlie Scott . ..."
            if filtered_text.startswith("By ."):
                filtered_text_list = filtered_text.split(".")[2:]
                filtered_text = ".".join(filtered_text_list).strip()
    
            return {"article": filtered_text}
        
        dataset = dataset.map(remove_bloat)
        
        return dataset
    
    def process_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Main method to process the dataset called by load_data.
        
        Parameters:
            dataset: DatasetDict
                The dataset to process.
        
        Returns:
            DatasetDict
                The processed dataset.
        """
        
        dataset = self.clean_dataset(dataset)    
        
        # only take max_sample_len characters of the text field
        dataset = dataset.map(lambda x: {self.text_field: x[self.text_field][:self.max_sample_len]})
        
        dataset = filter_duplicates(dataset, self.text_field) 
        
        # filter out samples with text > max_sample_len
        nb_samples_before_filter = len(dataset)
        dataset = dataset.filter(lambda x: len(x[self.text_field]) <= self.max_sample_len)
        nb_samples_after_filter = len(dataset)
        print(f"Filtered out {nb_samples_before_filter - nb_samples_after_filter} samples with text > {self.max_sample_len} characters.")
        
        # create label 0 (human) and create empty texts with label 1 (AI)
        dataset = dataset.map(lambda x: {"label": 0, self.text_field: x[self.text_field]})
        
        # create prefix column, which contains the first self.prefix_size words of the human text
        dataset = dataset.map(lambda x: {"prefix": " ".join(x[self.text_field].split()[:self.prefix_size])})
        
        # copy all human samples into AI samples with label 1 and empty text
        dataset_fake = dataset.map(lambda x: {"label": 1, self.text_field: "", "prefix": x["prefix"]})
        dataset = self.regroup_pairs(dataset, dataset_fake)
        
        # rename text field to more generic column name text, if text_field is not "text" yet
        if self.text_field != "text":
            dataset = dataset.rename_column(self.text_field, "text")
    
        dataset = create_train_from_dataset(dataset)
        return dataset

    
    def load_data(self) -> DatasetDict:
        """
        Function that we call to load the dataset.
        
        Returns:
            DatasetDict
                The processed dataset.
        """
        
        
        # we take the train split but we'll split later into train, val, test
        dataset_base = load_dataset(self.hf_dataset_path, "3.0.0")["train"]
        
        # select the first dataset_size samples
        dataset_base = dataset_base.shuffle(self.seed)
        dataset_base = dataset_base.select(range(self.dataset_size))
        
        # only keep the text field
        cols_to_remove = [col for col in dataset_base.column_names if col != self.text_field]
        dataset_base = dataset_base.remove_columns(cols_to_remove)

        processed_dataset = self.process_data(dataset_base)
        
        # split into train, val, test
        train_split_size_percent = self.train_fraction
        eval_split_size_percent = self.eval_fraction
        test_split_size_percent = self.test_fraction
        processed_dataset = create_splits(processed_dataset, train_split_size_percent, eval_split_size_percent, test_split_size_percent)
        
        return processed_dataset