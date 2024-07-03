from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, disable_progress_bar, enable_progress_bar
import pandas as pd
import numpy as np

from .dataset_loader_utils import *
from .fake_true_dataset import FakeTruePairsDataLoader
    
    
class CNNDataLoader(FakeTruePairsDataLoader):
    
    def __init__(self, dataset_size, hf_dataset_path="abisee/cnn_dailymail", text_field="article", prefix_size=10,
                 max_sample_len=500, seed=42) -> None:
        self.dataset_size = dataset_size
        #self.test_size = self.dataset_size * 0.1
        self.text_field = text_field
        self.prefix_size = prefix_size
        self.hf_dataset_path = hf_dataset_path
        self.max_sample_len = max_sample_len
        self.seed = seed
        
        self.dataset_name = "cnn_dailymail"
    
    def regroup_pairs(self, dataset_true, dataset_fake) -> Dataset:
        # merge the two datasets by regrouping the pairs of human and AI samples with the same prefix
        # the first element of the pair is chosen randomly
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
        
        dataset = dataset.map(remove_bloat)
        
        return dataset
    
    def process_data(self, dataset: DatasetDict) -> DatasetDict:
        
        dataset = self.clean_dataset(dataset)    
        
        # only take max_sample_len characters of the text field
        dataset = dataset.map(lambda x: {self.text_field: x[self.text_field][:self.max_sample_len]})
        
        dataset = filter_duplicates(dataset, self.text_field) 
        
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
        train_split_size_percent = 0.8
        eval_split_size_percent = 0.1
        test_split_size_percent = 0.1
        processed_dataset = create_splits(processed_dataset, train_split_size_percent, eval_split_size_percent, test_split_size_percent)
        
        
        return processed_dataset