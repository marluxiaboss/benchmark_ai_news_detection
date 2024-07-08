from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, disable_progress_bar, enable_progress_bar
import pandas as pd
import numpy as np


### Helper functions
def create_train_from_dataset(dataset: Dataset) -> DatasetDict:
    """
    Create a train split from a dataset. We go from Dataset to DatasetDict.
    
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

def filter_duplicates(dataset: Dataset, text_field: str) -> Dataset:
    """
    Filter duplicates in the dataset based on the text_field.
    
    Parameters:
    dataset: Dataset
        The dataset to filter duplicates from
    text_field: str
        The field to use for filtering duplicates
    
    Returns:
        Dataset
            The dataset without duplicates
        
    """
    
    # check duplicates in the text_field
    dataset_df = pd.DataFrame(dataset)
    len_before_discard = dataset_df.shape[0]
    
    dataset_df = dataset_df.drop_duplicates(subset=[text_field])
    len_after_discard = dataset_df.shape[0]
    print(f"Percent of data discarded after removing duplicate {text_field}: {100*(1 - len_after_discard/len_before_discard):.2f}%")
    
    return Dataset.from_pandas(dataset_df)

def create_splits(dataset: Dataset, train_size: float, eval_size: float, test_size: float) -> DatasetDict:
    """
    Create train, eval and test splits from a dataset.
    
    Parameters:
    dataset: Dataset
        The dataset to create the splits from
    train_size: float
        The size of the train split
    eval_size: float
        The size of the eval split
    test_size: float
        The size of the test split
        
    Returns:
        DatasetDict
            The dataset with the train, eval and test splits
        
    """
    
    train_size = len(dataset["train"])
    eval_size = int(train_size * eval_size)
    test_size = int(train_size * test_size)

    dataset = DatasetDict({
    'train': dataset["train"].select(range(train_size - eval_size - test_size)),
    'eval': dataset["train"].select(range(train_size - eval_size - test_size, train_size - test_size)),
    'test': dataset["train"].select(range(train_size - test_size, train_size))})

    print("Train size:", len(dataset['train']))
    print("Eval size:", len(dataset['eval']))
    print("Test size:", len(dataset['test']))

    return dataset