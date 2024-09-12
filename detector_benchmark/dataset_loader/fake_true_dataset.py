from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    disable_progress_bar,
    enable_progress_bar,
    load_from_disk,
)
import pandas as pd
import numpy as np

from .dataset_loader_utils import filter_duplicates


# TODO: need to heavily modify it to make it general. Need to see how we integrate different datasets other than cnn_dailymail
class FakeTruePairsDataLoader:
    def __init__(
        self,
        dataset_size,
        dataset_path,
        text_field,
        prefix_size=10,
        max_sample_len=500,
        load_local=True,
        dataset_name="",
        seed=42,
    ) -> None:
        """
        Parameters:
        ----------
            dataset_size: int
                The size of the dataset
            dataset_path: str
                The path to the dataset
            text_field: str
                The field to use for filtering duplicates
        """
        self.dataset_size = dataset_size
        self.text_field = text_field
        self.prefix_size = prefix_size
        self.dataset_path = dataset_path
        self.max_sample_len = max_sample_len
        self.dataset_name = dataset_name
        self.load_local = load_local
        self.seed = seed

    def regroup_pairs(self, dataset_true, dataset_fake) -> Dataset:
        """
        Merge the two datasets by regrouping the pairs of human and AI samples with the same prefix.
        The first element of the pair is chosen randomly.

        Parameters:
        ----------
            dataset_true: Dataset
                The dataset containing the true samples
        """
        # merge the two datasets by regrouping the pairs of human and AI samples with the same prefix
        # the first element of the pair is chosen randomly
        merged_dataset = []
        for i in range(len(dataset_true)):

            # choose randomly the first element of the pair
            random_first = np.random.choice([0, 1])

            if random_first == 0:
                merged_dataset.append(
                    {
                        "label": dataset_true[i]["label"],
                        self.text_field: dataset_true[i][self.text_field],
                        "prefix": dataset_true[i]["prefix"],
                    }
                )
                merged_dataset.append(
                    {
                        "label": dataset_fake[i]["label"],
                        self.text_field: dataset_fake[i][self.text_field],
                        "prefix": dataset_fake[i]["prefix"],
                    }
                )

            else:
                merged_dataset.append(
                    {
                        "label": dataset_fake[i]["label"],
                        self.text_field: dataset_fake[i][self.text_field],
                        "prefix": dataset_fake[i]["prefix"],
                    }
                )
                merged_dataset.append(
                    {
                        "label": dataset_true[i]["label"],
                        self.text_field: dataset_true[i][self.text_field],
                        "prefix": dataset_true[i]["prefix"],
                    }
                )

        dataset = Dataset.from_pandas(pd.DataFrame(merged_dataset))

        return dataset

    def process_data(self, dataset: DatasetDict) -> DatasetDict:
        dataset = filter_duplicates(dataset, self.text_field)

        # create prefix column, which contains the first self.prefix_size words of the human text
        dataset = dataset.map(
            lambda x: {"prefix": " ".join(x[self.text_field].split()[: self.prefix_size])}
        )

        # regroup pairs in case not already done
        dataset_true = dataset.filter(lambda x: x["label"] == 0)
        dataset_fake = dataset.filter(lambda x: x["label"] == 1)
        dataset = self.regroup_pairs(dataset_true, dataset_fake)

        # rename text field to more generic column name text, if text_field is not "text" yet
        if self.text_field != "text":
            dataset = dataset.rename_column(self.text_field, "text")

        return dataset

    def load_data(self) -> DatasetDict:

        if self.load_local:
            dataset = load_from_disk(self.dataset_path)
        else:
            dataset = load_dataset(self.dataset_path)

        dataset = self.process_data(dataset)
        return dataset
