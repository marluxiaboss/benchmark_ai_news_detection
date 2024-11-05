import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json
from time import gmtime, strftime
import logging
import sys
import os
from datasets import load_from_disk, Dataset
from abc import ABC, abstractmethod
import pandas as pd

from ..dataset_loader import FakeTruePairsDataLoader
from ..generation import ArticleGenerator
from .experiment_pipeline import ExperimentPipeline
from .pipeline_utils import create_logger_file


class CreateDatasetPipeline(ExperimentPipeline):
    def __init__(
        self,
        cfg: dict,
        dataset_loader: FakeTruePairsDataLoader,
        attack: ArticleGenerator,
        experiment_path: str,
        batch_size: int = 1,
        skip_cache: bool = False,
        skip_train_split: bool = False,
    ):
        """
        Pipeline for creating a dataset of fake and true articles.

        Parameters:
        ----------
            cfg: dict
                The configuration of the experiment.
            dataset_loader: FakeTruePairsDataLoader
                The dataset loader to use for loading the dataset.
            attack: ArticleGenerator
                The generator to use for generating the fake articles, potentially adversarial.
            experiment_path: str
                The path to the experiment used for saving the dataset.
            batch_size: int
                The batch size to use for generation. Default is 1.
            skip_cache: bool
                If set to True, overwrite the saved dataset. Default is False.
        """

        self.cfg = cfg
        self.dataset_loader = dataset_loader
        self.attack = attack
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        self.generator_name = cfg.generation.generator_name

        self.experiment_name = f"{self.generator_name}_{cfg.generation.experiment_name}"

        # if set to true, overwrite the cached datasets
        self.skip_cache = skip_cache

        # if set to true, not generate fake articles for the train split
        self.skip_train_split = skip_train_split

        # check that folder at experiment path exists, if not create it
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if not os.path.exists(f"{experiment_path}/log"):
            os.makedirs(f"{experiment_path}/log")

        # setup log
        log_path = f"{experiment_path}/log/log_{self.experiment_name}.txt"
        self.log = create_logger_file(log_path)

    def create_experiment_dataset(self) -> Dataset:
        """
        Create the fake true dataset for the experiment by generating fake articles using the generator.

        Returns:
        -------
            dataset: Dataset
                The generated fake true dataset.
        """
        ### CREATE THE (ADVERSARIAL) DATASET AND SAVE IT ###

        # Load the base dataset
        dataset = self.dataset_loader.load_data()

        def fuse_fake_true_articles(sample, fake_articles):
            if sample["label"] == 1:
                sample["text"] = fake_articles.pop(0)

            return sample

        # generate fake articles for each split
        if self.skip_train_split:
            data_splits = ["eval", "test"]
        else:
            data_splits = ["train", "eval", "test"]

        for split in data_splits:
            split_data = dataset[split]

            # if self.skip_train_split and split == "train":
            fake_articles = split_data.filter(lambda x: x["label"] == 1)
            fake_articles_prefixes = fake_articles["prefix"][:]
            fake_articles = self.attack.generate_adversarial_text(
                fake_articles_prefixes, batch_size=self.batch_size
            )

            # replace the empty text field for label 1 samples (AI) with the generated fake articles
            split_data = split_data.map(lambda x: fuse_fake_true_articles(x, fake_articles))

            # remove samples with empty text
            split_data = split_data.filter(lambda x: len(x["text"]) > 0)

            dataset[split] = split_data

        # normalize data in cfg to save it safely, i.e. convert to string
        for key, value in self.cfg.items():
            if not isinstance(value, (str, int, float, bool)):
                self.cfg[key] = str(value)

        # add generation and watermark config as fields in the dataset to identify how it was generated
        dataset = dataset.map(
            lambda x: {
                "generation_config": self.cfg.generation,
                "watermark_config": self.cfg.watermark,
            }
        )

        # Save the dataset using a specific naming convention
        # dataset.save_to_disk(f"data/generated_datasets/{dataset_name}")
        dataset.save_to_disk(f"{self.experiment_path}/{self.experiment_name}")

        # save also to json for each split, including train even if we skipped it
        data_splits = ["train", "eval", "test"]

        for split in data_splits:
            split_data = dataset[split]

            # transfor to pandas dataframe
            df = pd.DataFrame(split_data)
            df.to_json(
                f"{self.experiment_path}/{self.experiment_name}_{split}.json",
                force_ascii=False,
                indent=4,
            )

        return dataset

    def run_pipeline(self):
        """
        Main function of the class, runs the pipeline.
        Creates the dataset for the experiment and saves it.
        """
        log = self.log

        dataset_path = f"{self.experiment_path}/{self.experiment_name}"
        if not self.skip_cache and os.path.isdir(dataset_path):
            log.info(f"Dataset at {dataset_path} already exists!")
        else:
            log.info(f"Dataset at {dataset_path} does not exist, creating it")
            log.info("Generating the dataset...")
            self.create_experiment_dataset()

        # save the parameter of the generation at the very end so that we only have them if the rest succeded
        log.info("Parameters for the generation:")
        log.info(self.cfg)

        return
