import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json
from time import gmtime, strftime
import logging
import sys
import os
from datasets import load_from_disk
from abc import ABC, abstractmethod
import pandas as pd

from .experiment_pipeline import ExperimentPipeline
from .pipeline_utils import *

class CreateDatasetPipeline(ExperimentPipeline):
    def __init__(self, dataset_loader, attack, device, experiment_path, batch_size=1, skip_cache=False):
        self.dataset_loader = dataset_loader
        self.attack = attack
        self.device = device
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        
        # if set to true, overwrite the cached datasets
        self.skip_cache = skip_cache
        
        # setup log
        log_path = f"{experiment_path}/log"
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=log_path)
        
        
    def create_logger(self):
        if log_path is None:
            if self.experiment_path is None:
                raise ValueError("Experiment path not set")
            log_path = self.experiment_path
        
        # create log file
        with open(f"{log_path}/log.txt", "w") as f:
            f.write("")

        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{log_path}/log.txt")
        self.log = log
        
    def create_experiment_dataset(self, dataset_name):
        ### CREATE THE (ADVERSRIAL) DATASET AND SAVE IT ###
        
        # Load the base dataset
        dataset = self.dataset_loader.load_data()
        
        # We only use the test data split here
        #dataset = dataset["test"]
                
        # Generate adversarial examples
        true_articles = dataset.filter(lambda x: x["label"] == 0)

        data_splits = ["train", "eval", "test"]
        for split in data_splits:
            split_data = dataset[split]
            true_articles = split_data.filter(lambda x: x["label"] == 0)
            true_articles_prefixes = true_articles["prefix"][:]
            fake_articles = self.attack.generate_adversarial_text(true_articles_prefixes, batch_size=self.batch_size)
            
            # Fuse true and fake articles by filling samples in dataset with label = 1
            def fuse_fake_true_articles(sample):
                if sample["label"] == 1:
                    sample["text"] = fake_articles.pop(0)
                return sample
            
            split_data = split_data.map(fuse_fake_true_articles)
            dataset[split] = split_data
            
        # Save the dataset using a specific naming convention
        dataset.save_to_disk(f"data/generated_datasets/{dataset_name}")
        
        # save also to json for each split
        for split in data_splits:
            split_data = dataset[split]
            
            # transfor to pandas dataframe
            df = pd.DataFrame(split_data)
            df.to_json(f"data/generated_datasets/{dataset_name}_{split}.json", force_ascii=False, indent=4)
        
        return dataset
        
    def run_pipeline(self):
        log = self.log
        dataset_name = self.dataset_loader.dataset_name

        # check if the dataset has already been generated for the attack and base dataset
        base_dataset_name = self.dataset_loader.dataset_name
        attack_name  = self.attack.attack_name
        gen_name = self.attack.gen_name
        #use_watermarking = self.watermarking_scheme is not None
        watermarking_scheme = self.attack.watermarking_scheme
        
        dataset_size = self.dataset_loader.dataset_size
        dataset_name = f"{base_dataset_name}_{gen_name}_{attack_name}_{dataset_size}"
        
        if watermarking_scheme is not None:
            log.info(f"Using watermarking scheme {self.attack.watermarking_scheme_name}")
            dataset_name += f"_{self.attack.watermarking_scheme_name}"
            
        if not self.skip_cache and os.path.isdir(f"data/generated_datasets/{dataset_name}"):
            log.info(f"Dataset {dataset_name} already exists!")
        else:
            log.info(f"Dataset {dataset_name} does not exist, creating it")
            log.info("Generating the dataset...")
            dataset = self.create_experiment_dataset(dataset_name)
            
        return