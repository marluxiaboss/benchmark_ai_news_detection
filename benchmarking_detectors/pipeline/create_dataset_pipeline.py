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
    def __init__(self, cfg, dataset_loader, attack, device, experiment_path, batch_size=1, skip_cache=False):
        self.cfg = cfg
        self.dataset_loader = dataset_loader
        self.attack = attack
        self.device = device
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        self.generator_name = cfg.generation.generator_name
        
        self.experiment_name = f"{self.generator_name}_{cfg.generation.experiment_name}"
        
        # if set to true, overwrite the cached datasets
        self.skip_cache = skip_cache
        
        # check that folder at experiment path exists, if not create it
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if not os.path.exists(f"{experiment_path}/log"):
            os.makedirs(f"{experiment_path}/log")

        # setup log
        log_path = f"{experiment_path}/log/log_{self.experiment_name}.txt"
        self.log = self.create_logger_file(log_path)
        
    def create_logger_file(self, log_path):
        if log_path is None:
            if self.experiment_path is None:
                raise ValueError("Experiment path not set")
            log_path = self.experiment_path
        
        # create log file
        with open(f"{log_path}/log.txt", "w") as f:
            f.write("")

        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{log_path}/log.txt")
        
        return log
        
    def create_experiment_dataset(self):
        ### CREATE THE (ADVERSRIAL) DATASET AND SAVE IT ###
        
        # Load the base dataset
        dataset = self.dataset_loader.load_data()

        def fuse_fake_true_articles(sample, fake_articles):
            if sample["label"] == 1:
                sample["text"] = fake_articles.pop(0)
                
            return sample  
               
        # generate fake articles for each split
        data_splits = ["train", "eval", "test"]
        for split in data_splits:
            split_data = dataset[split]
            fake_articles = split_data.filter(lambda x: x["label"] == 1)
            fake_articles_prefixes = fake_articles["prefix"][:]
            fake_articles = self.attack.generate_adversarial_text(fake_articles_prefixes, batch_size=self.batch_size)
            
            # replace the empty text field for label 1 samples (AI) with the generated fake articles
            split_data = split_data.map(lambda x: fuse_fake_true_articles(x, fake_articles))
            
            # remove samples with empty text
            split_data = split_data.filter(lambda x: len(x["text"]) > 0)
            
            dataset[split] = split_data
            
        # add generation and watermark config as fields in the dataset to identify how it was generated
        dataset = dataset.map(lambda x: {"generation_config": self.cfg.generation, "watermark_config": self.cfg.watermark})    
        
        # Save the dataset using a specific naming convention
        #dataset.save_to_disk(f"data/generated_datasets/{dataset_name}")
        dataset.save_to_disk(f"{self.experiment_path}/{self.experiment_name}")
        
        # save also to json for each split
        for split in data_splits:
            split_data = dataset[split]
            
            # transfor to pandas dataframe
            df = pd.DataFrame(split_data)
            df.to_json(f"{self.experiment_path}/{self.experiment_name}_{split}.json", force_ascii=False, indent=4)
        
        return dataset
        
    def run_pipeline(self):
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