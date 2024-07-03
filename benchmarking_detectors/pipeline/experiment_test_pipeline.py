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

class ExperimentTestPipeline(ExperimentPipeline):
    def __init__(self, dataset_loader, attack, detector, device, experiment_path, batch_size=1, skip_cache=False):
        self.dataset_loader = dataset_loader
        self.attack = attack
        self.device = device
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        #self.watermarking_scheme = watermarking_scheme
        
        # if set to true, overwrite the cached datasets
        self.skip_cache = skip_cache
        
        # setup log
        log_path = f"{experiment_path}/log"
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=log_path)
        
        # set the detector
        self.detector = detector
        
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
        dataset = dataset["test"]
                
        # Generate adversarial examples
        true_articles = dataset.filter(lambda x: x["label"] == 0)
        true_articles_prefixes = true_articles["prefix"][:]
        fake_articles = self.attack.generate_adversarial_text(true_articles_prefixes, batch_size=self.batch_size)
        
        # Fuse true and fake articles by filling samples in dataset with label = 1
        def fuse_fake_true_articles(sample):
            if sample["label"] == 1:
                sample["text"] = fake_articles.pop(0)
            return sample
        
        dataset = dataset.map(fuse_fake_true_articles)
                
        # Save the dataset using a specific naming convention
        dataset.save_to_disk(f"data/generated_datasets/{dataset_name}")
        
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
            log.info(f"Dataset {dataset_name} already exists, loading it")
            dataset = load_from_disk(f"data/generated_datasets/{dataset_name}")
        else:
            log.info(f"Dataset {dataset_name} does not exist, creating it")
            log.info("Generating the dataset...")
            dataset = self.create_experiment_dataset(dataset_name)
                
        ### TEST THE DETECTOR ###
        
        # clean the memory
        del self.dataset_loader
        del self.attack
        
        fake_true_articles = dataset["text"][:]
        
        log.info("Classifying the articles...")
        preds, logits, preds_at_threshold = self.detector.detect(fake_true_articles, batch_size=self.batch_size)
        labels = dataset["label"]
        
        preds = np.array(preds)
        logits = np.array(logits)
        labels = np.array(labels)
        preds_at_threshold = np.array(preds_at_threshold)
        
        # TODO: better to handle this in a helper
        # compute metrics
        nb_pos_labels = np.sum(dataset["label"] == 1)
        nb_neg_labels = np.sum(dataset["label"] == 0)
        
        if nb_pos_labels == 0 or nb_neg_labels == 0:
            #log.info("Only one class in the dataset, cannot compute roc_auc")
            roc_auc = 0
            fpr = np.zeros(1)
            tpr = np.zeros(1)
            thresholds = np.zeros(1)
        else:
            roc_auc = roc_auc_score(labels, logits)
            fpr, tpr, thresholds = roc_curve(labels, logits)
        
        results = compute_bootstrap_metrics(preds, labels)
        
        log.info("Test metrics:")
        for key, value in results.items():
            log.info(f"{key}: {value}")
            
        # also log the roc_auc and the fpr, tpr, thresholds
        log.info(f"roc_auc: {roc_auc}")
        log.info(f"fpr: {fpr}")
        log.info(f"tpr: {tpr}")
        log.info(f"thresholds: {thresholds}")
        
        results["roc_auc"] = roc_auc
        results["fpr_at_thresholds"] = fpr.tolist()
        results["tpr_at_thresholds"] = tpr.tolist()
        results["thresholds"] = thresholds.tolist()
    
        # Compute metrics at the given threshold
        classifier_threshold = self.detector.detection_threshold
        results_at_threshold = compute_bootstrap_metrics(preds_at_threshold, labels)
        log.info("Test metrics at specific given threshold:")
        
        for key, value in results_at_threshold.items():
            log.info(f"{key}: {value}")
            
        # add them to results dict as f"{key}_at_given_threshold"
        results["given_threshold"] = classifier_threshold
        for key, value in results_at_threshold.items():
            results[f"{key}_at_given_threshold"] = value
                
        # save results
        experiment_path = self.experiment_path

        if not os.path.isdir(f"{experiment_path}/test"):
            os.makedirs(f"{experiment_path}/test")
        
        json_res_file_path_base = f"{experiment_path}/test/test_metrics_{dataset_name}.json"
                
        with open(json_res_file_path_base, "w") as f:
            f.write(json.dumps(results, indent=4))