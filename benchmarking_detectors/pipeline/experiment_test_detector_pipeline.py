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

class ExperimentTestDetectorPipeline(ExperimentPipeline):
    def __init__(self, cfg, detector, device, experiment_path, dataset_experiment_path, batch_size=1):
        self.cfg = cfg
        self.device = device
        self.experiment_path = experiment_path
        self.batch_size = batch_size
        self.generator_name = cfg.generation.generator_name
        self.detector_name = cfg.detection.detector_name
        
        self.dataset_experiment_path = dataset_experiment_path
        self.dataset_experiment_name = cfg.generation.experiment_name
        self.experiment_name = f"{self.detector_name}_{cfg.detection.experiment_name}"
        
        # check that folder at experiment path exists, if not create it
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if not os.path.exists(f"{experiment_path}/log"):
            os.makedirs(f"{experiment_path}/log")

        # setup log
        log_path = f"{experiment_path}/log"
        self.log = self.create_logger_file(log_path)
        
        # set the detector
        self.detector = detector
        
    def create_logger_file(self, log_path):
        
        # create log file
        with open(f"{log_path}/log.txt", "w") as f:
            f.write("")

        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{log_path}/log.txt")
        return log
        
    def evaluate_detector(self, preds, logits, preds_at_threshold, labels, dataset):               
        
        log = self.log
        dataset_name = self.dataset_experiment_name
        
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
            
            
    def run_pipeline(self):
        log = self.log
        
        # See if the dataset exists. This pipeline assumes that the dataset already exists
        dataset_path = f"{self.dataset_experiment_path}/{self.cfg.generation.generator_name}_{self.dataset_experiment_name}"
        if os.path.isdir(dataset_path):
            log.info(f"Dataset at {dataset_path} exists, loading it")
            dataset = load_from_disk(dataset_path)
        else:
            raise ValueError(f"Dataset at {dataset_path} does not exist, please create it first with create_dataset.py")
                
        ### TEST THE DETECTOR ###
        
        # here we only test on the test set!
        dataset = dataset["test"]
        fake_true_articles = dataset["text"][:]
        
        log.info("Classifying the articles...")
        preds, logits, preds_at_threshold = self.detector.detect(fake_true_articles, batch_size=self.batch_size)
        labels = dataset["label"]
        self.evaluate_detector(preds, logits, preds_at_threshold, labels, dataset)
