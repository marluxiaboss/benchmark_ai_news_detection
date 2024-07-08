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



def compute_bootstrap_metrics(data, labels, n_bootstrap=1000, flip_labels=False):

    # compute false postives, false negatives, true positives, true negatives using bootstrap
    nb_false_positives = np.zeros(n_bootstrap)
    nb_false_negatives = np.zeros(n_bootstrap)
    nb_true_positives = np.zeros(n_bootstrap)
    nb_true_negatives = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(range(len(data)), len(data), replace=True)
        nb_false_positives[i] = np.sum((data[bootstrap_sample] == 1) & (labels[bootstrap_sample] == 0))
        nb_false_negatives[i] = np.sum((data[bootstrap_sample] == 0) & (labels[bootstrap_sample] == 1))
        nb_true_positives[i] = np.sum((data[bootstrap_sample] == 1) & (labels[bootstrap_sample] == 1))
        nb_true_negatives[i] = np.sum((data[bootstrap_sample] == 0) & (labels[bootstrap_sample] == 0))
    
    metrics = ["accuracy", "precision", "recall", "f1_score", "fp_rate", "tp_rate"]
    avg_metrics = {}
    std_metrics = {}
    for metric in metrics:
        metric_results = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            nb_false_positives_i = nb_false_positives[i]
            nb_false_negatives_i = nb_false_negatives[i]
            nb_true_positives_i = nb_true_positives[i]
            nb_true_negatives_i = nb_true_negatives[i]
            
            if flip_labels:
                nb_false_positives_i = nb_false_negatives[i]
                nb_false_negatives_i = nb_false_positives[i]
                nb_true_positives_i = nb_true_negatives[i]
                nb_true_negatives_i = nb_true_positives[i]
            
            # we need to test cases where the denominator is 0 because there might dataset with only 0 labels or 1 labels
            match metric:
                case "accuracy":
                    if len(data) == 0:
                        metric_results[i] = 0
                    else:
                        metric_results[i] = (nb_true_positives_i + nb_true_negatives_i) / len(data)
                    
                case "precision":
                    if (nb_true_positives_i + nb_false_positives_i == 0):
                        metric_results[i] = 0
                    else:
                        metric_results[i] = nb_true_positives_i / (nb_true_positives_i + nb_false_positives_i)
                        
                case "recall":
                    if (nb_true_positives_i + nb_false_negatives_i == 0):
                        metric_results[i] = 0
                    else:
                        metric_results[i] = nb_true_positives_i / (nb_true_positives_i + nb_false_negatives_i)
                case "f1_score":
                    if (2 * nb_true_positives_i + nb_false_positives_i + nb_false_negatives_i) == 0:
                        metric_results[i] = 0
                    else:
                        metric_results[i] = 2 * nb_true_positives_i / (2 * nb_true_positives_i + nb_false_positives_i + nb_false_negatives_i)
                case "fp_rate":
                    if  (nb_false_positives_i + nb_true_negatives_i) == 0:
                        metric_results[i] = 0
                    else:
                        metric_results[i] = nb_false_positives_i / (nb_false_positives_i + nb_true_negatives_i)
                        
                case "tp_rate":
                    if  (nb_true_positives_i + nb_false_negatives_i) == 0:
                        metric_results[i] = 0
                    else:
                        metric_results[i] = nb_true_positives_i / (nb_true_positives_i + nb_false_negatives_i)
            
        avg_metrics[metric] = np.mean(metric_results)
        std_metrics[metric] = np.std(metric_results)

    print("Average metrics: ", avg_metrics)
    print("Standard deviation of metrics: ", std_metrics)

    # change name of std_metrics as std_{metric_name}
    for metric in metrics:
        std_metrics["std_" + metric] = std_metrics[metric]
        del std_metrics[metric]
    
    avg_metrics.update(std_metrics)
    metrics_dict = avg_metrics
    
    # add TP, TN, FP, FN to the metrics_dict
    metrics_dict["TP"] = np.mean(nb_true_positives)
    metrics_dict["TN"] = np.mean(nb_true_negatives)
    metrics_dict["FP"] = np.mean(nb_false_positives)
    metrics_dict["FN"] = np.mean(nb_false_negatives)
    
    return metrics_dict

def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log

def create_logger_file(log_path):
    if log_path is None:
        if self.experiment_path is None:
            raise ValueError("Experiment path not set")
        log_path = self.experiment_path
    
    # create log file
    with open(f"{log_path}", "w") as f:
        f.write("")

    log = create_logger(__name__, silent=False, to_disk=True,
                                log_file=f"{log_path}")
    
    return log
def get_threshold_for_results(eval_json_path: str, target_fpr: float):
    
    # load the test results
    with open(eval_json_path) as f:
        data = json.load(f)
    df_eval = pd.json_normalize(data)
            
    # get the fpr, tpr and thresholds
    fpr_at_thresholds = df_eval["fpr_at_thresholds"].values[0]
    thresholds = df_eval["thresholds"].values[0]

    # Find the threshold that gives the target FPR
    threshold = None
    for i, fpr in enumerate(fpr_at_thresholds):
        if fpr > target_fpr:
            threshold = thresholds[i]
            break
        
    # safety check. TODO: check if this branch is ever reached
    if threshold is None:
        print(f"Could not find a threshold that gives a FPR > {target_fpr}")
        threshold = thresholds[-1]
    
    return float(threshold)