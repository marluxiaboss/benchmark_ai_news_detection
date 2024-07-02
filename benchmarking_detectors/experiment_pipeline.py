import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json
from time import gmtime, strftime
import logging
import sys
import os
from datasets import load_from_disk
from abc import ABC, abstractmethod



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

class ExperimentPipeline(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def run_pipeline(self):
        pass
    
class ExperimentTestPipelineFull(ExperimentPipeline):
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
                
                
class ExperimentTestPipelineFull(ExperimentPipeline):
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
