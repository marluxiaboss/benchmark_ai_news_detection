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

from .experiment_pipeline import ExperimentPipeline
from .pipeline_utils import create_logger_file, get_threshold_for_results, compute_bootstrap_metrics
from ..detector import Detector


class ExperimentTestDetectorPipeline(ExperimentPipeline):
    def __init__(
        self,
        cfg: dict,
        detector: Detector,
        experiment_path: str,
        non_attack_experiment_path: str,
        dataset_experiment_path: str,
        batch_size: int = 1,
    ) -> None:
        """
        Pipeline class used for testing a detector on an already existing dataset of fake and true articles.

        Parameters:
        ----------
            cfg: dict
                The hydra configuration of the experiment (generation, watermarking and detection config)
            detector: Detector
                The detector to test.
            experiment_path: str
                The path to the experiment used for saving the results.
            dataset_experiment_path: str
                The path to the experiment used for loading the dataset.
            batch_size: int
                The batch size to use for detection. Default is 1.
        """

        self.cfg = cfg
        self.experiment_path = experiment_path
        self.non_attack_experiment_path = non_attack_experiment_path
        self.batch_size = batch_size
        self.detector_name = cfg.detection.detector_name

        self.dataset_experiment_path = dataset_experiment_path
        self.dataset_experiment_name = cfg.generation.experiment_name
        self.experiment_name = f"{self.detector_name}_{cfg.detection.experiment_name}"

        # used to determine the detection threshold depending on the target fpr
        self.target_fpr = cfg.detection.target_fpr

        # check that folder at experiment path exists, if not create it
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if not os.path.exists(f"{experiment_path}/log"):
            os.makedirs(f"{experiment_path}/log")

        # setup log
        log_path = f"{experiment_path}/log/log_{self.experiment_name}.txt"
        self.log = create_logger_file(log_path)

        # set the detector
        self.detector = detector

    def find_threshold(self, eval_set: Dataset, target_fpr: float) -> float:
        """
        Find the detection threshold for the target false positive rate (FPR) on the evaluation set.
        Uses the fpr and thresholds obtained with the roc_curve function from sklearn.

        Parameters:
        ----------
            eval_set: Dataset
                The evaluation set to use for finding the threshold.
            target_fpr: float
                The target false positive rate to find the threshold for.

        Returns:
            threshold: float
                The detection threshold for the target FPR.
        """

        log = self.log
        # dataset_name = self.dataset_experiment_name
        detection_experiment_name = self.cfg.detection.experiment_name

        # check if eval folder already exists
        eval_json_path = f"{self.non_attack_experiment_path}/{self.cfg.generation.generator_name}/eval/{detection_experiment_name}.json"
        if not os.path.exists(eval_json_path):
            # os.makedirs(eval_path)

            if self.cfg.generation.attack_type != "no_attack":
                raise ValueError(
                    "Results for finding the threshold do not exist yet. Please run the detection pipeline with the non-attack dataset first."
                )
            else:
                log.info(
                    "Results for finding the threshold do not exist yet. Computing them now..."
                )

                fake_true_articles = eval_set["text"][:]

                log.info("Classifying the articles...")

                # base threshold
                threshold = self.cfg.detection.detection_threshold
                preds, logits, preds_at_threshold = self.detector.detect(
                    fake_true_articles, self.batch_size, threshold
                )
                labels = eval_set["label"]

                data_split = "eval"
                self.evaluate_detector(
                    preds, logits, preds_at_threshold, labels, eval_set, threshold, data_split
                )

                # get the threshold
                threshold = get_threshold_for_results(eval_json_path, target_fpr)

        else:
            log.info("Results for finding the threshold already exist. Loading them...")
            threshold = get_threshold_for_results(eval_json_path, target_fpr)

        log.info(f"Threshold for target FPR {target_fpr}: {threshold}")

        return threshold

    def evaluate_detector(
        self,
        preds: list[int],
        logits: list[float],
        preds_at_threshold: list[int],
        labels: list[int],
        dataset: Dataset,
        detection_threshold: float,
        data_split: str = "test",
    ):
        """
        Use the predictions and labels to compute the metrics of the detector and save them.

        Parameters:
        ----------
            preds: list[int]
                The predictions of the detector.
            logits: list[float]
                The logits of the detector.
            preds_at_threshold: list[int]
                The predictions of the detector at the given threshold.
            labels: list[int]
                The true labels of the dataset.
            dataset: Dataset
                The dataset used for testing the detector.
            detection_threshold: float
                The detection threshold used for the predictions at the given threshold.
            data_split: str
                The data split used for testing the detector. Default is "test".
        """

        log = self.log
        # dataset_experiment_name = self.dataset_experiment_name
        detection_experiment_name = self.cfg.detection.experiment_name

        preds = np.array(preds)
        logits = np.array(logits)
        labels = np.array(labels)
        preds_at_threshold = np.array(preds_at_threshold)

        # TODO: better to handle this in a helper
        # compute metrics
        nb_pos_labels = np.sum(labels == 1)
        nb_neg_labels = np.sum(labels == 0)

        if nb_pos_labels == 0 or nb_neg_labels == 0:
            log.info("Only one class in the dataset, cannot compute roc_auc")
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
        results["logits"] = logits.tolist()
        results["avg_logits"] = np.mean(logits)
        results["labels"] = labels.tolist()

        # compute average logits for the positive and negative class
        avg_logits_pos = np.mean(logits[labels == 1])
        avg_logits_neg = np.mean(logits[labels == 0])
        results["avg_logits_pos"] = avg_logits_pos
        results["avg_logits_neg"] = avg_logits_neg

        # Compute metrics at the given threshold
        results_at_threshold = compute_bootstrap_metrics(preds_at_threshold, labels)
        log.info("Test metrics at specific given threshold:")

        for key, value in results_at_threshold.items():
            log.info(f"{key}: {value}")

        # add them to results dict as f"{key}_at_given_threshold"
        results["given_threshold"] = detection_threshold
        for key, value in results_at_threshold.items():
            results[f"{key}_at_given_threshold"] = value

        # add generation + watermarking config to results
        results["generation_config"] = dict(self.cfg.generation)
        results["watermarking_config"] = dict(self.cfg.watermark)

        # save results
        experiment_path = self.experiment_path

        if not os.path.isdir(f"{experiment_path}/{data_split}"):
            os.makedirs(f"{experiment_path}/{data_split}")

        # json_res_file_path_base = (
        #    f"{experiment_path}/{data_split}/test_metrics_{detection_experiment_name}.json"
        # )
        json_res_file_path_base = f"{experiment_path}/{self.cfg.generation.generator_name}/{data_split}/{detection_experiment_name}.json"

        with open(json_res_file_path_base, "w") as f:
            f.write(json.dumps(results, indent=4))

    def run_pipeline(self):
        """
        Main function of the pipeline, runs the pipeline.
        First, find the detection threshold for the target FPR on the evaluation set.
        Then, test the detector on the test set using the detection threshold and save the results.
        """

        log = self.log

        # See if the dataset exists. This pipeline assumes that the dataset already exists
        dataset_path = f"{self.dataset_experiment_path}/{self.cfg.generation.generator_name}_{self.dataset_experiment_name}"
        if os.path.isdir(dataset_path):
            log.info(f"Dataset at {dataset_path} exists, loading it")
            dataset = load_from_disk(dataset_path)
        else:
            raise ValueError(
                f"Dataset at {dataset_path} does not exist, please create it first with create_dataset.py"
            )

        ### FIND THE DETECTION THRESHOLD FOR THE TARGET FPR ###
        eval_set = dataset["eval"]
        detection_threshold = self.find_threshold(eval_set, self.target_fpr)

        ### TEST THE DETECTOR ###

        # here we only test on the test set!
        dataset = dataset["test"]
        fake_true_articles = dataset["text"][:]

        log.info("Classifying the articles...")
        batch_size = self.batch_size
        preds, logits, preds_at_threshold = self.detector.detect(
            fake_true_articles, batch_size, detection_threshold
        )
        labels = dataset["label"]
        data_split = "test"
        self.evaluate_detector(
            preds, logits, preds_at_threshold, labels, dataset, detection_threshold, data_split
        )
