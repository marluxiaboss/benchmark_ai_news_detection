import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertModel,
 RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback, ElectraForSequenceClassification, ElectraTokenizer, T5ForSequenceClassification, T5Tokenizer, get_scheduler,
    RobertaConfig, AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification)
from torch.optim import AdamW
from copy import deepcopy
from tqdm import tqdm
from accelerate import Accelerator
import math

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import evaluate
import wandb
import os
import argparse
import sys
import adapters
import jsonlines

from datetime import datetime

SRC_PATH = ["src"]
for module_path in SRC_PATH:
    if module_path not in sys.path:
        sys.path.append(module_path)

from detector import LLMDetector
from detector_trainer import DetectorTrainer
from utils import create_logger, Signal, compute_bootstrap_acc, compute_bootstrap_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector", type=str, help="Name of the model to train from: roberta, bert, e5", default="roberta")
    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size to train the model", default=8)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train the model", default=3)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the model", default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for the model", default=0.06)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the model", default=0.1)
    parser.add_argument("--device", type=str, help="Device to train the model", default="cuda")
    parser.add_argument("--evaluation", type=str, help="Evaluation mode for the model: True or False", default="False")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate", default="model")
    parser.add_argument("--log_mode", type=str, help="'offline' or 'online' (wandb)", default="offline")
    parser.add_argument("--freeze_base", type=str, help="Whether to freeze the base model", default="False")
    parser.add_argument("--save_dir", type=str, help="Directory to save the model and logs", default="./training_logs/detector_freeze_base")
    parser.add_argument("--fp16", type=str, help="Whether to use fp16", default="True")
    parser.add_argument("--check_degradation", type=int, help="If set to > 0, then check for the degration of the model after each check_degradation steps", default=0)
    parser.add_argument("--degradation_threshold", type=float, help="Threshold for the degradation of the model", default=-1.0)
    parser.add_argument("--log_loss_steps", type=int, help="How many samples seen before logging the loss", default=200)
    parser.add_argument("--eval_steps", type=int, help="How many samples seen before evaluating the model", default=500)
    parser.add_argument("--add_more_layers", type=str, help="Whether to add more layers to the classifier", default="False")
    parser.add_argument("--use_adapter", type=str, help="Whether to use adapter layers. If set to True, will use adapter and freeze the rest.", default="False")
    parser.add_argument("--nb_error_bar_runs", type=int, help="Number of runs to calculate the error bars for the metrics", default=5)
    parser.add_argument("--take_samples", type=int, help="Number of samples to take from the dataset", default=-1)
    parser.add_argument("--wandb_experiment_name", type=str, help="Name of the wandb experiment", default="detector_training")
    parser.add_argument("--round_robin_training", type=str, help="Whether to train the model in a round robin fashion with multiple datasets", default="False")
    parser.add_argument("--nb_samples_per_dataset", type=int, help="Number of samples to take from each dataset in round robin training", default=2500)
    parser.add_argument("--stop_after_n_samples", type=int, help="Stop training after n samples", default=-1)
    parser.add_argument("--stop_on_perfect_acc", type=str, help="Whether to stop training when the model reaches 99.9% accuracy", default="False")
    parser.add_argument("--stop_on_loss_plateau", help="Whether to stop the training when the eval loss plateaus", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--classifier_threshold", type=float, help="Threshold for the classifier", default=None)
    parser.add_argument("--experiment_path", type=str, help="Path to the experiment folder", default="")
    parser.add_argument("--use_eval_set", action=argparse.BooleanOptionalAction, help="Whether to use the eval set for training", default=False)
    args = parser.parse_args()

    # builder for training
    detector_trainer = DetectorTrainer()

    ### GENERAL SETUP ###
    os.environ["WANDB_PROJECT"] = "detector_training"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # set wandb to offline mode
    if args.log_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif args.log_mode == "online":
        os.environ["WANDB_MODE"] = "online"
    else:
        raise ValueError("Log mode must be either 'offline' or 'online'")
    
    detector_trainer.set_wandb_experiment_name(args.wandb_experiment_name)
    
    # train on multiple datasets in a round robin fashion
    if args.round_robin_training == "True":
        detector_trainer.set_round_robin_dataset(nb_samples_per_dataset=args.nb_samples_per_dataset)
    else:
        detector_trainer.set_dataset(args.dataset_path, args.take_samples, args.evaluation, args.use_eval_set)

    # general hyperparameters
    detector_trainer.set_batch_size(args.batch_size)

    ### TRAINING SETUP ###
    if args.evaluation == "False":
 
        detector_trainer.set_detector(args.detector)

        # training hyperparameters
        detector_trainer.set_learning_rate(args.learning_rate)
        detector_trainer.set_num_epochs(args.num_epochs)
        detector_trainer.set_warmup_ratio(args.warmup_ratio)
        detector_trainer.set_weight_decay(args.weight_decay)
        detector_trainer.set_check_degradation_steps(args.check_degradation)
        detector_trainer.set_degradation_threshold(args.degradation_threshold)
        detector_trainer.set_stop_on_perfect_acc(args.stop_on_perfect_acc)
        detector_trainer.set_stop_on_loss_plateau(args.stop_on_loss_plateau)
        detector_trainer.set_stop_after_n_samples(args.stop_after_n_samples)
        detector_trainer.set_log_loss_steps(args.log_loss_steps)
        detector_trainer.set_eval_steps(args.eval_steps)

        if args.freeze_base == "True":
            detector_trainer.freeze_base()

        if args.add_more_layers == "True":
            detector_trainer.add_more_layers()

        if args.use_adapter == "True":
            detector_trainer.use_adapter()

        detector_trainer.tokenize_dataset()

        detector_trainer.create_experiment_folder(args)
        detector_trainer.create_train_logger()

        # run the training loop
        detector_trainer.train()
        
        # evaluate model on the test set after training
        detector_trainer.test()

    elif args.evaluation == "True":

        detector_trainer.set_pretrained_detector(args.detector)

        if args.freeze_base == "True":
            detector_trainer.freeze_base()

        if args.use_adapter == "True":
            detector_trainer.use_adapter(train_adapter=False)

        # load the trained weights
        if detector_trainer.set_weights:
            detector_trainer.set_pretrained_weights(args.model_path)

        detector_trainer.tokenize_dataset()

        # if args.model_path is experiment_path/saved_models/best_model.pt, then the experiment_path is experiment_path
        if args.experiment_path == "":
            experiment_path = args.model_path.split("/saved_models")[0]
        else:
            experiment_path = detector_trainer.create_text_experiment_folder(args.experiment_path, args.detector)

        detector_trainer.set_experiment_folder(experiment_path)

        # create log file
        detector_trainer.create_test_logger(log_path=experiment_path)
        
        # set a threshold for classification if provided
        detector_trainer.set_classifier_threshold(args.classifier_threshold)

        detector_trainer.test(args.use_eval_set)
    else:
        raise ValueError("Evaluation mode must be either True or False")

    


    