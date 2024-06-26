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
from sklearn.metrics import roc_auc_score, roc_curve, det_curve


from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
import evaluate
import wandb
import os
import argparse
import sys
import adapters
import jsonlines
import json

from datetime import datetime

from detector import LLMDetector
from utils import (create_logger, Signal, compute_bootstrap_acc, compute_bootstrap_metrics, create_round_robbin_dataset,
                   plot_nb_samples_metrics, plot_nb_samples_loss, plot_degradation_loss)

class DetectorTrainer:
    """
    Follows Builder pattern
    """
    def __init__(self):
        pass
    
    ### GENERAL METHODS (Builder pattern) ###

    def set_dataset(self, dataset_path, take_samples=-1, evaluation_mode=False, use_eval_set=False):
    
        dataset = load_from_disk(dataset_path)
        if evaluation_mode == "False":
            dataset_train = dataset["train"]
            dataset_valid = dataset["valid"]
            dataset_test = dataset["test"]

            dataset = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})

            if take_samples > 0:
                print(f"Taking {take_samples} samples from the dataset")
                dataset_train = dataset["train"].select(range(int(take_samples)))
                dataset_valid = dataset["valid"].select(range(int(take_samples / 10)))
                dataset_test = dataset["test"].select(range(int(take_samples / 10)))
                dataset = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})
        else:
            
            if use_eval_set:
                dataset_valid = dataset["valid"]
                
                # here we consider the valid set as the test set for simplicity
                dataset = DatasetDict({"test": dataset_valid})
            else:
                dataset_test = dataset["test"]
                dataset = DatasetDict({"test": dataset_test})

        self.dataset = dataset
        self.dataset_name = dataset_path.split("/")[-1]
        self.dataset_path = dataset_path

    def tokenize_dataset(self):

        if self.tokenizer is None:
            raise ValueError("Tokenizer not set")
        
        def tokenize_text(x, tokenizer):
            return tokenizer(x["text"], padding="max_length", truncation=True, return_tensors="pt")
        
        self.dataset = self.dataset.map(lambda x: tokenize_text(x, self.tokenizer), batched=True)

    def set_round_robin_dataset(self, take_samples=-1, nb_samples_per_dataset=2500):
        if not os.path.isdir("./fake_true_datasets/fake_true_dataset_round_robin"):
            base_dataset_path = "./fake_true_datasets"
            datasets_names = ["fake_true_dataset_gpt2_10k", "fake_true_dataset_phi_10k", "fake_true_dataset_gemma_10k", "fake_true_dataset_mistral_10k"]
            datasets = [load_from_disk(f"{base_dataset_path}/{dataset_name}") for dataset_name in datasets_names]
            
            nb_samples_per_dataset = nb_samples_per_dataset
            datasets_train = [dataset["train"] for dataset in datasets]
            dataset_train = create_round_robbin_dataset(datasets_train, take_samples=nb_samples_per_dataset, seed=42)

            nb_samples_per_dataset = nb_samples_per_dataset / 10
            datasets_valid = [dataset["valid"] for dataset in datasets]
            dataset_valid = create_round_robbin_dataset(datasets_valid, take_samples=nb_samples_per_dataset, seed=42)

            nb_samples_per_dataset = nb_samples_per_dataset / 10
            datasets_test = [dataset["test"] for dataset in datasets]
            dataset_test = create_round_robbin_dataset(datasets_test, take_samples=nb_samples_per_dataset, seed=42)

            dataset = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})
            dataset.save_to_disk("./fake_true_datasets/fake_true_dataset_round_robin")
        else:
            dataset = load_from_disk("./fake_true_datasets/fake_true_dataset_round_robin")

        self.dataset = dataset
        self.dataset_name = "fake_true_dataset_round_robin"
        self.dataset_path = "./fake_true_datasets/fake_true_dataset_round_robin"

    def create_experiment_folder(self, training_args):

        training_method = None

        if training_args.use_adapter == "True":
            training_method = "adapter"
        elif training_args.freeze_base == "True" and training_args.use_adapter == "False":
            training_method = "freeze_base"
        elif training_args.freeze_base == "False":
            training_method = "full_finetuning"
        else:
            raise ValueError("Training method must be either 'freeze_base', 'adapter' or 'full_finetuning'")
        
        if training_method is None:
            raise ValueError("Training method must be either 'freeze_base', 'adapter' or 'full_finetuning'")
        
        model_name = training_args.detector
        base_path = training_args.save_dir

        # check if there exists a subfolder already for the model_name
        dataset_name = self.dataset_path.split("/")[-1]
        if not os.path.isdir(f"{base_path}/{model_name}/{training_method}/{dataset_name}"):
            os.makedirs(f"{base_path}/{model_name}/{training_method}/{dataset_name}")
        
        experiment_path = f"{base_path}/{model_name}/{training_method}/{dataset_name}/{datetime.now().strftime('%d_%m_%H%M')}"
        os.makedirs(experiment_path)
        experiment_saved_model_path = f"{experiment_path}/saved_models"
        os.makedirs(experiment_saved_model_path)
        plots_path = f"{experiment_path}/plots"
        os.makedirs(plots_path)
        test_path = f"{experiment_path}/test"
        os.makedirs(test_path)

        # create a file args_log.txt with all the args
        with open(f"{experiment_path}/args_log.txt", "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"dataset_path: {self.dataset_path}\n")
            f.write(f"num_epochs: {training_args.num_epochs}\n")
            f.write(f"batch_size: {training_args.batch_size}\n")
            f.write(f"learning_rate: {training_args.learning_rate}\n")
            f.write(f"warmup_ratio: {training_args.warmup_ratio}\n")
            f.write(f"weight_decay: {training_args.weight_decay}\n")
            f.write(f"device: {training_args.device}\n")
            f.write(f"evaluation: {training_args.evaluation}\n")
            f.write(f"model_path: {training_args.model_path}\n")
            f.write(f"log_mode: {training_args.log_mode}\n")
            f.write(f"freeze_base: {training_args.freeze_base}\n")
            f.write(f"save_dir: {training_args.save_dir}\n")
            f.write(f"fp16: {training_args.fp16}\n")
            f.write(f"check_degradation: {training_args.check_degradation}\n")
            f.write(f"add_more_layers: {training_args.add_more_layers}\n")
            f.write(f"use_adapter: {training_args.use_adapter}\n")

        self.experiment_path = experiment_path

    def create_text_experiment_folder(self, base_path, detector_name):

        #experiment_path = f"{base_path}/{detector_name}/{datetime.now().strftime('%d_%m_%H%M')}"
        
        # it makes more sense to overwrite the last test than recreating a folder
        experiment_path = f"{base_path}/{detector_name}"
        
        # check if test folder exists
        test_folder = f"{experiment_path}/test"
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)
            
        return experiment_path

    def set_experiment_folder(self, experiment_path):
        self.experiment_path = experiment_path

    def create_train_logger(self, log_path=None):

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
    
    def freeze_base(self):
        LLMDetector.freeze_base(self.detector)
        self.freeze_base = True

    def use_adapter(self, train_adapter=True):
        LLMDetector.use_adapter(self.detector, self.detector.device, train_adapter)
        self.use_adapter = True

    def add_more_layers(self):
        LLMDetector.add_more_layers(self.detector)
        self.add_more_layers = True
        
    def set_classifier_threshold(self, threshold):
        self.classifier_threshold = threshold

    
    ### METHODS FOR SETTING TRAINING PARAMTERS ###
    def set_detector(self, detector_name, fp16=True, device="cuda"):

        self.fp16 = fp16

        # base models
        if detector_name == "roberta_base":
            #detector_path = "openai-community/roberta-base-openai-detector"
            #detector_path = "FacebookAI/roberta-large"
            detector_path = "FacebookAI/roberta-base"
            detector_model = RobertaForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif detector_name == "bert_base":
            detector_path = "bert-base-uncased"
            detector_model = BertForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif detector_name == "electra_base":
            detector_path = "google/electra-base-discriminator"
            detector_model = ElectraForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif detector_name == "t5_base":
            #detector_path = "google-t5/t5-base"
            # the path above has issues when fp16 is set to True
            detector_path = "google-t5/t5-base"
            detector_model = T5ForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

        # large models
        elif detector_name == "roberta_large":
            #detector_path = "openai-community/roberta-base-openai-detector"
            #detector_path = "FacebookAI/roberta-large"
            detector_path = "FacebookAI/roberta-large"
            detector_model = RobertaForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif detector_name == "bert_large":
            detector_path = "bert-large-uncased"
            detector_model = BertForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif detector_name == "electra_large":
            detector_path = "google/electra-large-discriminator"
            detector_model = ElectraForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif detector_name == "t5_3b":
            #detector_path = "google-t5/t5-base"
            # the path above has issues when fp16 is set to True
            detector_path = "google-t5/t5-3b"
            detector_model = T5ForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

            self.bf16 = True
            self.fp16 = False

        # distil models
        elif detector_name == "distil_roberta-base":
            detector_path = "distilbert/distilroberta-base"
            #detector_model = DistilRobertaForSequenceClassification.from_pretrained(detector_path).to(args.device)
            detector_model = AutoModelForSequenceClassification.from_pretrained(detector_path).to(device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        else:
            raise ValueError("No other detector currently supported")
        
        self.tokenizer = bert_tokenizer
        self.detector = detector_model
        self.detector_name = detector_name

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def set_warmup_ratio(self, warmup_ratio):
        self.warmup_ratio = warmup_ratio
    
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_wandb_experiment_name(self, wandb_experiment_name):
        self.wandb_experiment_name = wandb_experiment_name

    def set_stop_on_perfect_acc(self, stop_on_perfect_acc):
        self.stop_on_perfect_acc = stop_on_perfect_acc
        
    def set_stop_on_loss_plateau(self, stop_on_loss_plateau):
        self.stop_on_loss_plateau = stop_on_loss_plateau

    def set_stop_after_n_samples(self, stop_after_n_samples):
        self.stop_after_n_samples = stop_after_n_samples

    def set_log_loss_steps(self, log_loss_steps):
        self.log_loss_steps = log_loss_steps
    
    def set_eval_steps(self, eval_steps):
        self.eval_steps = eval_steps

    ### METHODS FOR SETTING TESTING PARAMTERS ###
    def set_pretrained_detector(self, detector_name):

        # Flag to set the weights of the model using a local .pt file or not (use weights from huggingface)
        self.set_weights = True

        self.flip_labels = False
        if detector_name == "roberta_base":
            detector_path = "FacebookAI/roberta-base"
            config = AutoConfig.from_pretrained(detector_path)
            model = RobertaForSequenceClassification(config)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif detector_name == "bert_base":
            detector_path = "bert-base-uncased"
            config = AutoConfig.from_pretrained(detector_path)
            model = BertForSequenceClassification(config)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif detector_name == "electra_base":
            detector_path = "google/electra-base-discriminator"
            config = AutoConfig.from_pretrained(detector_path)
            model = ElectraForSequenceClassification(config)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif detector_name == "t5_base":
            detector_path = "google-t5/t5-base"
            config = AutoConfig.from_pretrained(detector_path)
            model = T5ForSequenceClassification(config)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

        if detector_name == "roberta_large":
            detector_path = "FacebookAI/roberta-large"
            config = AutoConfig.from_pretrained(detector_path)
            model = RobertaForSequenceClassification(config)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif detector_name == "bert_large":
            detector_path = "bert-large-uncased"
            config = AutoConfig.from_pretrained(detector_path)
            model = BertForSequenceClassification(config)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif detector_name == "electra_large":
            detector_path = "google/electra-large-discriminator"
            config = AutoConfig.from_pretrained(detector_path)
            model = ElectraForSequenceClassification(config)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif detector_name == "t5_3b":
            detector_path = "google-t5/t5-3b"
            config = AutoConfig.from_pretrained(detector_path)
            model = T5ForSequenceClassification(config)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

        elif detector_name == "distil_roberta-base":
            detector_path = "distilbert/distilroberta-base"
            config = AutoConfig.from_pretrained(detector_path)
            model = AutoModelForSequenceClassification.from_pretrained(detector_path)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif detector_name == "roberta_base_open_ai":
            detector_path = "openai-community/roberta-base-openai-detector"
            model = RobertaForSequenceClassification.from_pretrained(detector_path)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
            self.set_weights = False

            # if we use this detector, we need to flip the labels
            # ie. 0 -> 1 and 1 -> 0
            self.flip_labels = True


        else:
            raise ValueError("No other detector currently supported")
        
        self.tokenizer = bert_tokenizer
        self.detector = model

    def set_pretrained_weights(self, model_path, device="cuda"):
        self.detector.load_state_dict(torch.load(model_path))
        self.detector.to(device)

    def create_test_logger(self, log_path=None):
    
        dataset_name = self.dataset_name

        #with open(f"{log_path}/test/log_{dataset_name}.txt", "w") as f:
        #    f.write("")
        
        # check if test folder exists
        if not os.path.isdir(f"{log_path}/test"):
            os.makedirs(f"{log_path}/test")

        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{log_path}/test/log_{dataset_name}.txt")

        self.log = log

    ### METHODS FOR CHECKING MODEL DEGRADATION ###
    def set_check_degradation_steps(self, nb_steps):
        self.check_degradation_steps = nb_steps

    def set_degradation_threshold(self, threshold):
        self.degradation_threshold = threshold

    def load_fact_checking_dataset(self):
        """
        Process the fact completion dataset to filter out samples where the size of the question with the answer
        is not equal to the size of the question + mask size
        """

        fact_completion_dataset = load_dataset('Polyglot-or-Not/Fact-Completion')["English"]

        mask_token = self.tokenizer.mask_token + "."

        # filter out samples where size of question with answer is not equal to the size of the question + mask size
        fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(self.tokenizer(x["stem"] + " " + x["true"] + ".").input_ids) == len(self.tokenizer(x["stem"] + " " + mask_token).input_ids))

        self.fact_completion_dataset = fact_completion_dataset
    
    def prepare_dataset_for_checking_degradation(self, nb_samples=1000):
        """
        Prepare the given dataset format for checking the degradation of the model.
        We use MaskedLM task to check for degradation, so we need to add a mask token depending
        on the detector model
        """

        detector_name = self.detector_name
        fact_completion_dataset = self.fact_completion_dataset
        batch_size = self.batch_size

        nb_samples_test_degradation = nb_samples
        #fact_completion_dataset = fact_completion_dataset.shuffle(seed=42)

        # we don't seed the shuffling to be able to measure the incertainty of the measure
        fact_completion_dataset = fact_completion_dataset.shuffle()
        fact_completion_dataset = fact_completion_dataset.select(range(nb_samples_test_degradation))
        questions = fact_completion_dataset["stem"]
        answers = fact_completion_dataset["true"]

        # add [MASK] at the end of each question
        # we also add a ".", this is very important for bert models

        if detector_name == "bert_base" or detector_name == "bert_large":
            questions_masked = [q + " [MASK]." for q in questions]

        elif detector_name == "roberta_base" or detector_name == "roberta_large" or detector_name == "distil_roberta-base":
            questions_masked = [q + " <mask>." for q in questions]

        elif detector_name == "electra_base" or detector_name == "electra_large":
            questions_masked = [q + " [MASK]." for q in questions]

        else:
            raise ValueError("No other detector currently supported")
        
        batches_questions_masked = [questions_masked[i:i + batch_size] for i in range(0, len(questions_masked), batch_size)]
        question_with_answers = [questions[i] + " " + answers[i] + "." for i in range(len(questions))]
        batches_answers = [question_with_answers[i:i + batch_size] for i in range(0, len(question_with_answers), batch_size)]

        batches = list(zip(batches_questions_masked, batches_answers))
        
        return batches
    
    def create_mlm_model(self):
        """
        Create a MaskedLM model from the given model_name
        """
        model_name = self.detector_name
        device = self.detector.device

        # base version of the model
        if model_name == "roberta_base":
            model = AutoModelForMaskedLM.from_pretrained("roberta-base").to(device)
        
        if model_name == "bert_base":
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        
        if model_name == "electra_base":
            model = AutoModelForMaskedLM.from_pretrained("google/electra-base-discriminator").to(device)

        if model_name == "t5_base":
            model = AutoModelForMaskedLM.from_pretrained("google-t5/t5-base").to(device)

        # large version of the model
        if model_name == "roberta_large":
            model = AutoModelForMaskedLM.from_pretrained("roberta-large").to(device)
        
        if model_name == "bert_large":
            model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased").to(device)
        
        if model_name == "electra_large":
            model = AutoModelForMaskedLM.from_pretrained("google/electra-large-discriminator").to(device)

        if model_name == "t5_3b":
            model = AutoModelForMaskedLM.from_pretrained("google-t5/t5-3b").to(device)

        # distil version of the model
        if model_name == "distil_roberta-base":
            model = AutoModelForMaskedLM.from_pretrained("distilroberta-base").to(device)
        
        self.mlm_model = model

    def adapt_model_to_mlm(self):
        """
        Adapt the given classification model to the MaskedLM model by transferring the weights
        """

        # transfer the roberta weights to the MaskedLM model
        if self.detector_name == "bert_base" or self.detector_name == "bert_large":
            self.mlm_model.bert = self.detector.bert

        elif self.detector_name == "roberta_base" or self.detector_name == "roberta_large" or self.detector_name == "distil_roberta-base":
            self.mlm_model.roberta = self.detector.roberta

        elif self.detector_name == "electra_base" or self.detector_name == "electra_large":
            self.mlm_model.electra = self.detector.electra

        else:
            raise ValueError("No other detector currently supported")
        

    def check_model_degradation(self, batches, nb_samples):
        """
        Check the degradation of the model by tracking the loss on the MaskedLM task
        """

        self.adapt_model_to_mlm()
        tokenizer = self.tokenizer
        log = self.log
        model = self.mlm_model

        losses = []
        model.eval()

        with torch.no_grad():
            for batch_question, batch_answer in tqdm(batches, desc="Answering fact completion questions..."):
            #for batch_question, batch_answer in batches:
                inputs = tokenizer(batch_question, return_tensors='pt', padding=True, truncation=True).to(model.device)

                questions_with_answer = batch_answer
                #questions_with_answer = [questions[i] + " " + answers[i] + "." for i in range(len(batch))]
                labels = tokenizer(questions_with_answer, return_tensors='pt', padding=True, truncation=True)["input_ids"].to(model.device)
                labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        log.info(f"Average Loss on fact answering task after {nb_samples} samples: {avg_loss:.4f}")

        return avg_loss

    def check_degradation(self, nb_error_bar_runs, nb_samples_seen):
        self.create_mlm_model()
        degradation_losses = []

        for i in range(nb_error_bar_runs):           
            degradation_check_batches = self.prepare_dataset_for_checking_degradation(nb_samples=1000)
            ref_degredation_loss = self.check_model_degradation(degradation_check_batches, nb_samples_seen)
            degradation_losses.append(ref_degredation_loss) 

        mean_degradation_loss = sum(degradation_losses) / len(degradation_losses)
        std_degradation_loss = np.std(degradation_losses)
        
        
        # clear memory
        del self.mlm_model
        torch.cuda.empty_cache()
        
        return mean_degradation_loss, std_degradation_loss

    ### METHODS FOR TRAINING AND EVALUATING ###
    def train(self):

        batch_size = self.batch_size
        num_epochs = self.num_epochs
        check_degradation = self.check_degradation_steps
        degradation_threshold = self.degradation_threshold
        stop_on_perfect_acc = self.stop_on_perfect_acc
        stop_on_loss_plateau = self.stop_on_loss_plateau
        weight_decay = self.weight_decay
        dataset = self.dataset
        model = self.detector
        log_loss_steps = self.log_loss_steps
        eval_steps = self.eval_steps

        experiment_saved_model_path = f"{self.experiment_path}/saved_models"
        sig = Signal("run_signal.txt")

        def process_tokenized_dataset(dataset):
            dataset = dataset.remove_columns(["text"])
            dataset = dataset.rename_column("label", "labels")
            dataset.set_format("torch")
            return dataset
        
        print(dataset["train"])
        # process both datasets
        train_dataset = process_tokenized_dataset(dataset["train"])
        val_dataset = process_tokenized_dataset(dataset["valid"])

        # we set shuffle to False for the train_loader, since we want to keep the order of the samples to preserve the pairing
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)


        optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=self.learning_rate, weight_decay=weight_decay)
        #scheduler = lr_scheduler.LinearLR(optimizer, warmup_ratio)
        
        if self.fp16:
            accelerator = Accelerator(mixed_precision='fp16')
            
        elif self.bf16:
            accelerator = Accelerator(mixed_precision='bf16')
        else:
            accelerator = Accelerator()
        

        num_training_steps = math.ceil(self.num_epochs * len(train_loader))
        warmup_steps = math.ceil(num_training_steps * self.warmup_ratio)
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        val_loader = accelerator.prepare(val_loader)

        # round both up to the nearest multiple of batch_size
        log_loss_steps = (log_loss_steps + self.batch_size - 1) // self.batch_size * self.batch_size
        eval_steps = (eval_steps + self.batch_size - 1) // self.batch_size * self.batch_size
        check_degradation = (check_degradation + self.batch_size - 1) // self.batch_size * self.batch_size

        log = self.log
        log.info(f"log_loss_steps: {log_loss_steps}")
        log.info(f"eval_steps: {eval_steps}")
        log.info(f"check_degradation: {check_degradation}")

        eval_acc_logs = []
        eval_loss_logs = []
        train_loss_logs = []
        loss_degradation_logs = []

        best_model = None
        
        num_training_steps = len(train_loader) * self.num_epochs
        progress_bar = tqdm(range(num_training_steps))

        tags = [self.detector_name, self.dataset_name]

        if self.fp16:
            tags.append("fp16")

        if self.freeze_base:
            tags.append("freeze_base")
        else:
            tags.append("full_finetuning")

        if self.use_adapter:
            tags.append("adapter")

        run = wandb.init(project=self.wandb_experiment_name, tags=tags, dir=self.experiment_path)

        if check_degradation > 0:
            nb_error_bar_runs = 5
            nb_samples_seen = 0

            # load fact completion dataset for the first time
            self.load_fact_checking_dataset()
            mean_degradation_loss, std_degradation_loss = self.check_degradation(nb_error_bar_runs, nb_samples_seen)
            orig_degradation_loss = mean_degradation_loss
            loss_degradation_logs.append({"samples": nb_samples_seen, "degrad_loss": mean_degradation_loss, "std": std_degradation_loss})

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0
            log.info(f"----------------Epoch {epoch+1}/{num_epochs}----------------")
            sig.update()
            if sig.training_sig:
                for i, batch in enumerate(train_loader):
                    sig.update()

                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                        
                    running_loss += loss.detach().item()

                    accelerator.backward(loss)

                    # clip the norm of the gradient to 1.0 (default) and save the base grad_norm before clipping
                    #grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    
                    # this should not affect the training, but is used to log the gradient norm
                    #grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.zero_grad()
                    scheduler.step()
                    progress_bar.update(1)

                    if ((i + 1) * batch_size) % log_loss_steps == 0:
                        nb_steps_log = log_loss_steps // batch_size
                        avg_loss = running_loss / nb_steps_log
                        nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                        log.info(f'Epoch {epoch+1}/{num_epochs}, Loss after {nb_samples_seen} samples: {avg_loss:.4f}')
                        train_loss_logs.append({"samples": nb_samples_seen, "loss": avg_loss})
                        metrics = {}
                        metrics["train/loss"] = avg_loss
                        metrics["train/learning_rate"] = scheduler.get_last_lr()[0]
                        metrics["train/epoch"] = epoch
                        #metrics["train/grad_norm"] = grad_norm

                        run.log(metrics, step=nb_samples_seen)

                        running_loss = 0

                    
                    # check degradation of the model before evaluating the model, because is meaningless if the model has
                    # too much degradation
                    if check_degradation > 0 and ((i + 1) * batch_size) % check_degradation == 0:
                        nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size

                        mean_degradation_loss, std_degradation_loss = self.check_degradation(nb_error_bar_runs, nb_samples_seen)
                        loss_degradation_logs.append({"samples": nb_samples_seen, "degrad_loss": mean_degradation_loss, "std": std_degradation_loss})

                        degradation_loss_threshold = degradation_threshold
                        if degradation_loss_threshold > 0:
                            if orig_degradation_loss * (1 + degradation_loss_threshold) < mean_degradation_loss:
                                log.info(f"Model has degraded, original loss: {orig_degradation_loss}, current loss: {mean_degradation_loss}")
                                log.info(f"Stopping training")
                                break

                    nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                    if (self.stop_after_n_samples > 0 and
                        nb_samples_seen > self.stop_after_n_samples):
                        log.info(f"Number of samples seen is above {self.stop_after_n_samples}")
                        log.info("Stopping training")
                        break
                    
                    if ((i + 1) * batch_size) % eval_steps == 0:
                        model.eval()
                        nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                        #for i in range(nb_error_bar_runs):
                        total, correct = 0, 0

                        # correct_list is a list of 0s and 1s of size len(val_loader) * batch_size
                        correct_list = []
                        eval_loss = 0
                        for batch in val_loader:
                            with torch.no_grad():
                                input_ids = batch["input_ids"]
                                labels = batch["labels"]
                                attention_mask = batch["attention_mask"]
                                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                                loss = outputs.loss
                                eval_loss += loss.item()
                                _, predicted = torch.max(outputs.logits, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                                correct_list.extend((predicted == labels).cpu().numpy().tolist())
                                
                        eval_loss /= len(val_loader)
                        eval_loss_logs.append(eval_loss)

                        # compute error bars with bootstrapping
                        nb_bootstraps = 1000
                        mean_acc, std_acc, lower_bound, upper_bound = compute_bootstrap_acc(correct_list, n_bootstrap=nb_bootstraps)
                        log.info(f"Mean accuracy: {mean_acc:.4f}, std: {std_acc:.4f}, lower bound: {lower_bound:.4f}, upper bound: {upper_bound:.4f} for {nb_bootstraps} bootstraps")
                        eval_acc = correct / total
                        log.info(f'Epoch {epoch+1}/{num_epochs}, Validation accuracy after {nb_samples_seen} samples: {eval_acc:.4f} with eval loss: {eval_loss:.4f}')
                        eval_acc_logs.append({"samples": nb_samples_seen, "accuracy": mean_acc, "std": std_acc, "lower_bound": lower_bound, "upper_bound": upper_bound})
                        run.log({"eval/accuracy": eval_acc}, step=nb_samples_seen)
                        run.log({"eval/loss": eval_loss}, step=nb_samples_seen)

                        if best_model is None or eval_loss < best_model["eval_loss"]:
                            best_model = {"eval_acc": eval_acc, "nb_samples": nb_samples_seen, "eval_loss": eval_loss}
                            
                            # save the model
                            torch.save(model.state_dict(), f"{experiment_saved_model_path}/best_model.pt")
                            log.info(f"Best model with eval loss {eval_loss} and eval accuracy {eval_acc} with {nb_samples_seen} samples seen is saved")
                        model.train()

                        if stop_on_perfect_acc == "True" and eval_acc >= 0.99:
                            log.info("Accuracy is equal or above 99.9%")
                            log.info("Stopping training")
                            break
                        
                        # stop training if no improvement in the last 3 eval_loss
                        n_plateau_loss = 5
                        if stop_on_loss_plateau:
                            curr_loss = eval_loss_logs[-1]
                            
                            # compare loss against the last 3 losses
                            if len(eval_loss_logs) > n_plateau_loss:
                                last_losses = eval_loss_logs[-n_plateau_loss:]
                                if all(curr_loss >= loss for loss in last_losses):
                                    log.info(f"No improvement in the last {n_plateau_loss} eval_loss")
                                    log.info("Stopping training")
                                    break

                log.info("Training signal is False, stopping training")
                break
        
        log.info(f"Training finished")
        log.info(f"Best model: {best_model}")
        log.info(f"Training loss logs: {train_loss_logs}")
        log.info(f"Evaluation accuracy logs: {eval_acc_logs}")
        run.finish()
        plot_nb_samples_metrics(eval_acc_logs, save_path=f"{self.experiment_path}/plots")
        plot_nb_samples_loss(train_loss_logs, save_path=f"{self.experiment_path}/plots")

        if check_degradation > 0:
            plot_degradation_loss(loss_degradation_logs, save_path=f"{self.experiment_path}/plots")


        # combine all 3 logs list to a list of dictionaries with the following keys: nb_samples_seen, train_loss, eval_accuracy, degradation_loss
        # this will be saved to a json file
        combined_logs = eval_acc_logs + train_loss_logs + loss_degradation_logs
        with jsonlines.open(f"{self.experiment_path}/training_logs.json", "w") as combined_logs_file:
            for log in combined_logs:
                combined_logs_file.write(log)

    def test(self, use_eval_set=False):

        model = self.detector
        batch_size = self.batch_size
        dataset = self.dataset
        experiment_path = self.experiment_path
        log = self.log
        dataset_path = self.dataset_path
        dataset_name = self.dataset_name

        if hasattr(self, "flip_labels") and self.flip_labels:
            log.info("Flipping labels for the dataset")
            dataset["test"] = dataset["test"].map(lambda x: {"label": 1 - x["label"]})

        model.eval()
        metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        test_dataset = dataset["test"]
            
        # load trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                per_device_eval_batch_size=batch_size,
                output_dir=f"{experiment_path}/test",
            ),
            compute_metrics=compute_metrics,
            eval_dataset=test_dataset
        )
        
        log.info(f"Evaluating the best model on the test set of dataset {dataset_path}...")
        predictions = trainer.predict(dataset["test"])
        preds = np.argmax(predictions.predictions, axis=-1)

        if not hasattr(self, "flip_labels"):
            flip_labels = False
        else:
            flip_labels = self.flip_labels
        
        predictions_pos_class = predictions.predictions[:, 1]
        
        nb_pos_labels = np.sum(predictions.label_ids)
        nb_neg_labels = len(predictions.label_ids) - nb_pos_labels
        
        if nb_pos_labels == 0 or nb_neg_labels == 0:
            log.info("Only one class in the dataset, cannot compute roc_auc")
            roc_auc = 0
            fpr = np.zeros(1)
            tpr = np.zeros(1)
            thresholds = np.zeros(1)
        else:
            if flip_labels:
                labels_flipped = 1 - predictions.label_ids
                preds_flipped = predictions.predictions[:, 0]
                roc_auc = roc_auc_score(labels_flipped, preds_flipped)
                fpr, tpr, thresholds = roc_curve(labels_flipped, preds_flipped)
            else:
                roc_auc = roc_auc_score(predictions.label_ids, predictions_pos_class)
                fpr, tpr, thresholds = roc_curve(predictions.label_ids, predictions_pos_class)
            
        #    fpr, tpr = 1 - fpr, 1 - tpr
        
        results = compute_bootstrap_metrics(preds, predictions.label_ids, flip_labels=flip_labels)
        
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
        
        if self.classifier_threshold is not None:
            
            #if flip_labels:
            #    self.classifier_threshold = 1 - self.classifier_threshold
            
            preds_at_threshold = np.where(predictions.predictions[:, 1] > self.classifier_threshold, 1, 0)

            results_at_threshold = compute_bootstrap_metrics(preds_at_threshold, predictions.label_ids, flip_labels=flip_labels)
            log.info("Test metrics at specific given threshold:")
            
            for key, value in results_at_threshold.items():
                log.info(f"{key}: {value}")
                
            # add them to results dict as f"{key}_at_given_threshold"
            results["given_threshold"] = self.classifier_threshold
            for key, value in results_at_threshold.items():
                results[f"{key}_at_given_threshold"] = value
            
        # save the results to a json file
        #with jsonlines.open(f"{experiment_path}/test/test_metrics_{dataset_name}.json", "w") as test_metrics_file:
        #    test_metrics_file.write(results)
        
        
        # define where to save the results
        if use_eval_set:
            
            if not os.path.isdir(f"{experiment_path}/eval"):
                os.makedirs(f"{experiment_path}/eval")
            
            json_res_file_path = f"{experiment_path}/eval/eval_metrics_{dataset_name}.json"
            
        else:
            if self.classifier_threshold is not None:
                if not os.path.isdir(f"{experiment_path}/test_at_threshold"):
                    os.makedirs(f"{experiment_path}/test_at_threshold")
                    
                json_res_file_path = f"{experiment_path}/test_at_threshold/test_metrics_{dataset_name}.json"
                
            else:
                if not os.path.isdir(f"{experiment_path}/test"):
                    os.makedirs(f"{experiment_path}/test")
            
                json_res_file_path = f"{experiment_path}/test/test_metrics_{dataset_name}.json"
                
        with open(json_res_file_path, "w") as f:
            f.write(json.dumps(results, indent=4))
 