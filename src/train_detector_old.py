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

from detector import LLMDetector
from utils import create_logger, Signal, compute_bootstrap_acc, compute_bootstrap_metrics



def tokenize_text(x, tokenizer):
    return tokenizer(x["text"], truncation=True, padding="max_length", return_tensors="pt")

def prepare_fact_checking_dataset(tokenizer):
    """
    Process the fact completion dataset to filter out samples where the size of the question with the answer
    is not equal to the size of the question + mask size
    """

    fact_completion_dataset = load_dataset('Polyglot-or-Not/Fact-Completion')["English"]

    mask_token = tokenizer.mask_token + "."
    mask_token_size = len(tokenizer(mask_token).input_ids)

    # this number seems to be 3 for all models of interest
    single_word_answer_token_size = 3
    #fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(tokenizer(x["true"]).input_ids) == single_word_answer_token_size)

    # filter out samples where size of question with answer is not equal to the size of the question + mask size
    fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(tokenizer(x["stem"] + " " + x["true"] + ".").input_ids) == len(tokenizer(x["stem"] + " " + mask_token).input_ids))

    return fact_completion_dataset

def prepare_dataset_for_checking_degradation(detector_name, fact_completion_dataset, batch_size, nb_samples=1000):
    """
    Prepare the given dataset format for checking the degradation of the model.
    We use MaskedLM task to check for degradation, so we need to add a mask token depending
    on the detector model
    """
    """
    fact_completion_dataset = load_dataset('Polyglot-or-Not/Fact-Completion')["English"]

    # filter out questions where the answer is not a single word for the tokenizer
    len_before = len(fact_completion_dataset)
    #fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(x["true"].split()) == 1)

    # this number seems to be 3 for all models of interest
    single_word_answer_token_size = 3
    fact_completion_dataset = fact_completion_dataset.filter(lambda x: len(tokenizer(x["true"]).input_ids) == single_word_answer_token_size)
    len_after = len(fact_completion_dataset)
    #print(f"Filtered out {len_before - len_after} questions out of {len_before}")
    """
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

def create_mlm_model(model_name, device):
    """
    Create a MaskedLM model from the given model_name
    """

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
    
    
    return model


def adapt_model_to_mlm(classif_model, mlm_model, model_name):
    """
    Adapt the given classification model to the MaskedLM model by transferring the weights
    """

    # transfer the roberta weights to the MaskedLM model
    if model_name == "bert_base" or model_name == "bert_large":
        mlm_model.bert = classif_model.bert

    elif model_name == "roberta_base" or model_name == "roberta_large" or model_name == "distil_roberta-base":
        mlm_model.roberta = classif_model.roberta

    elif model_name == "electra_base" or model_name == "electra_large":
        mlm_model.electra = classif_model.electra

    else:
        raise ValueError("No other detector currently supported")
    
    return mlm_model



def check_model_degradation(classif_model, tokenizer, batches, nb_samples, log, mlm_model, model_name):
    """
    Check the degradation of the model by tracking the loss on the MaskedLM task
    """


    model = adapt_model_to_mlm(classif_model, mlm_model, model_name)

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

def run(num_epochs, model, tokenizer, dataset, learning_rate, warmup_ratio, weight_decay, batch_size, save_dir):

    eval_acc_logs = []
    eval_steps = 0.05
    total_nb_steps = len(dataset["train"]) * num_epochs
    training_steps = int(total_nb_steps // batch_size)

    # list of number of steps at which to evaluate the model
    eval_milestones = list(range(0, training_steps, int(training_steps * eval_steps)))
    eval_milestones_samples = [step * batch_size for step in eval_milestones]

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # special case for the T5 model
        if type(logits) is tuple:
            logits = logits[0]

        predictions = np.argmax(logits, axis=-1)
        print("labels", labels)
        print("predictions", predictions)
        acc = metric.compute(predictions=predictions, references=labels)
        eval_acc_logs.append(acc)
        return acc
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_steps=0.05,
        eval_steps=eval_steps,
        # This is important to set evaluation strategy, otherwise there will be no evaluation
        evaluation_strategy="steps",
        save_steps=0.2,
        save_total_limit=4,
        #logging_dir="./logs",
        #report_to="wandb",
        seed=42,
        fp16=True,
    )

    """
    class CustomCallback(TrainerCallback):
        
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_evaluate(self, args, state, control, **kwargs):
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
    """
        

    
    """
    class CustomTrainer(Trainer):

        def compute_loss(self, model, inputs, return_outputs=False):
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)

            # code for calculating accuracy
            if "labels" in inputs:
                logits = outputs.logits.detach()
                preds = logits.argmax(dim=1)
                acc1 = metric.compute(predictions=preds, references=inputs.labels)
                self.log({'accuracy_score': acc1})
                acc = (
                    (preds.argmax(axis=-1) == inputs.labels.reshape(1, len(inputs.labels))[0])
                    .type(torch.float)
                    .mean()
                    .item()
                )
                self.log({"train_accuracy": acc})
            # end code for calculating accuracy
                        
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
        
    """

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    #trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    eval_acc_logs_milestones_samples = {
        eval_milestones_samples[i]: eval_acc_logs[i] for i in range(len(eval_milestones_samples))
    }
    print("eval_acc_logs", eval_acc_logs_milestones_samples)


def process_tokenized_dataset(dataset):
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset

def create_experiment_folder(model_name, experiment_args):
    """
    Create a folder with the folowing elements:
    log, description of training with all the args, model folder with all the saved models
    """

    # create a folder with the folowing elements:
    # log, description of training with all the args, model folder with all the saved models

    training_method = None

    if experiment_args.use_adapter == "True":
        training_method = "adapter"
    elif experiment_args.freeze_base == "True" and experiment_args.use_adapter == "False":
        training_method = "freeze_base"
    elif experiment_args.freeze_base == "False":
        training_method = "full_finetuning"
    else:
        raise ValueError("Training method must be either 'freeze_base', 'adapter' or 'full_finetuning'")
    
    if training_method is None:
        raise ValueError("Training method must be either 'freeze_base', 'adapter' or 'full_finetuning'")

    base_path = experiment_args.save_dir
    # check if there exists a subfolder already for the model_name
    dataset_name = experiment_args.dataset_path.split("/")[-1]
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
        f.write(f"dataset_path: {experiment_args.dataset_path}\n")
        f.write(f"num_epochs: {experiment_args.num_epochs}\n")
        f.write(f"batch_size: {experiment_args.batch_size}\n")
        f.write(f"learning_rate: {experiment_args.learning_rate}\n")
        f.write(f"warmup_ratio: {experiment_args.warmup_ratio}\n")
        f.write(f"weight_decay: {experiment_args.weight_decay}\n")
        f.write(f"device: {experiment_args.device}\n")
        f.write(f"evaluation: {experiment_args.evaluation}\n")
        f.write(f"model_path: {experiment_args.model_path}\n")
        f.write(f"log_mode: {experiment_args.log_mode}\n")
        f.write(f"freeze_base: {experiment_args.freeze_base}\n")
        f.write(f"save_dir: {experiment_args.save_dir}\n")
        f.write(f"fp16: {experiment_args.fp16}\n")
        f.write(f"check_degradation: {experiment_args.check_degradation}\n")
        f.write(f"add_more_layers: {experiment_args.add_more_layers}\n")
        f.write(f"use_adapter: {experiment_args.use_adapter}\n")


    return experiment_path


def run_training_loop(num_epochs, model, tokenizer, train_dataset, val_dataset,
                       learning_rate, warmup_ratio, weight_decay, batch_size,
                        save_dir, detector_name, experiment_path, dataset_path,
                        fp16=True, log=None, check_degradation=0,
                        log_loss_steps=200, eval_steps=500, freeze_base=False,
                        nb_error_bar_runs=5, wandb_experiment_name="detector_training",
                        training_args=None):

    experiment_saved_model_path = f"{experiment_path}/saved_models"
    sig = Signal("run_signal.txt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #shuffle the datasets
    #train_dataset = train_dataset.shuffle(seed=42)
    #val_dataset = val_dataset.shuffle(seed=42)

    # process both datasets
    train_dataset = process_tokenized_dataset(train_dataset)
    val_dataset = process_tokenized_dataset(val_dataset)

    # we set shuffle to False for the train_loader, since we want to keep the order of the samples to preserve the pairing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = lr_scheduler.LinearLR(optimizer, warmup_ratio)
    
    if fp16:
        accelerator = Accelerator(mixed_precision='fp16')
    else:
        accelerator = Accelerator()
    

    num_training_steps = math.ceil(num_epochs * len(train_loader))
    warmup_steps = math.ceil(num_training_steps * warmup_ratio)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare(val_loader)

    # round both up to the nearest multiple of batch_size
    log_loss_steps = (log_loss_steps + batch_size - 1) // batch_size * batch_size
    eval_steps = (eval_steps + batch_size - 1) // batch_size * batch_size
    log.info(f"log_loss_steps: {log_loss_steps}")
    log.info(f"eval_steps: {eval_steps}")

    eval_acc_logs = []
    train_loss_logs = []
    loss_degradation_logs = []

    best_model = None

    
    num_training_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(range(num_training_steps))

    tags = [detector_name, dataset_path]
    if fp16:
        tags.append("fp16")

    if freeze_base == "True":
        tags.append("freeze_base")
    else:
        tags.append("full_finetuning")

    if args.use_adapter == "True":
        tags.append("adapter")

    run = wandb.init(project=wandb_experiment_name, tags=tags, dir=experiment_path)

    # test model loss on mlm eval before training
    mlm_model = None
    degradation_check_batches = None
    nb_samples_seen = 0

    fact_completion_dataset = prepare_fact_checking_dataset(tokenizer)

    if args.check_degradation > 0:
        mlm_model = create_mlm_model(args.detector, args.device)
        degradation_losses = []
        for i in range(nb_error_bar_runs):           
            degradation_check_batches = prepare_dataset_for_checking_degradation(args.detector, fact_completion_dataset, args.batch_size, nb_samples=1000)
            ref_degredation_loss = check_model_degradation(detector_model, bert_tokenizer, degradation_check_batches, nb_samples_seen, log, mlm_model, args.detector)
            degradation_losses.append(ref_degredation_loss) 

        mean_degradation_loss = sum(degradation_losses) / len(degradation_losses)
        std_degradation_loss = np.std(degradation_losses)
        
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
                optimizer.step()
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

                    run.log(metrics, step=nb_samples_seen)

                    running_loss = 0

                
                # check degradation of the model before evaluating the model, because is meaningless if the model has
                # too much degradation
                if check_degradation > 0 and ((i + 1) * batch_size) % check_degradation == 0:
                    nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size

                    degradation_losses = []
                    for i in range(nb_error_bar_runs):
                        degradation_check_batches = prepare_dataset_for_checking_degradation(args.detector, fact_completion_dataset, args.batch_size, nb_samples=1000)
                        degradation_loss = check_model_degradation(model, tokenizer, degradation_check_batches, nb_samples_seen, log, mlm_model, detector_name)
                        degradation_losses.append(degradation_loss)

                    mean_degradation_loss = sum(degradation_losses) / len(degradation_losses)
                    std_degradation_loss = np.std(degradation_losses)
                    loss_degradation_logs.append({"samples": nb_samples_seen, "degrad_loss": mean_degradation_loss, "std": std_degradation_loss})

                    degradation_loss_threshold = args.degradation_threshold
                    if mean_degradation_loss > 0:
                        if orig_degradation_loss * (1 + degradation_loss_threshold) < mean_degradation_loss:
                            log.info(f"Model has degraded, original loss: {orig_degradation_loss}, current loss: {mean_degradation_loss}")
                            log.info(f"Stopping training")
                            break

                if ((i + 1) * batch_size) % eval_steps == 0:
                    model.eval()
                    nb_samples_seen = i*batch_size + epoch*len(train_loader)*batch_size
                    #for i in range(nb_error_bar_runs):
                    total, correct = 0, 0

                    # correct_list is a list of 0s and 1s of size len(val_loader) * batch_size
                    correct_list = []
                    for batch in val_loader:
                        with torch.no_grad():
                            input_ids = batch["input_ids"]
                            labels = batch["labels"]
                            attention_mask = batch["attention_mask"]
                            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                            _, predicted = torch.max(outputs.logits, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                            correct_list.extend((predicted == labels).cpu().numpy().tolist())

                    # compute error bars with bootstrapping
                    nb_bootstraps = 1000
                    mean_acc, std_acc, lower_bound, upper_bound = compute_bootstrap_acc(correct_list, n_bootstrap=nb_bootstraps)
                    log.info(f"Mean accuracy: {mean_acc:.4f}, std: {std_acc:.4f}, lower bound: {lower_bound:.4f}, upper bound: {upper_bound:.4f} for {nb_bootstraps} bootstraps")
                    eval_acc = correct / total
                    log.info(f'Epoch {epoch+1}/{num_epochs}, Validation accuracy after {nb_samples_seen} samples: {eval_acc:.4f}')
                    eval_acc_logs.append({"samples": nb_samples_seen, "accuracy": mean_acc, "std": std_acc, "lower_bound": lower_bound, "upper_bound": upper_bound})
                    run.log({"eval/accuracy": eval_acc}, step=nb_samples_seen)

                    if best_model is None or eval_acc > best_model["eval_acc"]:
                        best_model = {"eval_acc": eval_acc, "nb_samples": nb_samples_seen}
                        
                        # save the model
                        torch.save(model.state_dict(), f"{experiment_saved_model_path}/best_model.pt")
                        log.info(f"Best model with eval accuracy {eval_acc} with {nb_samples_seen} samples seen is saved")
                    model.train()

                    if args.stop_on_perfect_acc == "True" and eval_acc >= 0.999:
                        log.info("Accuracy is equal or above 99.9%")
                        log.info("Stopping training")
                        break
                               



            log.info("Training signal is False, stopping training")
            break
    
    log.info(f"Training finished")
    log.info(f"Best model: {best_model}")
    log.info(f"Training loss logs: {train_loss_logs}")
    log.info(f"Evaluation accuracy logs: {eval_acc_logs}")
    run.finish()
    plot_nb_samples_metrics(eval_acc_logs, save_path=f"{experiment_path}/plots")
    plot_nb_samples_loss(train_loss_logs, save_path=f"{experiment_path}/plots")

    if check_degradation > 0:
        plot_degradation_loss(loss_degradation_logs, save_path=f"{experiment_path}/plots")


    # combine all 3 logs list to a list of dictionaries with the following keys: nb_samples_seen, train_loss, eval_accuracy, degradation_loss
    # this will be saved to a json file
    combined_logs = eval_acc_logs + train_loss_logs + loss_degradation_logs
    with jsonlines.open(f"{experiment_path}/training_logs.json", "w") as combined_logs_file:
        for log in combined_logs:
            combined_logs_file.write(log)

def plot_nb_samples_metrics(eval_acc_logs, save_path):

    eval_acc_logs_df = pd.DataFrame(eval_acc_logs)
    plt.figure()
    sns.lineplot(x="samples", y="accuracy", data=eval_acc_logs_df)

    plt.savefig(f"{save_path}/accuracy_vs_nb_samples.png")

def plot_nb_samples_loss(train_loss_logs, save_path):

    train_loss_logs_df = pd.DataFrame(train_loss_logs)
    plt.figure()
    sns.lineplot(x="samples", y="loss", data=train_loss_logs_df)

    plt.savefig(f"{save_path}/loss_vs_nb_samples.png")

def plot_degradation_loss(loss_degradation_logs, save_path):

    loss_degradation_logs_df = pd.DataFrame(loss_degradation_logs)
    plt.figure()
    sns.lineplot(x="samples", y="degrad_loss", data=loss_degradation_logs_df)

    plt.savefig(f"{save_path}/degradation_loss_vs_nb_samples.png")


def test_model(model, batch_size, dataset, experiment_path, log, dataset_path):

    # load best model
    #model.load_state_dict(torch.load(f"{experiment_path}/saved_models/best_model.pt"))
    model.eval()
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # load trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_eval_batch_size=batch_size,
            output_dir=f"{experiment_path}/test",
        ),
        compute_metrics=compute_metrics,
        eval_dataset=dataset["test"]
    )
    
    log.info(f"Evaluating the best model on the test set of dataset {dataset_path}...")
    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    #results = metric.compute(predictions=preds, references=predictions.label_ids)

    results = compute_bootstrap_metrics(preds, predictions.label_ids)
    
    log.info("Test metrics:")
    for key, value in results.items():
        log.info(f"{key}: {value}")

    #test_metrics_df = pd.DataFrame(results.items(), columns=["metric", "value"])
    #test_metrics_df.to_json(f"{experiment_path}/test/test_metrics.json")
        
    # save the results to a json file
    with jsonlines.open(f"{experiment_path}/test/test_metrics_{dataset_path}.json", "w") as test_metrics_file:
        test_metrics_file.write(results)


def create_round_robbin_dataset(datasets, take_samples=-1, seed=42):
    """
    Create a round robbin dataset from the given datasets
    """

    if take_samples > 0:
        datasets = [dataset.select(range(take_samples)) for dataset in datasets]

    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=seed)
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector", type=str, help="Name of the model to train from: roberta, bert, e5", default="roberta")
    parser.add_argument("--dataset_path", type=str, help="Path to the fake true dataset (generated with generate_fake_true_dataset.py)", default="fake_true_dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size to train the model", default=8)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train the model", default=3)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the model", default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for the model", default=0.1)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the model", default=0.01)
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
    parser.add_argument("--stop_on_perfect_acc", type=str, help="Whether to stop training when the model reaches 99.9% accuracy", default="False")
    parser.add_argument("--round_robin_training", type=str, help="Whether to train the model in a round robin fashion with multiple datasets", default="False")
    args = parser.parse_args()


    os.environ["WANDB_PROJECT"] = "detector_training"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # set wandb to offline mode
    if args.log_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif args.log_mode == "online":
        os.environ["WANDB_MODE"] = "online"
    else:
        raise ValueError("Log mode must be either 'offline' or 'online'")
    
    # train on multiple datasets in a round robin fashion
    if args.round_robin_training == "True":

        # check if the round_robin_datasets folder exists, otherwise create it
        if not os.path.isdir("./fake_true_datasets/fake_true_dataset_round_robin"):
            base_dataset_path = "./fake_true_datasets"
            datasets_names = ["fake_true_dataset_gpt2_10k", "fake_true_dataset_phi_10k", "fake_true_dataset_gemma_10k", "fake_true_dataset_mistral_10k"]
            datasets = [load_from_disk(f"{base_dataset_path}/{dataset_name}") for dataset_name in datasets_names]
            
            nb_samples_per_dataset = 2500
            datasets_train = [dataset["train"] for dataset in datasets]
            dataset_train = create_round_robbin_dataset(datasets_train, take_samples=nb_samples_per_dataset, seed=42)

            nb_samples_per_dataset = 250
            datasets_valid = [dataset["valid"] for dataset in datasets]
            dataset_valid = create_round_robbin_dataset(datasets_valid, take_samples=nb_samples_per_dataset, seed=42)

            nb_samples_per_dataset = 250
            datasets_test = [dataset["test"] for dataset in datasets]
            dataset_test = create_round_robbin_dataset(datasets_test, take_samples=nb_samples_per_dataset, seed=42)

            dataset = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})
            dataset.save_to_disk("./fake_true_datasets/fake_true_dataset_round_robin")
        else:
            dataset = load_from_disk("./fake_true_datasets/round_robin_dataset")
    else:
        dataset = load_from_disk(args.dataset_path)

    # only take a subset of the dataset
    if args.take_samples > 0:
        print(f"Taking {args.take_samples} samples from the dataset")
        dataset_train = dataset["train"].select(range(int(args.take_samples)))
        dataset_valid = dataset["valid"].select(range(int(args.take_samples / 10)))
        dataset_test = dataset["test"].select(range(int(args.take_samples / 10)))
        dataset = DatasetDict({"train": dataset_train, "valid": dataset_valid, "test": dataset_test})

    if args.evaluation == "False":
 
        # base models
        if args.detector == "roberta_base":
            #detector_path = "openai-community/roberta-base-openai-detector"
            #detector_path = "FacebookAI/roberta-large"
            detector_path = "FacebookAI/roberta-base"
            detector_model = RobertaForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "bert_base":
            detector_path = "bert-base-uncased"
            detector_model = BertForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "electra_base":
            detector_path = "google/electra-base-discriminator"
            detector_model = ElectraForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "t5_base":
            #detector_path = "google-t5/t5-base"
            # the path above has issues when fp16 is set to True
            detector_path = "google-t5/t5-base"
            detector_model = T5ForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

            # set fp16 to False for T5 model, t5 has issues with fp16
            args.fp16 = False

        # large models
        elif args.detector == "roberta_large":
            #detector_path = "openai-community/roberta-base-openai-detector"
            #detector_path = "FacebookAI/roberta-large"
            detector_path = "FacebookAI/roberta-large"
            detector_model = RobertaForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "bert_large":
            detector_path = "bert-large-uncased"
            detector_model = BertForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "electra_large":
            detector_path = "google/electra-large-discriminator"
            detector_model = ElectraForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        elif args.detector == "t5_3b":
            #detector_path = "google-t5/t5-base"
            # the path above has issues when fp16 is set to True
            detector_path = "google-t5/t5-3b"
            detector_model = T5ForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

            # set fp16 to False for T5 model, t5 has issues with fp16
            args.fp16 = False

        # distil models
        elif args.detector == "distil_roberta-base":
            detector_path = "distilbert/distilroberta-base"
            #detector_model = DistilRobertaForSequenceClassification.from_pretrained(detector_path).to(args.device)
            detector_model = AutoModelForSequenceClassification.from_pretrained(detector_path).to(args.device)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)
            detector = LLMDetector(detector_model, bert_tokenizer, 2)

        else:
            raise ValueError("No other detector currently supported")
        

        if args.freeze_base == "True":
            LLMDetector.freeze_base(detector_model)

        if args.add_more_layers == "True":
            LLMDetector.add_more_layers(detector_model)

        if args.use_adapter == "True":
            adapters.init(detector_model)
            config = adapters.BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
            detector_model.add_adapter("fake_true_detection", config=config)
            detector_model.train_adapter("fake_true_detection")
            detector_model.to(args.device)


        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)
        #run(args.num_epochs, detector_model, bert_tokenizer, dataset, args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size, args.save_dir)       
        experiment_path = create_experiment_folder(args.detector, args)

        # create log file
        with open(f"{experiment_path}/log.txt", "w") as f:
            f.write("")

        degradation_check_batches = None   
        mlm_model = None
        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{experiment_path}/log.txt")        

        dataset_name = args.dataset_path.split("/")[-1]
        # run the training loop
        run_training_loop(args.num_epochs, detector_model, bert_tokenizer, dataset["train"], dataset["valid"],
                           args.learning_rate, args.warmup_ratio, args.weight_decay, args.batch_size, args.save_dir, args.detector, experiment_path, dataset_name,
                           args.fp16, log, args.check_degradation, args.log_loss_steps, args.eval_steps, args.freeze_base,
                           args.nb_error_bar_runs, args.wandb_experiment_name, args)
        
        # evaluate model on the test set after training by loading the best model
        test_model(detector_model, args.batch_size, dataset, experiment_path, log, args.dataset_path)

    elif args.evaluation == "True":

        if args.detector == "roberta_base":
            detector_path = "FacebookAI/roberta-base"
            config = AutoConfig.from_pretrained(detector_path)
            model = RobertaForSequenceClassification(config)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif args.detector == "bert_base":
            detector_path = "bert-base-uncased"
            config = AutoConfig.from_pretrained(detector_path)
            model = BertForSequenceClassification(config)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif args.detector == "electra_base":
            detector_path = "google/electra-base-discriminator"
            config = AutoConfig.from_pretrained(detector_path)
            detector_model = ElectraForSequenceClassification(config)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif args.detector == "t5_base":
            detector_path = "google-t5/t5-base"
            config = AutoConfig.from_pretrained(detector_path)
            detector_model = T5ForSequenceClassification(config)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

        if args.detector == "roberta_large":
            detector_path = "FacebookAI/roberta-large"
            config = AutoConfig.from_pretrained(detector_path)
            model = RobertaForSequenceClassification(config)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        elif args.detector == "bert_large":
            detector_path = "bert-large-uncased"
            config = AutoConfig.from_pretrained(detector_path)
            model = BertForSequenceClassification(config)
            bert_tokenizer = BertTokenizer.from_pretrained(detector_path)

        elif args.detector == "electra_large":
            detector_path = "google/electra-large-discriminator"
            config = AutoConfig.from_pretrained(detector_path)
            detector_model = ElectraForSequenceClassification(config)
            bert_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

        elif args.detector == "t5_3b":
            detector_path = "google-t5/t5-3b"
            config = AutoConfig.from_pretrained(detector_path)
            detector_model = T5ForSequenceClassification(config)
            bert_tokenizer = T5Tokenizer.from_pretrained(detector_path)

        elif args.detector == "distil_roberta-base":
            detector_path = "distilbert/distilroberta-base"
            config = AutoConfig.from_pretrained(detector_path)
            model = AutoModelForSequenceClassification.from_pretrained(detector_path)
            bert_tokenizer = RobertaTokenizer.from_pretrained(detector_path)

        else:
            raise ValueError("No other detector currently supported")

        if args.use_adapter == "True":
            adapters.init(model)
            config = adapters.BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
            model.add_adapter("fake_true_detection", config=config)

        # load the trained weights
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(args.device)

        dataset = dataset.map(lambda x: tokenize_text(x, bert_tokenizer), batched=True)

        # if args.model_path is experiment_path/saved_models/best_model.pt, then the experiment_path is experiment_path
        experiment_path = args.model_path.split("/saved_models")[0]

        # create log file
        dataset_name = args.dataset_path.split("/")[-1]
        with open(f"{experiment_path}/test/log_{dataset_name}.txt", "w") as f:
            f.write("")

        log = create_logger(__name__, silent=False, to_disk=True,
                                    log_file=f"{experiment_path}/test/log_{dataset_name}.txt")

        test_model(model, args.batch_size, dataset, experiment_path, log, dataset_name)
    else:
        raise ValueError("Evaluation mode must be either True or False")

    


    