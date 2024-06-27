import logging
from time import strftime, gmtime
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import json
import jsonlines

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets

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


def compute_bootstrap_acc(data, n_bootstrap=1000):

    accs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, len(data), replace=True)
        accs[i] = np.mean(bootstrap_sample)

    lower_bound = np.percentile(accs, 2.5)
    upper_bound = np.percentile(accs, 97.5)

    return (np.mean(accs), np.std(accs), lower_bound, upper_bound)


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

class Signal:
    """Running signal to control training process"""

    def __init__(self, signal_file):
        self.signal_file = signal_file
        self.training_sig = True

        self.update()

    def update(self):
        signal_dict = self.read_signal()
        self.training_sig = signal_dict['training']

    def read_signal(self):
        with open(self.signal_file, 'r') as fin:
            return eval(fin.read())
        
### Dataset utils ###
def create_round_robbin_dataset(datasets, take_samples=-1, seed=42):
    """
    Create a round robbin dataset from the given datasets
    """

    if take_samples > 0:
        datasets = [dataset.select(range(take_samples)) for dataset in datasets]

    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=seed)
    
    return dataset
    

### Ploting utils ###

# plots just after training
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

# plots for experiment 1
# distil-roberta trained on mistral with adapter

def create_df_from_training_logs(detector, training_method, model_code):

    detector_logs_path = f"./saved_training_logs_experiment_1/{detector}/{training_method}/fake_true_dataset_mistral_10k/{model_code}/training_logs.json"

    training_logs_data = []
    for line in jsonlines.open(detector_logs_path):
        training_logs_data.append(line)

    # transform to pandas dataframe
    df = pd.DataFrame(training_logs_data)

    # 1) eval_acc data
    eval_acc_df = df[df["accuracy"].notna()]
    eval_acc_df = eval_acc_df[["accuracy", "samples", "std", "lower_bound", "upper_bound"]]


    # 2) degrad_loss data
    degrad_loss_df = df[df["degrad_loss"].notna()]
    degrad_loss_df = degrad_loss_df[["degrad_loss", "samples", "std"]]


    # 3) training loss data
    training_loss_df = df[df["loss"].notna()]
    training_loss_df = training_loss_df[["loss", "samples"]]

    return eval_acc_df, degrad_loss_df, training_loss_df

def plot_eval_acc_vs_nb_samples_old(eval_acc_df, save_path=None):
    plt.fill_between(eval_acc_df["samples"], eval_acc_df["accuracy"] - eval_acc_df["std"],
                  eval_acc_df["accuracy"] + eval_acc_df["std"], alpha = 0.5, color = 'gray')
    plt.plot(eval_acc_df["samples"], eval_acc_df["accuracy"], color = 'black')
    # 
    plt.xlabel('Number of training samples seen')
    plt.ylabel('Evaluation accuracy')

    plt.title('Evaluation accuracy during training with standard deviation')


def plot_eval_acc_vs_nb_samples(eval_acc_df, save_path=None):

    plt.errorbar(eval_acc_df["samples"], eval_acc_df["accuracy"], yerr=eval_acc_df["std"], color = 'black', capsize=3)
    # 
    plt.xlabel('Number of training samples seen')
    plt.ylabel('Evaluation accuracy')

    plt.title('Evaluation accuracy during training with standard deviation')

def plot_degrad_loss_vs_nb_samples_old(degrad_loss_df, save_path=None):
    plt.fill_between(degrad_loss_df["samples"], degrad_loss_df["degrad_loss"] - degrad_loss_df["std"],
                  degrad_loss_df["degrad_loss"] + degrad_loss_df["std"], alpha = 0.5, color = 'gray')
    plt.plot(degrad_loss_df["samples"], degrad_loss_df["degrad_loss"], color = 'black')

    # add an horizontal line at 10.5, which is the baseline for a random model
    plt.axhline(y=10.5, color='r', linestyle='--')

    # name it "random model baseline"
    plt.text(0, 10.55, "random model baseline", color = 'red')

    plt.xlabel('Number of training samples seen')
    plt.ylabel('Degradation loss')

    plt.title('Degradation loss during training with standard deviation')

def plot_degrad_loss_vs_nb_samples(degrad_loss_df, save_path=None):
    #plt.fill_between(degrad_loss_df["samples"], degrad_loss_df["degrad_loss"] - degrad_loss_df["std"],
    #              degrad_loss_df["degrad_loss"] + degrad_loss_df["std"], alpha = 0.5, color = 'gray')
    #plt.plot(degrad_loss_df["samples"], degrad_loss_df["degrad_loss"], color = 'black')

    plt.errorbar(degrad_loss_df["samples"], degrad_loss_df["degrad_loss"], yerr=degrad_loss_df["std"], color = 'black', capsize=3)

    # add an horizontal line at 10.5, which is the baseline for a random model
    plt.axhline(y=10.5, color='r', linestyle='--')

    # name it "random model baseline"
    plt.text(0, 10.55, "random model baseline", color = 'red')

    plt.xlabel('Number of training samples seen')
    plt.ylabel('Degradation loss')

    plt.title('Degradation loss during training with standard deviation')

def plot_training_loss_vs_nb_samples(training_loss_df, save_path=None):
    plt.plot(training_loss_df["samples"], training_loss_df["loss"], color = 'black')
    plt.xlabel('Number of training samples seen')
    plt.ylabel('Training loss')

    plt.title('Train loss during training')

def plot_compared_model_size_eval_acc(eval_acc_df_distil_freeze_base, eval_acc_df_large_freeze_base, save_path=None):
    #fig, ax = plt.subplots(1,1,figsize= (8,6), sharey = True, sharex = True)

    ### Freeze base method ###
    plt.errorbar(eval_acc_df_distil_freeze_base["samples"], eval_acc_df_distil_freeze_base["accuracy"], yerr=eval_acc_df_distil_freeze_base["std"], color = 'black', capsize=3)

    # distil full
    plt.errorbar(eval_acc_df_large_freeze_base["samples"], eval_acc_df_large_freeze_base["accuracy"], yerr=eval_acc_df_large_freeze_base["std"], color = 'yellow', capsize=3)
    
    
    plt.title('Freeze base method')
    plt.xlabel('Number of training samples seen')
    plt.ylabel('Eval Accuracy')
    plt.legend(["Distil-RoBERTa", "RoBERTa-Large"])

    plt.show()


def plot_panel_model_and_training_method_eval_acc(eval_acc_df_distil_adapter, eval_acc_df_distil_full, eval_acc_df_large_adapter, eval_acc_df_large_full, save=False):
    fig, ax = plt.subplots(1,2,figsize= (12,6), sharey = True, sharex = True)

    ### Adapter method ###
    ax[0].errorbar(eval_acc_df_distil_adapter["samples"], eval_acc_df_distil_adapter["accuracy"], yerr=eval_acc_df_distil_adapter["std"], color = 'black', capsize=3)
    ax[0].set_title('Adapter method')
    ax[0].set_xlabel('Number of training samples seen')
    ax[0].set_ylabel('Eval Accuracy')

    # distil full
    ax[0].errorbar(eval_acc_df_large_adapter["samples"], eval_acc_df_large_adapter["accuracy"], yerr=eval_acc_df_large_adapter["std"], color = 'yellow', capsize=3)
    ax[0].legend(["Distil-RoBERTa", "RoBERTa-Large"])

    ### Full finetuning method ###
    ax[1].errorbar(eval_acc_df_distil_full["samples"], eval_acc_df_distil_full["accuracy"], yerr=eval_acc_df_distil_full["std"], color = 'black', capsize=3)
    ax[1].set_title('Full finetuning method')
    ax[1].set_xlabel('Number of training samples seen')
    ax[1].set_ylabel('Eval Accuracy')

    # distil full
    ax[1].errorbar(eval_acc_df_large_full["samples"], eval_acc_df_large_full["accuracy"], yerr=eval_acc_df_large_full["std"], color = 'yellow', capsize=3)
    ax[1].legend(["Distil-RoBERTa", "RoBERTa-Large"])
    
    # show y ticks each 0.05
    ax[0].set_yticks(np.arange(0.5, 1, 0.05))
    ax[1].set_yticks(np.arange(0.5, 1, 0.05))
    
    if save:
        plt.savefig("notebooks/plots/check_degradation_eval_loss.png")


def plot_panel_model_and_training_method_degrad(degrad_loss_df_distil_adapter, degrad_loss_df_distil_full, degrad_loss_df_large_adapter, degrad_loss_df_large_full, save=False):
    fig, ax = plt.subplots(1,2,figsize= (12,6), sharey = True, sharex = True)

    ### Adapter method ###
    ax[0].errorbar(degrad_loss_df_distil_adapter["samples"], degrad_loss_df_distil_adapter["degrad_loss"], yerr=degrad_loss_df_distil_adapter["std"], color = 'black', capsize=3)
    ax[0].set_title('Adapter method')
    ax[0].set_xlabel('Number of training samples seen')
    ax[0].set_ylabel('Degradation loss')

    # distil full
    ax[0].errorbar(degrad_loss_df_large_adapter["samples"], degrad_loss_df_large_adapter["degrad_loss"], yerr=degrad_loss_df_large_adapter["std"], color = 'yellow', capsize=3)
    ax[0].legend(["Distil-RoBERTa", "RoBERTa-Large"])

    ax[0].axhline(y=10.5, color='r', linestyle='--')
    ax[0].text(0, 10.55, "random model baseline", color = 'red')

    ### Full finetuning method ###
    ax[1].errorbar(degrad_loss_df_distil_full["samples"], degrad_loss_df_distil_full["degrad_loss"], yerr=degrad_loss_df_distil_full["std"], color = 'black', capsize=3)
    ax[1].set_title('Full finetuning method')
    ax[1].set_xlabel('Number of training samples seen')
    ax[1].set_ylabel('Degradation loss')

    # distil full
    ax[1].errorbar(degrad_loss_df_large_full["samples"], degrad_loss_df_large_full["degrad_loss"], yerr=degrad_loss_df_large_full["std"], color = 'yellow', capsize=3)
    ax[1].legend(["Distil-RoBERTa", "RoBERTa-Large"])

    ax[1].axhline(y=10.5, color='r', linestyle='--')
    ax[1].text(0, 10.55, "random model baseline", color = 'red')

    if save:
        plt.savefig("notebooks/plots/check_degradation.png")
    

# plots for experiment 2
def create_df_from_test_logs(training_method, trained_on_models, dataset_names, use_eval_split=False, use_test_at_threshold=False):

    results = []
    for detector in trained_on_models.keys():
        for model_code, base_model in trained_on_models[detector].items():
            for dataset in dataset_names:
                if use_eval_split:
                    results_path = f"./saved_training_logs_experiment_2/{detector}/{training_method}/fake_true_dataset_{base_model}_10k/{model_code}/eval/eval_metrics_fake_true_dataset_{dataset}_10k.json"
                else:
                    if use_test_at_threshold:
                        results_path = f"./saved_training_logs_experiment_2/{detector}/{training_method}/fake_true_dataset_{base_model}_10k/{model_code}/test_at_threshold/test_metrics_fake_true_dataset_{dataset}_10k.json"
                    else:   
                        results_path = f"./saved_training_logs_experiment_2/{detector}/{training_method}/fake_true_dataset_{base_model}_10k/{model_code}/test/test_metrics_fake_true_dataset_{dataset}_10k.json"

                with open(results_path, "r") as f:
                    result_dict = json.load(f)
                    result_dict["base_detector"] = detector
                    result_dict["trained_on_dataset"] = base_model
                    result_dict["detector"] = f"{detector}_{base_model}"
                    result_dict["dataset"] = dataset
                    results.append(result_dict)

    # transform to pandas dataframe
    results_df = pd.DataFrame(results)

    # order by detector and dataset
    results_df = results_df.sort_values(by=["trained_on_dataset", "detector", "dataset"])
    #results_df = results_df.sort_values(by=["detector", "dataset"])

    #display(results_df)

    return results_df

def create_df_from_test_logs_modified(training_method, trained_on_models, dataset_names, suffix):

    results = []
    for detector in trained_on_models.keys():
        for model_code, base_model in trained_on_models[detector].items():
            for dataset in dataset_names:
                results_path = f"./saved_training_logs_experiment_2/{detector}/{training_method}/fake_true_dataset_{base_model}_10k/{model_code}/test/test_metrics_fake_true_dataset_{dataset}_10k_{suffix}.json"

                with open(results_path, "r") as f:
                    result_dict = json.load(f)
                    result_dict["base_detector"] = detector
                    result_dict["trained_on_dataset"] = base_model
                    result_dict["detector"] = f"{detector}_{base_model}"
                    result_dict["dataset"] = dataset
                    results.append(result_dict)

    # transform to pandas dataframe
    results_df = pd.DataFrame(results)

    # order by detector and dataset
    results_df = results_df.sort_values(by=["trained_on_dataset", "detector", "dataset"])
    #results_df = results_df.sort_values(by=["detector", "dataset"])

    #display(results_df)

    return results_df

def add_test_logs_to_results_df(results_df, test_logs_dict, use_timestamp=False, use_eval_split=False, use_test_at_threshold=False):
    results = []
    detector = list(test_logs_dict.keys())[0]
    results_dict = test_logs_dict[detector]
    for timestamp, dataset in results_dict.items():
        if use_timestamp:
            if use_eval_split:
                results_path = f"./saved_training_logs_experiment_2/{detector}/{timestamp}/eval/eval_metrics_fake_true_dataset_{dataset}_10k.json"
            else:
                if use_test_at_threshold:
                    results_path = f"./saved_training_logs_experiment_2/{detector}/{timestamp}/test_at_threshold/test_metrics_fake_true_dataset_{dataset}_10k.json"
                else:
                    results_path = f"./saved_training_logs_experiment_2/{detector}/{timestamp}/test/test_metrics_fake_true_dataset_{dataset}_10k.json"
        else:
            if use_eval_split:
                results_path = f"./saved_training_logs_experiment_2/{detector}/eval/eval_metrics_fake_true_dataset_{dataset}_10k.json"
            else:
                if use_test_at_threshold:
                    results_path = f"./saved_training_logs_experiment_2/{detector}/test_at_threshold/test_metrics_fake_true_dataset_{dataset}_10k.json"
                else:   
                    results_path = f"./saved_training_logs_experiment_2/{detector}/test/test_metrics_fake_true_dataset_{dataset}_10k.json"
        
        with open(results_path, "r") as f:
            result_dict = json.load(f)
            result_dict["base_detector"] = detector
            result_dict["trained_on_dataset"] = "z"
            result_dict["detector"] = f"{detector}"
            result_dict["dataset"] = dataset
            results.append(result_dict)

    # transform to pandas dataframe
    results_additionnal_df = pd.DataFrame(results)

    # merge with the passed results_df
    results_df = pd.concat([results_df, results_additionnal_df])

    return results_df


def heatmap_from_df(results_df, metric="accuracy", with_std=False):
    fig, ax = plt.subplots(figsize=(10,10))  

    cmap_g_r = LinearSegmentedColormap.from_list('rg',["r", "y", "g"], N=256) 

    # if metric is the false positive rate, we want the colormap to be green for low values and red for high values
    if metric == "fp_rate":
        cmap_g_r = LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N=256)

    pivoted_results_df = results_df.pivot(index="detector", columns="dataset", values=metric)
    pivoted_results_df["base_model"] = pivoted_results_df.index.map(lambda x: "_".join(x.split("_")[:2]))
    pivoted_results_df["trained_on_dataset"] = pivoted_results_df.index.map(lambda x: x.split("_")[2])

    # sort by trained_on_dataset
    pivoted_results_df = pivoted_results_df.sort_values("trained_on_dataset", na_position="last")

    # put rows with trained_on_dataset = "" at the end
    #pivoted_results_df = pivoted_results_df.sort_values("trained_on_dataset", na_position="last")

    # remove the base_model and trained_on_dataset columns
    pivoted_results_df = pivoted_results_df.drop(columns=["base_model", "trained_on_dataset"])

    # 4x4 heatmap with the results where columns are the detectors trained on datasets and values are the accuracy
    heatmap = sns.heatmap(pivoted_results_df, annot=True, cmap=cmap_g_r, ax=ax)
    #heatmap = sns.heatmap(results_df.pivot(index="detector", columns="dataset", values="accuracy"), annot=True)
    #nb_detectors = len(results_df["detector"].unique())

    if with_std:
        nb_datasets = len(results_df["dataset"].unique())
        pivoted_results_error_df = results_df.pivot(index="detector", columns="dataset", values=f"std_{metric}")

        for i, detector in enumerate(results_df["detector"].unique()):
            for j, dataset in enumerate(results_df["dataset"].unique()):

                # get the correct postion
                pos = nb_datasets * i + j
                heatmap.texts[pos].set_text(f"{results_df[(results_df['detector'] == detector) & (results_df['dataset'] == dataset)][metric].values[0]:.2f} +/- {results_df[(results_df['detector'] == detector) & (results_df['dataset'] == dataset)][f"std_{metric}"].values[0]:.2f}")

        ## Set the text on the heatmap to add uncertainty
        #for i, detector in enumerate(results_df["detector"].unique()):
        #    for j, dataset in enumerate(results_df["dataset"].unique()):
    #
        #        # get the correct postion
        #        pos = nb_datasets * i + j
        #        heatmap.texts[pos].set_text(f"{results_df[(results_df['detector'] == detector) & (results_df['dataset'] == dataset)][metric].values[0]:.2f} +/- {results_df[(results_df['detector'] == detector) & (results_df['dataset'] == dataset)][f"std_{metric}"].values[0]:.2f}")

    plt.xlabel("Tested on")
    plt.ylabel("Trained on")

    ax.xaxis.tick_top()

    plt.title(f"{metric} of detectors on the different datasets")