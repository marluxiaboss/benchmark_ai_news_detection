# Short LLM-Generated News Detection Still Falls Short 📰 :mag:

## File structure
<a name="file_structure"></a>

### 0. Model classes
- `src/generator.py`: wrapper arround any LLM used to generate text from a prompt
- `src/detector.py`: wrapper around any BERT type (bi-directional encoder) used to classify text as human or LLM generated
- `src/detector_trainer.py`: class for training and testing detectors (except Fast-DetectGPT)

### 1. Python scripts
- `src/generate_fake_true_dataset.py`: takes an LLM and generate responses using prompts from a given dataset
- `src/generate_round_robin_dataset.py`: used to generate the Round-Robin dataset from the other datasets
- `src/generate_fake_true_dataset_adversarial`: used to generate the adversarial datasets to escape the detector
- `src/format_out_of_domain_dataset.py`: used to format the Xsum dataset for testing the detectors on it
- `src/train_detector.py`: train a chosen detector on a dataset created by `generate_fake_true_dataset.py`, can also be used to test an already trained model
- `src/utils.py`
- `src/zero_shot_detector/test_fast_detect_gpt.py`: script to test Fast-DetectGPT

### 2. Jupyter notebooks

- `notebooks/main_paper_plots`: plots in the main part of the paper
- `notebooks/appendix_plots`: plots in the appendix part of the paper
- `notebooks/find_threshold`: used to find the appropriate thresholds for the target FPR

### 3. Training logs file structure

```
saved_training_logs  
│
└───{detector_name} (e.g. roberta_large)
    │
    └───{training_method} (e.g. freeze_base)
        │ 
        └───{dataset_name} (e.g. fake_true_dataset_gpt2_10k)
            │  
            └───{experiment_time} (e.g. 12_12_1212)
                │   args_logs.txt (training arguments)
                │   log.txt (terminal logs of the training)
                │ 
                └───eval
                │   eval_metrics_{dataset_name}.json (metrics on eval set to find thresholds)
                └───saved_models 
                │   best_model.pt (best model on eval set obtained during training)
                │ 
                └───test
                │   test_metrics_{dataset_name}.json (metrics on test set)
                │ 
                └───test_at_threshold
                    test_metrics_{dataset_name}.json (metrics on test set with a specific threshold)

```

### 4. Datasets
- `fake_true_datasets`: folder containing all the generated datasets  (need to run the scripts to have them)
- `fake_true_datasets/modifed_dataset` folder containing the adversarial version of the dataset above (need to run the scripts to have them)


## Reproducing the experiments and the plots

We provide scripts to run the experiments and get the data. We also provide notebooks to use this data to produce the plots in the paper.  
We also provide the data we obtained when running the experiments. You need to replace them to be able to re-run the experiments.

Results data (main paper): `saved_training_logs_experiment_2`  
Results data (degradation check in appendix): `saved_training_logs_experiment_1`

### Main paper part

#### Generate the data
Run all scripts in: `scripts/generate_dataset`  

#### Training the models
Run all scripts in: `scripts/experiment_1_training`  
Then you need to change the timestamps in the subsequent testing/threshold finding scripts according to the new trained models.

#### Finding the thresholds
Run: `scripts/experiment1/experiment_1_testing/test_with_threshold/slurm_script_find_thresholds_full_finetuning.sh`  
Run: `scripts/experiment2/test_with_threshold/slurm_script_find_thresholds.sh`  
Then run the cells in notebook: `notebooks/find_threshold.ipynb` and adjust the thresholds in the subsequent testing scripts.

#### Testing the detectors
Run all scripts in: `scripts/experiment_1/test_with_threshold` with the correct thresholds modified in the script  
Run all scripts in: `scripts/experiment_2/test_with_threshold` with the correct thresholds modified in the script  

#### Plots
Run all cells in notebook: `notebooks/main_paper_plots.ipynb` with the correct timestamps for the results (see log file structure)  

### Appendix part
You should run all scripts for the main paper part first.  

#### Degradation check
Run all scripts in: `scripts/experiment_1/check_degradation`  

#### Plots 
Run all cells in notebook: `notebooks/appendix_plots.ipynb`  







## Datasets used

- [CNN Dailymail Dataset](https://huggingface.co/datasets/cnn_dailymail?row=31) to create fake and true news samples
- [Xsum Dataset](https://huggingface.co/datasets/EdinburghNLP/xsum) to test the detectors on another news dataset
- [Fact completion](https://huggingface.co/datasets/Polyglot-or-Not/Fact-Completion?row=0) to test detector degradation