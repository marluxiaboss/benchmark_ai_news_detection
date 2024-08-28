# LLM-generated news benchmark  :mag: 📰
The purpose of this benchmark is to evaluate LLM detectors, especially against evasion attacks. So far, the benchmark is based on the detection of short LLM-generated news articles, but it can be extended to cover different detection tasks.  
The main consideration is to make the benchmark easy to extend with different datasets, detectors and evasion attacks

## Features
- Detectors and watermark detection benchmarking (adversarial + non-adversarial)
- LLM-generated watermarked text quality benchmarking
- Modularity: possible to add new datasets, detectors, attacks and watermarking schemes without much effort

## Repo structure

### 0. Class folders
- `detector_benchmark/dataset_loader`: folder for the dataset_loader classes
- `detector_benchmark/detector`: folder for the detector classes and detector loader
- `detector_benchmark/generation`: folder for the bases generation + adversarial generation classes and also the generator loader
- `detector_benchmark/pipeline`: folder for the pipeline classes (text generation, testing detection and evaluating generated text quality)
- `detector_benchmark/text_quality_evaluation`: folder for text quality evaluator classes

### 1. Config files
Configuration files (hydra configuration) are located in:
- `detector_benchmark/conf/` for the detection, generation and pipeline configurations.

### 2. Python scripts
- `detector_benchmark/create_dataset.py`: script to create a dataset using a dataset loader and a generation config
- `detector_benchmark/test_detector.py`: script to test a detector on a dataset created using the script above
- `detector_benchmark/test_text_quality.py`: script to run basic text quality evaluation on generated text (non-watermarked, watermarked or even human written)

### 3. External libraries submodules
Some existing github repositories were forked and adapted to work with watermarking, namely:
- `lm-evaluation-harness/` from EleutherAI that supports a wide range of benchmarking tasks. 
- `bigcode-evaluation-harness/` from BigCode that supports a wide range of code benchmarking tasks. 
- `prometheus-eval/BiGGen-Bench/` from the authors of the following paper <https://arxiv.org/abs/2406.05761> which benchmarks LLMs with an open source LLM-as-Judge.

Our fork of the respecitives repos allows to run theirs benchmark with text generated with a watermarking scheme. See [in the following section](#evaluation-with-lm-harness) more details about how to run the benchmarks from the external libraries with a watermaking scheme.

### 4. Bash scripts

### 5. Jupyter notebooks
- `big_gen_bench_res.ipynb` to visualize the results after running the benchmark from BiGGen-Bench with different watermarking schemes
- `lm_harness_bench_res.ipynb` to visualize the results after running the benchmark from lm-harness with different watermarking schemes see [the following section](#evaluation-with-lm-harness) for the tasks to run in lm-harness
- `watermark_detection_results.ipynb` to visualize the detection results after having run the script `detector_benchmark/test_detector.py`
- `bigcode_eval_res.ipnyb` TODO

### 6. Data folders
- `detector_benchmark/data/generated_datasets` for the datasets adversarial + non-adversarial datasets generated with the LLMs with or without watermarking. The arboresence is as follows:

```
generated_datasets  
│
└───{source_dataset} (e.g. cnn_dailymail)
    │
    └───{adversarial_attack} (e.g. no_attack)
        │ 
        └───{watermarking_scheme} (e.g. no_watermark)
            │  
            └───log 
            │   log.txt (terminal logs)
            │
            │   
            └───{generator_name}_{experiment_name} (actual generated hugginface dataset)

```
Where `source_dataset` is the name of the dataset used to obtain the true human written samples and the prefixes for the fake AI-written samples, `adversarial_attack` and `watermarking_scheme` are the respective attack and watermark used to generate the datasets. `experiment_name` is the name used for the specific run of the generation.

### 7. Results folders
Here are the following folders used to save the results from the scripts/benchmarks:

- `detector_benchmark/detection_test_results` containing the results when running the detection script `detector_benchmark/test_detector.py`.
- `detector_benchmark/text_quality_eval_results` containing the results when running the text quality evaluation script `detector_benchmark/test_text_quality.py`
- TODO: add info about the respective results folder for the external libraries


## Reproducing the experiments and the plots


## Running the python scripts + external libraries benchmarks
See the corresponding bash script in [the bash script section](#4-bash-scripts) for examples.

## Adding a watermarking scheme, attack, detector or dataset

### How to add a watermarking scheme on the benchmark
Credits to <https://github.com/THU-BPM/MarkLLM> for most of the watermarking code structure.


### How to test a new detector on the detection benchmark
To add a new detectors, 3 files needs to be added/modified:
- add: a .py file in `detector_benchmark/detector` containing the class for the new detector, inheriting the base detector class.
- add: a configuration file under `detector_benchmark/conf/detection` to configure the new detector.
- modify: `detector_benchmark/detector/detector_loader.py` to be able to load the detector from the .py file created above.


The added detector class should have at least a constructor and a `detect` function with the following signature:
```
def detect(self, texts: list, batch_size: int, detection_threshold: float) -> tuple[list[int], list[float], list[int]]:
```
Where `texts` is a list of text to be detected as LLM-written or not and it should return the following variables:
- `preds`: a list of 0s (human-written) and 1s (LLM-generated) with the predicted labels computed as argmax of the logits of both classes
- `logits_pos_class`: the list of logits or probabilities for the positive class (softmaxed or not) 
- `preds_at_threshold`: same as `preds` but where the prediction is made using a threshold on the logits rather than argmax  

You can look at already added detectors for guiding examples.

### How to add an attack
To add an attack, 3 files needs to be added/modified:
- add: a .py file in `generation` containing the class for the attack, inheriting the base ArticleGenerator class
- add: a configuration file under `detector_benchmark/conf/generation` to configure the new attack.
- modify: `detector_benchmark/generation/attack_loader.py` to be able to load the attack.

The added detector class should have at least a constructor and a `generate_adversarial_text` function with the following signature:
```
def generate_adversarial_text(self, prefixes: list[str], batch_size: int) -> list[str]:
```
Where `prefixes` is a list of inputs text to continue for the generation (see how the dataset samples look like) and it should return the list of generated text which is the continuation of the prefixes.

You can look at already added detectors for guiding examples.

Note: for any attack that involves either using a specific prompt for the generation or modifying a prompt parameter, the already existing attacks in `detector_benchmark/generation/gen_params_attack.py` and `detector_benchmark/generation/prompt_attack.py` can be used for this purpose by only modifying the related configuration file in `conf`.

### How to add a dataset
First, let's describe the datasets format used for testing the detectors. The datasets are huggingface Datasets where each samples has the following fields:
- `label`: the label of the text (0 for human written, 1 for AI-written)
- `text`: contains the full text of the sample
- `prefix`: prefix from the human written texts used to generated the AI-written text. For each prefix, we always have the corresponding true sample (label 0) and the fake one (label 1) sharing the same prefix.
- `generation_config`: the config (data from the config file in `conf`) used to generate the text AI-generated text for the dataset
- `watermark_config`: the watermarking config used to generate the text (which watermarking algorithm if any,...)

Now to add a different dataset than the existing ones, we need to add dataset loader class inheriting the base `FakeTruePairsDataLoader` class inside the `dataset_loader` folder.
This class should implement the `load_data` function with the following signature:
```
def load_data(self) -> DatasetDict:
```
This function should return a DatasetDict (huggingface dataset format) with a train, eval and test split and respecting the dataset format. 
To see how to apply the correct format to the dataset, see the existing dataset loaders.

Note: the load_data returns a dataset where the fake samples (label 1) have an empty `text` and no `generation_config` nor `watermark_config` fields since the AI texts have not been generated yet. The only fields that should be fully field for AI texts are the label and the prefix.


## Class descriptions


## Future work


