# How to generate the benchmark

cf. `bash_scripts/generating_datasets/generate_no_watermark_dataset.sh` (see other scripts in `bash_scripts/generating_datasets` for other examples)

## How to generate the benchmark without watermarking

```sh
attack="generation_base"

generator_name="llama3_instruct_3.1"
prefix_size=10
dataset_size=5000
batch_size=128
experiment_name="test"
data_folder="data/generated_datasets"
skip_train_split=True
max_sample_len=100
max_new_tokens=50
min_new_tokens=40
temperature=0.8
top_p=0.95
repetition_penalty=1
do_sample=True
top_k=50

python create_dataset.py generation=$attack watermark=$watermark_scheme generation.generator_name=$generator \
        generation.dataset_size=$dataset_size generation.experiment_name=$experiment_name \
        generation.prefix_size=$prefix_size \
        generation.skip_train_split=$skip_train_split generation.skip_cache=$skip_cache \
        generation.batch_size=$batch_size generation.data_folder=$data_folder \
        generation.max_sample_len=$max_sample_len generation.max_new_tokens=$max_new_tokens generation.min_new_tokens=$min_new_tokens \
        generation.temperature=$temperature generation.top_p=$top_p generation.repetition_penalty=$repetition_penalty \
        generation.do_sample=$do_sample generation.top_k=$top_k
```


```{admonition} Configure the benchmark
:class: hint

This section outlines the parameters used for generating datasets with the specified configurations. See the configuration files under `conf/generation` for the complete list.
Note that all default values assume that attack="generation_base" which means that no attack is used.

- `attack`: Sets the base hydra configuration file for the generation used to generate the fake samples (base parameters). Here, "generation_base" corresponds to the base file which means no attack is used. This will determine the default parameter values. See [here](description_lists/supported_attacks.md) for the list of supported attacks and [here](how_to_add/how_to_add_attack.md) to add yours. Default value: "generation_base"

- `batch_size`: Specifies the number of samples to be generated in parallel by the GPU. Default value: 2

- `data_folder`: The directory where the generated datasets will be stored. Here, it is set to "data/generated_datasets". Default value: "data/generated_datasets"

- `dataset_name`: Base dataset used for the true samples and the prefixes Default value: cnn_dailymail

- `dataset_size`: Sets the total number of samples to be generated. This includes the train, eval and test split (80/10/10 split). This means that for dataset_size=5000, there will be a test split of size 500. Default value: 100

- `do_sample`: Whether to use top_p sampling or greedy decoding. Default value: True

- `experiment_name`: A string that names the experiment. In this case, it is set to "test", which can be useful for tracking results. Default value: base

- `generator_name`: Specifies the name of the generator to be used, in this case, "llama3_instruct_3.1". This determines the model that will generate the text. See [here](description_lists/supported_generators.md) for the list of supported generators and [here](how_to_add/how_to_add_detector.md) to add yours. Default value: qwen2_chat_0_5B

- `max_new_tokens`: The maximum number of new tokens to generate. Default value: 220

- `max_sample_len`: The maximum length of each of the fake/true samples in number of characters. All samples fake or true are cut to this value. Default value: 500

- `min_new_tokens`: The minimum number of new tokens to generate, set to 40. This ensures that each generated sample has a minimum length. Default value: 200

- prefix_size: Number of first words to take from the true samples that will be forced into the fake samples to start the generation. Default value: 10

- `repetition_penalty=1`: Repetion value used for the generation. Default value: 1

- `skip_train_split`: A boolean parameter that, when set to True, indicates that the training split of the dataset should be skipped. It will still create a train split, but the fake samples will be empty. Note that there is an eval split used for finding the correct threshold for a given target FPR and the test split used to test the detector on that threshold. The train split could be used to train a detector on that dataset. Default value: False

- `temperature`: Temperature used for the generation. Default value: 0.8

- `top_k`: top k value used for the generation. Default value: 50

- `top_p`: top p value used for the generation. Default value: 0.95
```

## How to generate the benchmark with watermarking

```sh
attack="generation_base"

generator_name="llama3_instruct_3.1"
watermark_scheme="watermark_base"
prefix_size=10
dataset_size=5000
batch_size=128
experiment_name="test"
data_folder="data/generated_datasets"
skip_train_split=True
max_sample_len=100
max_new_tokens=50
min_new_tokens=40
temperature=0.8
top_p=0.95
repetition_penalty=1
do_sample=True
top_k=50

python create_dataset.py generation=$attack watermark=$watermark_scheme generation.generator_name=$generator \
        generation.dataset_size=$dataset_size generation.experiment_name=$experiment_name \
        generation.prefix_size=$prefix_size \
        generation.skip_train_split=$skip_train_split generation.skip_cache=$skip_cache \
        generation.batch_size=$batch_size generation.data_folder=$data_folder \
        generation.max_sample_len=$max_sample_len generation.max_new_tokens=$max_new_tokens generation.min_new_tokens=$min_new_tokens \
        generation.temperature=$temperature generation.top_p=$top_p generation.repetition_penalty=$repetition_penalty \
        generation.do_sample=$do_sample generation.top_k=$top_k
```

```{admonition} Configure the benchmark
:class: hint

The parameters here are the same as above except for the following:

- `watermark_scheme`: Indicates the hydra config file for the watermarking scheme to be applied when generating the text. "watermark_base" means no watermark is used (base generation). See [here](description_lists/supported_watermarks.md) for the full list of watermarking schemes. Default value: "watermark_base".

``` 

## Configure the watermark:
- **TODO**


## Extra parameters depending on the attack:
- **TODO**


````