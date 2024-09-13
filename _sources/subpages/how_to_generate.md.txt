# How to generate


```
generators="llama3_instruct_3.1"
attack="generation_base"
watermark_scheme="watermark_base"
dataset_size=5000
batch_size=128
experiment_name="test"
data_folder="data/generated_datasets"
skip_train_split=True

max_sample_len=100
max_new_tokens=50
min_new_tokens=40

python create_dataset.py generation=$attack watermark=$watermark_scheme generation.generator_name=$generator \
        generation.dataset_size=$dataset_size generation.experiment_name=$experiment_name \
        generation.skip_train_split=$skip_train_split generation.skip_cache=$skip_cache \
        generation.batch_size=$batch_size generation.data_folder=$data_folder \
        generation.max_sample_len=$max_sample_len generation.max_new_tokens=$max_new_tokens generation.min_new_tokens=$min_new_tokens
        
```

```{admonition} Configure the benchmark
:class: hint

This section outlines the parameters used for generating datasets with the specified configurations.

- `generators`: Specifies the name of the generator to be used, in this case, "llama3_instruct_3.1". This determines the model that will generate the text. See [here](description_lists/supported_generators.md) for the list of supported generators and [here](how_to_add/how_to_add_detector.md) to add yours.

- `attack`: Defines the type of attack to be simulated during the generation process. Here, "generation_base" indicates a baseline generation attack. 

- `watermark_scheme`: Indicates the watermarking scheme to be applied to the generated text. "watermark_base" is the chosen scheme for this experiment.

- `dataset_size`: Sets the total number of samples to be generated. A value of 5000 means that the dataset will consist of 5000 generated samples.

- `batch_size`: Specifies the number of samples to be processed in one batch during generation. A batch size of 128 allows for efficient processing of the data.

- `experiment_name`: A string that names the experiment. In this case, it is set to "test", which can be useful for tracking results.

- `data_folder`: The directory where the generated datasets will be stored. Here, it is set to "data/generated_datasets".

- `skip_train_split`: A boolean parameter that, when set to True, indicates that the training split of the dataset should be skipped.

- `max_sample_len`: The maximum length of each generated sample, set to 100 tokens in this case.

- `max_new_tokens`: The maximum number of new tokens to generate, which is set to 50.

- `min_new_tokens`: The minimum number of new tokens to generate, set to 40. This ensures that each generated sample has a minimum length.

These parameters are crucial for controlling the behavior of the dataset generation process and ensuring that the generated data meets the requirements of the experiment.

```