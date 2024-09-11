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