#!/bin/bash
#SBATCH --job-name=generate_watermark_sweet_p_datasets
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/benchmark_ai_news_detection
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Activate environment
eval "$(conda shell.bash hook)"
conda activate llm_detector

print_current_time () {
   current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
    echo $current_date_time;
}

attack="generation_base"
watermark_scheme="watermark_sweet"

generator_name="llama_instruct_3.1"
prefix_size=10
dataset_size=5000
batch_size=128
data_folder="data/generated_datasets"
skip_train_split=True
max_sample_len=500
max_new_tokens=220
min_new_tokens=200
temperature=0.8
top_p=0.95
repetition_penalty=1
do_sample=True
top_k=50

experiment_name="final_table_high_detectability_delta_2_5"
#delta=3.0
delta=2.5
#gamma=0.25
gamma=0.5
entropy_threshold=1.2


print_current_time
echo "Generating the dataset with generator: $generator_name, watermark: $watermark_scheme, attack: $attack", dataset_size: $dataset_size, experiment_name: $experiment_name

create_dataset generation=$attack watermark=$watermark_scheme generation.generator_name=$generator_name \
        generation.dataset_size=$dataset_size generation.experiment_name=$experiment_name \
        generation.prefix_size=$prefix_size \
        generation.skip_train_split=$skip_train_split generation.skip_cache=$skip_cache \
        generation.batch_size=$batch_size generation.data_folder=$data_folder \
        generation.max_sample_len=$max_sample_len generation.max_new_tokens=$max_new_tokens generation.min_new_tokens=$min_new_tokens \
        generation.temperature=$temperature generation.top_p=$top_p generation.repetition_penalty=$repetition_penalty \
        generation.do_sample=$do_sample generation.top_k=$top_k \
        watermark.delta=$delta watermark.gamma=$gamma watermark.entropy_threshold=$entropy_threshold

conda deactivate llm_detector