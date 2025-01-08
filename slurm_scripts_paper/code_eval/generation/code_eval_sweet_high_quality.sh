#!/bin/bash
#SBATCH --job-name=benchmark_bigcode_eval_sweet_p
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/benchmark_ai_news_detection/bigcode-evaluation-harness
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

#model_name="meta-llama/Meta-Llama-3.1-8B"
model_name="Qwen/Qwen2.5-Coder-7B"
task_names="humanevalplus"
#limit=100
max_length_generation=512
temperature=0.2
do_sample=True
n_samples=50
batch_size=64
precision=fp16
watermarking_scheme="SWEET_high_quality"
metric_output_path="code_evaluation_results/qwen/${watermarking_scheme}_results.json"
save_generations_path="code_evaluation_results/generations/qwen/generations_${watermarking_scheme}.json"

accelerate launch  main.py \
  --model $model_name \
  --tasks $task_names \
  --max_length_generation $max_length_generation \
  --temperature $temperature \
  --do_sample True \
  --n_samples $n_samples \
  --batch_size $batch_size \
  --precision $precision \
  --allow_code_execution \
  --save_generations \
  --watermarking_scheme $watermarking_scheme \
  --metric_output_path $metric_output_path \
  --save_generations_path $save_generations_path \
  --generation_only \
  #--limit $limit 

conda deactivate