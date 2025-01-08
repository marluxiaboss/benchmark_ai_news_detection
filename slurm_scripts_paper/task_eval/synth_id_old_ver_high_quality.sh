#!/bin/bash
#SBATCH --job-name=task_eval_synth_id_old
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/benchmark_ai_news_detection/lm-evaluation-harness
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Activate environment
eval "$(conda shell.bash hook)"

conda activate llm_detector_transformers_4_46_0

print_current_time () {
   current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
    echo $current_date_time;
}
watermarking_scheme=SynthID_high_quality

print_current_time
#--tasks leaderboard_mmlu_pro,ifeval,gsm8k  \
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype="bfloat16" \
    --tasks gsm8k \
    --device cuda:0  \
    --batch_size 32 \
    --watermarking_scheme=$watermarking_scheme \
    --output_path "saved_results/eval_$watermarking_scheme" \
    --apply_chat_template \
    --num_fewshot 8 \
    --fewshot_as_multiturn \
    --log_samples \

conda deactivate