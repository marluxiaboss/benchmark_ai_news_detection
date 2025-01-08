#!/bin/bash
#SBATCH --job-name=detect_sweet_p_watermark_paraphrasing
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/benchmark_ai_news_detection/
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

generator_name="llama_instruct_3.1"
attack="prompt_paraphrasing_attack"
watermark_scheme="watermark_sweet"
detector="watermark_detector"
batch_size=128
test_res_dir="detection_test_results"
gen_experiment_name="robustness_test_paraphrasing"

target_fpr=0.05
detection_experiment_name="final_table_tpr_at_5_high_detectability_delta_2_5"

print_current_time
echo "Detecting text generated with generator: $generator_name, watermark: $watermark_scheme, attack: $attack", experiment_name: $experiment_name
test_detector detection=$detector generation=$attack generation.generator_name=$generator_name watermark=$watermark_scheme generation.experiment_name=$gen_experiment_name \
        detection.experiment_name=$detection_experiment_name  \
        detection.test_res_dir=$test_res_dir detection.batch_size=$batch_size \
        detection.target_fpr=$target_fpr         

conda deactivate