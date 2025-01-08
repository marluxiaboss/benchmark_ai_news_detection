#!/bin/bash
#SBATCH --job-name=detect_sweet_p_watermark
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

prob_ratios=(0.00001 0.05 0.1 0.25 0.35 0.5)

generator_name="llama_instruct_3.1"
#generators=("qwen2_chat_0_5B")
attack="generation_base"

watermark_scheme="watermark_sweet_p"
detector="watermark_detector"

batch_size=128
test_res_dir="detection_test_results"

for i in ${!prob_ratios[@]}; do
    prob_ratio=${prob_ratios[$i]}
    gen_experiment_name="prob_ratio_variation_${prob_ratio}"
    detection_experiment_name=$gen_experiment_name

    print_current_time
    echo "Detecting text generated with generator: $generator_name, watermark: $watermark_scheme, attack: $attack", experiment_name: $gen_experiment_name
    test_detector detection=$detector generation=$attack generation.generator_name=$generator_name watermark=$watermark_scheme generation.experiment_name=$gen_experiment_name \
            detection.experiment_name=$detection_experiment_name  \
            detection.test_res_dir=$test_res_dir detection.batch_size=$batch_size             
done

conda deactivate