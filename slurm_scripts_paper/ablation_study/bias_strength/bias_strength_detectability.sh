#!/bin/bash
#SBATCH --job-name=detect_sweet_p_watermark
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

bias_values=(0.5 1.0 2.0 5.0 10.0 20.0)

generator_name="llama_instruct_3.1"
#generators=("qwen2_chat_0_5B")
attack="generation_base"

watermark_scheme="watermark_sweet_p"
detector="watermark_detector"

batch_size=128
test_res_dir="detection_test_results"

target_fpr=0.01


for i in ${!bias_values[@]}; do
    watermark_delta=${bias_values[$i]}
    gen_experiment_name="bias_variation_bias_${watermark_delta}"
    detection_experiment_name=$gen_experiment_name

    print_current_time
    echo "Detecting text generated with generator: $generator_name, watermark: $watermark_scheme, attack: $attack", experiment_name: $experiment_name
    test_detector detection=$detector generation=$attack generation.generator_name=$generator_name watermark=$watermark_scheme generation.experiment_name=$gen_experiment_name \
            detection.experiment_name=$detection_experiment_name  \
            detection.test_res_dir=$test_res_dir detection.batch_size=$batch_size \
            detection.target_fpr=$target_fpr         
             
done

conda deactivate