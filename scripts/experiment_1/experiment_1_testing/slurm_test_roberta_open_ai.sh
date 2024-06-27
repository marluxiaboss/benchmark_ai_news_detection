#!/bin/bash

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

tested_on_datasets=("phi" "gemma" "mistral" "round_robin" "gemma_chat" "zephyr" "llama3")
current_detector="roberta_base_open_ai"

echo testing with $current_detector
print_current_time
for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector with ${tested_on_datasets[$i]} dataset"
    print_current_time
    python src/train_detector.py  --dataset_path=./fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k  --detector=$current_detector --evaluation=True --experiment_path="saved_training_logs_experiment_2"
    sleep 1m
done

conda deactivate
