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

tested_on_datasets=("phi" "gemma" "mistral" "round_robin")

current_detector="roberta_base_open_ai"
echo testing with $current_detector
print_current_time
thresholds_roberta_open_ai=("0.9726407527923584" "0.9726407527923584" "0.9726407527923584" "0.9726407527923584")



for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector with ${tested_on_datasets[$i]} dataset"
    print_current_time
    classifier_threshold=${thresholds_roberta_open_ai[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py  --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --experiment_path="saved_training_logs_experiment_2"
    sleep 1m
done

# fast detect-gpt
current_detector="fast_detect_gpt"
echo testing with $current_detector
print_current_time
thresholds_roberta_fast_detect_gpt=("0.63" "0.63" "0.63")

for i in ${!tested_on_datasets[@]}; do
    echo Testing fast-detect-gpt on $dataset_name
    classifier_threshold=${thresholds_roberta_fast_detect_gpt[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/zero_shot_detector/test_fast_detect_gpt.py --reference_model_name=gpt-j-6B --classifier_threshold=$classifier_threshold --dataset_path=$dataset_path
done

conda deactivate

