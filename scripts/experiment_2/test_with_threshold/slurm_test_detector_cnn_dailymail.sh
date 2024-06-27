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


# ADAPTING DATASET FORMAT
python src/format_out_of_domain_dataset.py --save_path=fake_true_datasets/cnn_dailymail_true_only_test --original_dataset_path=fake_true_datasets/fake_true_dataset_zephyr_10k_test.json --dataset_name=cnn_dailymail_only_true --take_samples=10000 --orig_dataset_type=json

dataset_path="fake_true_datasets/cnn_dailymail_true_only_test"

# TESTING DETECTORS
trained_on_datasets=("mistral" "round_robin")

# same ordering as datasets i.e. fist path is for gpt2, second for phi,...
trained_detectors_path=("10_06_1242" "10_06_1308")
current_detector="electra_large"
train_method="full_finetuning"
echo testing with $current_detector

# testing with electra_mistral
print_current_time
threshold_mistral="0.10198206454515457"
trained_detector=${trained_detectors_path[0]}
training_dataset=${trained_on_datasets[0]}

echo "Testing $current_detector trained on $training_dataset with ${tested_on_datasets[$i]} dataset"
print_current_time
classifier_threshold=$threshold_mistral
python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt

# testing with electra_round_robin
print_current_time
threshold_round_robin="0.17757119238376617"
trained_detector=${trained_detectors_path[1]}
training_dataset=${trained_on_datasets[1]}

echo "Testing $current_detector trained on ${training_dataset} with ${tested_on_datasets[$j]} dataset"
print_current_time
echo "trained on dataset: ${training_dataset}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
classifier_threshold=$threshold_round_robin
python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt


current_detector="roberta_base_open_ai"
echo testing with $current_detector
print_current_time
threshold_roberta_open_ai="1.009648323059082"

echo "Testing $current_detector with ${tested_on_datasets[$i]} dataset"
print_current_time
classifier_threshold=$threshold_roberta_open_ai
python src/train_detector.py  --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --experiment_path="saved_training_logs_experiment_2"
sleep 1m

# fast detect-gpt
current_detector="fast_detect_gpt"
echo testing with $current_detector
print_current_time
threshold_fast_detect_gpt="0.63"

echo Testing fast-detect-gpt on $dataset_name
classifier_threshold=$threshold_fast_detect_gpt
python src/zero_shot_detector/test_fast_detect_gpt.py --reference_model_name=gpt-j-6B --classifier_threshold=$classifier_threshold --dataset_path=$dataset_path

conda deactivate
