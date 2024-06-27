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

# TESTING DETECTORS
trained_on_datasets=("mistral" "round_robin")
tested_on_datasets=("gemma_chat" "zephyr" "llama3")

# same ordering as datasets i.e. fist path is for gpt2, second for phi,...
trained_detectors_path=("10_06_1242" "10_06_1308")
current_detector="electra_large"
train_method="full_finetuning"
echo testing with $current_detector

# testing with electra_mistral
print_current_time
thresholds_mistral=("0.19038350880146027" "0.10198206454515457" "0.19038350880146027")
trained_detector=${trained_detectors_path[0]}
training_dataset=${trained_on_datasets[0]}

for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector trained on $training_dataset with ${tested_on_datasets[$i]} dataset"
    print_current_time
    classifier_threshold=${thresholds_mistral[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt
done

# testing with electra_round_robin
print_current_time
thresholds_round_robin=("0.4068852365016937" "0.17757119238376617" "0.290040522813797")
trained_detector=${trained_detectors_path[1]}
training_dataset=${trained_on_datasets[1]}

for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector trained on ${training_dataset} with ${tested_on_datasets[$j]} dataset"
    print_current_time
    echo "trained on dataset: ${training_dataset}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
    classifier_threshold=${thresholds_round_robin[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt
done


current_detector="roberta_base_open_ai"
echo testing with $current_detector
print_current_time
thresholds_roberta_open_ai=("0.9307770729064941" "1.009648323059082" "0.9726407527923584")

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
thresholds_fast_detect_gpt=("0.63" "0.63" "0.63")

for i in ${!tested_on_datasets[@]}; do
    echo Testing fast-detect-gpt on $dataset_name
    classifier_threshold=${thresholds_fast_detect_gpt[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/zero_shot_detector/test_fast_detect_gpt.py --reference_model_name=gpt-j-6B --classifier_threshold=$classifier_threshold --dataset_path=$dataset_path
done

conda deactivate

