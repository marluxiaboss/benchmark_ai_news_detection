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

trained_on_datasets=("phi" "gemma" "mistral" "round_robin")
tested_on_datasets=("phi" "gemma" "mistral" "round_robin" "gemma_chat" "zephyr" "llama3")

# same ordering as datasets i.e. fist path is for gpt2, second for phi,...
trained_detectors_path=("10_06_1026" "10_06_1030" "10_06_1034" "10_06_1037")

print_current_time () {
   current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
    echo $current_date_time;
}

train_method="adapter"
current_detector="distil_roberta-base"
echo testing with $current_detector
print_current_time
for i in ${!trained_detectors_path[@]}; do
    for j in ${!tested_on_datasets[@]}; do
        echo "Testing $current_detector trained on ${trained_on_datasets[$i]} with ${tested_on_datasets[$j]} dataset"
        print_current_time
        echo "trained on dataset: ${trained_on_datasets[$i]}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
        python src/train_detector.py --freeze_base=True --use_adapter=True --dataset_path=./fake_true_datasets/fake_true_dataset_${tested_on_datasets[$j]}_10k  --detector=$current_detector --evaluation=True --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${trained_on_datasets[$i]}_10k/${trained_detectors_path[$i]}/saved_models/best_model.pt
    done
done

trained_detectors_path=("10_06_1103" "10_06_1117" "10_06_1133" "10_06_1148")
current_detector="roberta_large"
echo testing with $current_detector
print_current_time
for i in ${!trained_detectors_path[@]}; do
    for j in ${!tested_on_datasets[@]}; do
        echo "Testing $current_detector trained on ${trained_on_datasets[$i]} with ${tested_on_datasets[$j]} dataset"
        print_current_time
        echo "trained on dataset: ${trained_on_datasets[$i]}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
        python src/train_detector.py --freeze_base=True --use_adapter=True --dataset_path=./fake_true_datasets/fake_true_dataset_${tested_on_datasets[$j]}_10k  --detector=$current_detector --evaluation=True --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${trained_on_datasets[$i]}_10k/${trained_detectors_path[$i]}/saved_models/best_model.pt
    done
done

trained_detectors_path=("10_06_1054" "10_06_1110" "10_06_1125" "10_06_1140")
current_detector="electra_large"
echo testing with $current_detector
print_current_time
for i in ${!trained_detectors_path[@]}; do
    for j in ${!tested_on_datasets[@]}; do
        echo "Testing $current_detector trained on ${trained_on_datasets[$i]} with ${tested_on_datasets[$j]} dataset"
        print_current_time
        echo "trained on dataset: ${trained_on_datasets[$i]}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
        python src/train_detector.py --freeze_base=True --use_adapter=True --dataset_path=./fake_true_datasets/fake_true_dataset_${tested_on_datasets[$j]}_10k  --detector=$current_detector --evaluation=True --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${trained_on_datasets[$i]}_10k/${trained_detectors_path[$i]}/saved_models/best_model.pt
    done
done

conda deactivate
