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

trained_on_datasets=("phi" "gemma" "mistral" "round_robin")
tested_on_datasets=("gemma_chat" "zephyr" "llama3")

# same ordering as datasets i.e. fist path is for gpt2, second for phi,...
trained_detectors_path=("10_06_1146" "10_06_1215" "10_06_1242" "10_06_1308")

current_detector="electra_large"
train_method="full_finetuning"
echo testing with $current_detector
print_current_time
for i in ${!trained_detectors_path[@]}; do
    for j in ${!tested_on_datasets[@]}; do
        echo "Testing $current_detector trained on ${trained_on_datasets[$i]} with ${tested_on_datasets[$j]} dataset"
        print_current_time
        echo "trained on dataset: ${trained_on_datasets[$i]}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
        python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=./fake_true_datasets/fake_true_dataset_${tested_on_datasets[$j]}_10k --detector=$current_detector --evaluation=True --use_eval_set --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${trained_on_datasets[$i]}_10k/${trained_detectors_path[$i]}/saved_models/best_model.pt
    done
done


current_detector="roberta_base_open_ai"

echo testing with $current_detector
print_current_time
for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector with ${tested_on_datasets[$i]} dataset"
    print_current_time
    python src/train_detector.py  --dataset_path=./fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k  --detector=$current_detector --evaluation=True --use_eval_set --experiment_path="saved_training_logs_experiment_2"
done

tested_on_datasets=("phi" "gemma" "mistral" "round_robin" "gemma_chat" "zephyr" "llama3")
echo testing with fast detect gpt
print_current_time

for i in ${!tested_on_datasets[@]}; do
    dataset_name=fake_true_dataset_${tested_on_datasets[$i]}_10k
    echo Testing fast-detect-gpt on $dataset_name
    python src/zero_shot_detector/test_fast_detect_gpt.py --reference_model_name=gpt-j-6B --dataset_path=fake_true_datasets/$dataset_name --use_eval_set
done

conda deactivate
