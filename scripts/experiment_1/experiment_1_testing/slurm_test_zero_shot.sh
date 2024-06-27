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
tested_on_datasets=("phi" "gemma" "mistral" "round_robin" "gemma_chat" "zephyr" "llama3")

for i in ${!tested_on_datasets[@]}; do
    dataset_name=fake_true_dataset_${tested_on_datasets[$i]}_10k
    echo Testing fast-detect-gpt on $dataset_name
    python src/zero_shot_detector/test_fast_detect_gpt.py --reference_model_name=gpt-j-6B --dataset_path=fake_true_datasets/$dataset_name
done

conda deactivate
