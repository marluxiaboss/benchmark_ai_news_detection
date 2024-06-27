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

#python generate_fake_true_dataset.py --generator=gemma_2b --batch_size=32 --experiment_name=gemma_10k --fake_dataset_size=10000 --access_token=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
python src/generate_fake_true_dataset.py --generator=llama3_instruct --batch_size=16 --experiment_name=llama3_10k --fake_dataset_size=10000 --prompt="Continue to write this news article:"
conda deactivate
