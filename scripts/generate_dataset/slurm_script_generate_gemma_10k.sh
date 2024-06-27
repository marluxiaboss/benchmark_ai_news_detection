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

python src/generate_fake_true_dataset.py --generator=gemma_2b --batch_size=64 --experiment_name=gemma_10k --fake_dataset_size=10000 --access_token=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
conda deactivate
