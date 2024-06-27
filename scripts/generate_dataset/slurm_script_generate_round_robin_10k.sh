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

# note: batch size 512 is too much
python src/generate_round_robin_dataset.py --save_dir=fake_true_datasets
conda deactivate
