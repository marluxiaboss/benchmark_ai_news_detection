#!/bin/bash
#SBATCH --job-name=generate_fake_true_dataset_mistral
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/text_llm_detector
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
#python generate_fake_true_dataset.py --generator=mistral --batch_size=2 --experiment_name=mistral_10k --fake_dataset_size=10000
#python generate_fake_true_dataset.py --generator=mistral --batch_size=2 --experiment_name=mistral_10k --fake_dataset_size=10000
#python generate_fake_true_dataset.py --batch_size=16 --experiment_name=mistral_10k --fake_dataset_size=10000 --load_from_cache=True
python src/generate_fake_true_dataset.py --batch_size=16 --generator=mistral --experiment_name=mistral_10k --fake_dataset_size=10000
conda deactivate