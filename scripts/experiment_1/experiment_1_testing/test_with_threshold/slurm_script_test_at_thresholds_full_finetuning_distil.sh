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

trained_on_datasets=("phi" "gemma" "mistral" "round_robin")
tested_on_datasets=("phi" "gemma" "mistral" "round_robin" "gemma_chat" "zephyr" "llama3")

# DISTIL ROBERTA BASE

# same ordering as datasets i.e. fist path is for gpt2, second for phi,...
trained_detectors_path=("10_06_1040" "10_06_1047" "10_06_1054" "10_06_1100")
current_detector="distil_roberta-base"
train_method="full_finetuning"
echo testing with $current_detector

# detector trained on phi
print_current_time
thresholds_phi=("0.0540054589509964" "0.004177845548838377" "-0.016925202682614326" "0.16298839449882507" "0.03146934509277344" "-0.03668294847011566" "0.03146934509277344")
trained_detector=${trained_detectors_path[0]}
training_dataset=${trained_on_datasets[0]}

for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector trained on $training_dataset with ${tested_on_datasets[$i]} dataset"
    print_current_time
    classifier_threshold=${thresholds_phi[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt
done

# detector trained on gemma
print_current_time 
thresholds_gemma=("0.26239141821861267" "0.27046048641204834" "0.27046048641204834" "0.16181477904319763" "0.27046048641204834" "0.16181477904319763" "0.27046048641204834")
trained_detector=${trained_detectors_path[1]}
training_dataset=${trained_on_datasets[1]}

for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector trained on ${training_dataset} with ${tested_on_datasets[$j]} dataset"
    print_current_time
    echo "trained on dataset: ${training_dataset}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
    classifier_threshold=${thresholds_gemma[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt
done


# detector trained on mistral
print_current_time
thresholds_mistral=("0.38079580664634705" "0.2950550615787506" "0.4447759687900543" "0.34728097915649414" "0.4447759687900543" "0.3131090998649597" "0.4473596513271332")
trained_detector=${trained_detectors_path[2]}
training_dataset=${trained_on_datasets[2]}

for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector trained on ${training_dataset} with ${tested_on_datasets[$j]} dataset"
    print_current_time
    echo "trained on dataset: ${training_dataset}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
    classifier_threshold=${thresholds_mistral[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt
done

# detector trained on round_robin
print_current_time
thresholds_round_robin=("0.40544983744621277" "0.43468260765075684" "0.444585919380188" "0.1273074895143509" "0.444585919380188" "0.13705286383628845" "0.444585919380188")
trained_detector=${trained_detectors_path[3]}
training_dataset=${trained_on_datasets[3]}

for i in ${!tested_on_datasets[@]}; do
    echo "Testing $current_detector trained on ${training_dataset} with ${tested_on_datasets[$j]} dataset"
    print_current_time
    echo "trained on dataset: ${training_dataset}, model code: ${trained_detectors_path[$i]}, tested on dataset: ${tested_on_datasets[$j]}" 
    classifier_threshold=${thresholds_round_robin[$i]}
    dataset_path="fake_true_datasets/fake_true_dataset_${tested_on_datasets[$i]}_10k"
    python src/train_detector.py --freeze_base=True --use_adapter=False --dataset_path=$dataset_path  --detector=$current_detector --evaluation=True --classifier_threshold=$classifier_threshold --model_path=./saved_training_logs_experiment_2/${current_detector}/${train_method}/fake_true_dataset_${training_dataset}_10k/${trained_detector}/saved_models/best_model.pt
done

conda deactivate

