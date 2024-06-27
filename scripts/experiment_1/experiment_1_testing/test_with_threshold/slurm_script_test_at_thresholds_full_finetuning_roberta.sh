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
trained_detectors_path=("10_06_1156" "10_06_1221" "10_06_1246" "10_06_1312")
current_detector="roberta_large"
train_method="full_finetuning"
echo testing with $current_detector

# detector trained on phi
print_current_time
thresholds_phi=("0.06297480314970016" "0.02794809266924858" "0.003424449823796749" "-0.3990408778190613" "0.025044936686754227" "-0.055254511535167694" "0.02113502286374569")
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
thresholds_gemma=("0.6218387484550476" "0.553473949432373" "0.6006136536598206" "0.44436612725257874" "0.5798153281211853" "0.507049560546875" "0.528086245059967")
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
thresholds_mistral=("0.006803194992244244" "0.006803194992244244" "0.006803194992244244" "-0.35557883977890015" "0.02354467287659645" "-0.127238929271698" "0.02354467287659645")
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
thresholds_round_robin=("0.33022233843803406" "0.33603039383888245" "0.3412937819957733" "0.004385998472571373" "0.3412937819957733" "0.17443764209747314" "0.30781668424606323")
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

