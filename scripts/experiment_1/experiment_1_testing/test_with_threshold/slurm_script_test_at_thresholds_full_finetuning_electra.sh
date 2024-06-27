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
trained_detectors_path=("10_06_1146" "10_06_1215" "10_06_1242" "10_06_1308")
current_detector="electra_large"
train_method="full_finetuning"
echo testing with $current_detector

# detector trained on phi
print_current_time
thresholds_phi=("-1.0533567667007446" "-1.106029748916626" "-1.0533567667007446" "-1.74638831615448" "-1.0533567667007446" "-1.308728814125061" "-1.0892279148101807")
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
thresholds_gemma=("-0.44769975543022156" "-0.5850827097892761" "-0.6413555145263672" "-1.6195952892303467" "-0.5850827097892761" "-0.7194408774375916" "-0.5809829831123352")
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
thresholds_mistral=("0.20634852349758148" "0.10198206454515457" "0.19038350880146027" "-0.17242999374866486" "0.19038350880146027" "0.10198206454515457" "0.19038350880146027")
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
thresholds_round_robin=("0.4765685498714447" "0.39848220348358154" "0.4068852365016937" "0.10936053097248077" "0.4068852365016937" "0.17757119238376617" "0.290040522813797")
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

