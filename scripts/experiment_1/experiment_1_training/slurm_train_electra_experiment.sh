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

detector="electra_large"
training_methods=("freeze_base" "adapter" "full_finetuning")
datasets=("phi" "gemma" "mistral" "round_robin")


echo Training $detector
for i in ${!training_methods[@]}; do
    for j in ${!datasets[@]}; do

        if [ ${training_methods[$i]} == "freeze_base" ]
        then
            learning_rate="1e-3"
            batch_size="64"
            freeze_base="True"
            use_adapter="False"
            eval_steps=500
            stop_after_n_samples=-1

        elif [ ${training_methods[$i]} == "adapter" ]
        then
            learning_rate="1e-4"
            batch_size="8"
            freeze_base="True"
            use_adapter="True"
            eval_steps=500
            stop_after_n_samples=-1

        elif [ ${training_methods[$i]} == "full_finetuning" ]
        then
            learning_rate="1e-5"
            batch_size="16"
            freeze_base="False"
            use_adapter="False"
            eval_steps=200
            stop_after_n_samples=-1
        fi 

        echo Training with training method ${training_methods[$i]} on ${datasets[$j]}
        print_current_time
        python src/train_detector.py --batch_size=$batch_size --num_epochs=1 --log_mode=online --freeze_base=$freeze_base --use_adapter=$use_adapter --save_dir=./saved_training_logs_experiment_2 --dataset_path=./fake_true_datasets/fake_true_dataset_${datasets[$j]}_10k --detector=$detector --eval_steps=$eval_steps --wandb_experiment_name=detector_training_experiment2 --learning_rate=$learning_rate
    done

done
conda deactivate
