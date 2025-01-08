cd detector_benchmark
conda activate llm_detector

#generators=("zephyr")
generators=("llama3_instruct_3.1")
#generators=("qwen2_chat_0_5B")
#attacks=("generation_base" "prompt_attack" "low_temp_attack" "high_temp_attack" "repetition_penalty_attack" "prompt_paraphrasing_attack")
attacks=("generation_base")

watermark_scheme="watermark_base"

dataset_size=5000
#dataset_size=5
batch_size=128
experiment_name="compare_watermark_len_100"
data_folder="data/generated_datasets"
skip_cache=True
skip_train_split=True

max_sample_len=100
max_new_tokens=50
min_new_tokens=40

for i in ${!generators[@]}; do

    generator=${generators[$i]}

    for j in ${!attacks[@]}; do
        attack=${attacks[$j]}
        print_current_time
        echo "Generating the dataset with generator: $generator, watermark: $watermark_scheme, attack: $attack", dataset_size: $dataset_size, experiment_name: $experiment_name
        python create_dataset.py generation=$attack watermark=$watermark_scheme generation.generator_name=$generator \
                generation.dataset_size=$dataset_size generation.experiment_name=$experiment_name \
                generation.skip_train_split=$skip_train_split generation.skip_cache=$skip_cache \
                generation.batch_size=$batch_size generation.data_folder=$data_folder \
                generation.max_sample_len=$max_sample_len generation.max_new_tokens=$max_new_tokens generation.min_new_tokens=$min_new_tokens
                
    done
done

conda deactivate