cd detector_benchmark
conda activate llm_detector

generators=("llama3_instruct_3")
#generators=("qwen2_chat_0_5B")
attacks=("generation_base" "prompt_attack" "low_temp_attack" "high_temp_attack" "repetition_penalty_attack" "prompt_paraphrasing_attack")
watermark_scheme="watermark_sweet"

dataset_size=5000
#dataset_size=5
batch_size=128
experiment_name="compare_watermark"
data_folder="data/generated_datasets"
skip_cache=True
skip_train_split=True


# specific to sweet/sweet_p
#entropy_threshold=1.0
entropy_threshold=0.9
watermark_delta=2.0

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
                watermark.entropy_threshold=$entropy_threshold watermark.delta=$watermark_delta
    done
done

conda deactivate