cd detector_benchmark
conda activate llm_detector

generators=("llama3_instruct_3")
#generators=("qwen2_chat_0_5B")
attacks=("generation_base" "prompt_attack" "low_temp_attack" "high_temp_attack" "repetition_penalty_attack" "prompt_paraphrasing_attack")
watermark_scheme="watermark_kgw_p"
detector="watermark_detector"

dataset_size=5000
#dataset_size=5
batch_size=128
gen_experiment_name="compare_watermark"
detection_experiment_name="compare_watermark"
test_res_dir="detection_test_results"

for i in ${!generators[@]}; do

    generator=${generators[$i]}

    for j in ${!attacks[@]}; do
        attack=${attacks[$j]}
        print_current_time
        echo "Detecting text generated with generator: $generator, watermark: $watermark_scheme, attack: $attack", dataset_size: $dataset_size, experiment_name: $experiment_name
        python test_detector.py detection=$detector generation=$attack generation.generator_name=$generator watermark=$watermark_scheme generation.experiment_name=$gen_experiment_name \
             detection.experiment_name=$detection_experiment_name  \
                detection.test_res_dir=$test_res_dir detection.batch_size=$batch_size             
    done
done

conda deactivate