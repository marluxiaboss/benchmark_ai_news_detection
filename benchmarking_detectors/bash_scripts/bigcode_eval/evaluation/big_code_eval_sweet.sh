cd bigcode-evaluation-harness
task_names="humanevalplus,mbppplus"

watermarking_scheme="SWEET"

task="humanevalplus"
generation_file="generations/generations_${watermarking_scheme}_${task}.json"
metric_output_path="evaluation_results/${watermarking_scheme}_${task}_results.json"

accelerate launch  main.py \
    --tasks $task \
    --allow_code_execution \
    --load_generations_path $generation_file \
    --metric_output_path $metric_output_path
    #--model incoder-temperature-08

task="mbppplus"
generation_file="generations/generations_${watermarking_scheme}_${task}.json"
metric_output_path="evaluation_results/${watermarking_scheme}_${task}_results.json"

accelerate launch  main.py \
    --tasks $task \
    --allow_code_execution \
    --load_generations_path $generation_file \
    --metric_output_path $metric_output_path
    #--model incoder-temperature-08