cd bigcode-evaluation-harness
conda activate llm_detector

model_name="meta-llama/Meta-Llama-3.1-8B"
task_names="humanevalplus,mbppplus"
#limit=100
max_length_generation=512
temperature=0.2
do_sample=True
n_samples=50
batch_size=32
precision=fp16
watermarking_scheme="SWEET"
metric_output_path="evaluation_results/${watermarking_scheme}_results.json"
save_generations_path="generations/generations_${watermarking_scheme}.json"

accelerate launch  main.py \
  --model $model_name \
  --tasks $task_names \
  --max_length_generation $max_length_generation \
  --temperature $temperature \
  --do_sample True \
  --n_samples $n_samples \
  --batch_size $batch_size \
  --precision $precision \
  --allow_code_execution \
  --save_generations \
  --watermarking_scheme $watermarking_scheme \
  --metric_output_path $metric_output_path \
  --save_generations_path $save_generations_path \
  --generation_only \
  #--limit $limit 

conda deactivate