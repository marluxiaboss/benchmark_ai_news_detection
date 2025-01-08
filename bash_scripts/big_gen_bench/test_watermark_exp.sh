cd prometheus-eval/BiGGen-Bench
conda activate big_gen_bench
python run_chat_inference.py --model_name "meta-llama/Meta-Llama-3.1-8B" --output_file_path "./outputs/chat_response_exp.json" --watermarking_scheme "EXP" --batch_size 8
python run_response_eval.py --model_name "prometheus-eval/prometheus-7b-v2.0" --input_file_path "./outputs/chat_response_exp.json" --output_file_path "./feedback/evaluated_exp.json" 
conda deactivate