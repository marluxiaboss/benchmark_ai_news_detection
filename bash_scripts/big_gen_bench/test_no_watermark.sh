cd prometheus-eval/BiGGen-Bench
conda activate big_gen_bench
python run_chat_inference.py --model_name "meta-llama/Meta-Llama-3.1-8B" --output_file_path "./outputs/chat_response.json"
python run_response_eval.py --model_name "prometheus-eval/prometheus-7b-v2.0" --input_file_path "./outputs/chat_response.json" --output_file_path "./feedback/evaluated.json"
conda deactivate