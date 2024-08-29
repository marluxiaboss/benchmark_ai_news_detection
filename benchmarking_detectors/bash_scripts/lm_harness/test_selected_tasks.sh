cd lm-evaluation-harness
conda activate llm_detector

print_current_time () {
   current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
    echo $current_date_time;
}

lm_eval --model hf \
    --model_args meta-llama/Meta-Llama-3.1-8B,dtype="bfloat16" \
    --tasks leaderboard_mmlu_pro,ifeval,gsm8k  \
    --device cuda:0  \
    --batch_size 16 \
    --watermarking_scheme=no_watermark \
    --output_path saved_results

print_current_time

lm_eval --model hf \
    --model_args meta-llama/Meta-Llama-3.1-8B,dtype="bfloat16" \
    --tasks leaderboard_mmlu_pro,ifeval,gsm8k  \
    --device cuda:0  \
    --batch_size 16 \
    --watermarking_scheme=KGW \
    --output_path saved_results

print_current_time

lm_eval --model hf \
    --model_args meta-llama/Meta-Llama-3.1-8B,dtype="bfloat16" \
    --tasks leaderboard_mmlu_pro,ifeval,gsm8k  \
    --device cuda:0  \
    --batch_size 16 \
    --watermarking_scheme=KGW_P \
    --output_path saved_results

print_current_time

lm_eval --model hf \
    --model_args meta-llama/Meta-Llama-3.1-8B,dtype="bfloat16" \
    --tasks leaderboard_mmlu_pro,ifeval,gsm8k  \
    --device cuda:0  \
    --batch_size 8 \
    --watermarking_scheme=SWEET_P \
    --output_path saved_results

print_current_time

conda deactivate