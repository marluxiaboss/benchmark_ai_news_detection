#!/bin/bash
#SBATCH --job-name=text_quality_pipeline
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/benchmark_ai_news_detection
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

#top_p_values=(0.6 0.8 0.9 0.95)
top_p_values=(0.9)

watermark_scheme_main="SWEET_P"
#watermark_scheme_compare="no_watermark"

#experiment_name_main="bias_variation_bias_20.0"
#experiment_name_compare="compare_watermark"

ppl_scorer_name="qwen2_chat_7B"
generator_name="llama_instruct_3.1"
return_loss_lists=True
batch_size=32

for i in ${!top_p_values[@]}; do
    top_p_value=${top_p_values[$i]}
    experiment_name_main="top_p_variation_${top_p_value}"

    test_text_quality pipeline=text_quality_pipeline  \
        pipeline.use_bert_scorer=False pipeline.use_idf_scorer=False pipeline.use_prometheus_scorer=False \
        pipeline.use_ppl_scorer=True \
        pipeline.watermarking_scheme_name_main=$watermark_scheme_main pipeline.watermarking_scheme_name_compare=$watermark_scheme_compare \
        pipeline.data_experiment_name_main=$experiment_name_main pipeline.data_experiment_name_compare=$experiment_name_compare \
        pipeline.ppl_scorer_name=$ppl_scorer_name pipeline.generator_name=$generator_name \
        pipeline.return_loss_lists=$return_loss_lists pipeline.batch_size=$batch_size
done


conda deactivate