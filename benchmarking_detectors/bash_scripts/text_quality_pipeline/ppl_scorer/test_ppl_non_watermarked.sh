cd detector_benchmark
conda activate llm_detector

watermark_scheme_main="no_watermark"
watermark_scheme_compare="no_watermark"

experiment_name_main="compare_watermark"
experiment_name_compare="compare_watermark"

ppl_scorer_name="qwen2_chat_7B"
generator_name="llama3_instruct_3.1"
return_loss_lists=True
batch_size=64

python test_text_quality.py pipeline=text_quality_pipeline  \
    pipeline.use_bert_scorer=False pipeline.use_idf_scorer=False pipeline.use_prometheus_scorer=False \
    pipeline.use_ppl_scorer=True \
    pipeline.watermarking_scheme_name_main=$watermark_scheme_main pipeline.watermarking_scheme_name_compare=$watermark_scheme_compare \
    pipeline.data_experiment_name_main=$experiment_name_main pipeline.data_experiment_name_compare=$experiment_name_compare \
    pipeline.ppl_scorer_name=$ppl_scorer_name pipeline.generator_name=$generator_name \
    pipeline.return_loss_lists=$return_loss_lists pipeline.batch_size=$batch_size



conda deactivate