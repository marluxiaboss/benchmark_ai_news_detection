import argparse
from text_quality_evaluation import (Scorer, RefScorer, BertScoreScorer,
    SemScoreScorer, IDFScorer, PrometheusScorer, PPLScorer)
from generation import GenLoader
from pipeline import TextQualityPipeline
import numpy as np
from datasets import load_dataset
import json
import os

# hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import create_logger_file

def init_pipelines(cfg: DictConfig, log):
    
    # print the pipeline config
    for key in cfg.pipeline:
        log.info(f"{key}: {cfg.pipeline[key]}")
    
    dataset_name = cfg.pipeline.dataset_name
    data_experiment_name_main = cfg.pipeline.data_experiment_name_main
    data_experiment_name_compare = cfg.pipeline.data_experiment_name_compare
    generator_name = cfg.pipeline.generator_name
    watermarking_scheme_name_main = cfg.pipeline.watermarking_scheme_name_main
    
    # can be set to another watermarking scheme (or no watermarking scheme) if we want to compare different watermarking scheme against each other
    watermarking_scheme_name_compare = cfg.pipeline.watermarking_scheme_name_compare


    watermarked_dataset_path_main = f"data/generated_datasets/{dataset_name}/no_attack/{watermarking_scheme_name_main}/{generator_name}_{data_experiment_name_main}"
    watermarked_dataset_path_compare = f"data/generated_datasets/{dataset_name}/no_attack/{watermarking_scheme_name_compare}/{generator_name}_{data_experiment_name_compare}"
    
    pipelines = []
    
    if cfg.pipeline.use_bert_scorer:
        bert_scorer = BertScoreScorer("bert_score")
        pipeline = TextQualityPipeline(bert_scorer, watermarked_dataset_path_main, batch_size=cfg.pipeline.batch_size)
        pipelines.append(pipeline)
        
    if cfg.pipeline.use_idf_scorer:
        cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")["train"]
        cnn_dailymail = cnn_dailymail.shuffle(seed=42)
        idf_scorer = IDFScorer("idf_score", cnn_dailymail["article"][:10000])
        pipeline = TextQualityPipeline(idf_scorer, watermarked_dataset_path_main, batch_size=cfg.pipeline.batch_size)
        pipelines.append(pipeline)
        
    if cfg.pipeline.use_prometheus_scorer:
        
        assert watermarking_scheme_name_main != watermarking_scheme_name_compare, "Need to provide two different watermarking schemes"
        
        # flag to decide whether the comparison is between the watermarking scheme or with human to ai
        compare_ai_to_human = cfg.pipeline.compare_ai_to_human
        
        prometheus_scorer = PrometheusScorer("prometheus_score", compare_ai_to_human)
        pipeline = TextQualityPipeline(prometheus_scorer, watermarked_dataset_path_main, watermarked_dataset_path_compare, batch_size=cfg.pipeline.batch_size)
        pipelines.append(pipeline)
        
    if cfg.pipeline.use_ppl_scorer:
        
        # not really relevant for perplexity, could be removed
        default_gen_params = {
            "max_new_tokens": 220,
            "min_new_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.95,
            "repetition_penalty": 1,
            "do_sample": True,
            "top_k": 50
        }
        
        #gen_name = cfg.pipeline.generator_name
        ppl_gen = "qwen2_chat_7B"
        gen_params = default_gen_params
        device = "cuda" 
        # load the generator and tokenizer similarl to the create_dataset.py
        gen_loader = GenLoader(ppl_gen, gen_params, device)
        gen, _, gen_config = gen_loader.load()
        ppl_scorer = PPLScorer("ppl_score", gen, gen_config.tokenizer)
        pipeline = TextQualityPipeline(ppl_scorer, watermarked_dataset_path_main, batch_size=cfg.pipeline.batch_size)
        pipelines.append(pipeline)
        
    return pipelines

@hydra.main(version_base=None, config_path="conf", config_name="main")
def evaluate_text_quality(cfg: DictConfig):
    
    # setup logger
    experiment_path = (f"{cfg.pipeline.save_res_dir}/{cfg.pipeline.watermarking_scheme_name_main}_vs_{cfg.pipeline.watermarking_scheme_name_compare}/"
                f"{cfg.pipeline.dataset_name}/{cfg.pipeline.generator_name}/quality_test_{cfg.pipeline.data_experiment_name_main}_vs_{cfg.pipeline.data_experiment_name_compare}")

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    
    log = create_logger_file(f"{experiment_path}/log.txt")
    
    pipelines = init_pipelines(cfg, log)
    
    assert len(pipelines) > 0, "At least one pipeline must be provided"
    
    results_dict = {}
    for pipeline in pipelines:
        scorer_name = pipeline.scorer.name
        log.info(f"Running pipeline with scorer: {scorer_name}")
        score, lower_bound, upper_bound = pipeline.run_pipeline()
        log.info(f"Score: {score} +/- {upper_bound - score}")
        
        # transform numbers into str for the json file
        score = str(score)
        lower_bound = str(lower_bound)
        upper_bound = str(upper_bound)
        
        # Save the results
        results_dict[scorer_name] = {"score": score, "lower_bound": lower_bound, "upper_bound": upper_bound}
        
    # save config to results dict
    results_dict["config"] = dict(cfg.pipeline)
    
    # save the results to a json file
    json_path = (f"{cfg.pipeline.save_res_dir}/{cfg.pipeline.watermarking_scheme_name_main}_vs_{cfg.pipeline.watermarking_scheme_name_compare}/"
                f"{cfg.pipeline.dataset_name}/{cfg.pipeline.generator_name}/quality_test_{cfg.pipeline.data_experiment_name}_vs_{cfg.pipeline.data_experiment_name_compare}.json")
    
    with open(json_path, "w") as f:
        f.write(json.dumps(results_dict, indent=4))
    

if __name__ == "__main__":
    
    evaluate_text_quality()
    
    
    
    
    