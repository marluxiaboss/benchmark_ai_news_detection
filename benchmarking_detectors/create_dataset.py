from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM,
                          ElectraForSequenceClassification, ElectraTokenizer, AutoConfig)
import torch
import argparse

# hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

from generation import GenParamsAttack, PromptAttack, PromptParaphrasingAttack, LLMGenerator, GenLoader, AttackLoader
from utils.configs import ModelConfig, PromptConfig
from detector import BertDetector, WatermarkDetector, FastDetectGPT, DetectorLoader
from dataset_loader import CNNDataLoader, FakeTruePairsDataLoader
from pipeline import ExperimentTestPipeline, CreateDatasetPipeline, ExperimentTestPipelineFull
from watermark.auto_watermark import AutoWatermark


def choose_dataset(dataset_name: str, dataset_size: int, max_sample_len: int, prefix_size: int):
    
    train_fraction = 0.8
    eval_fraction = 0.1
    test_fraction = 0.1
    
    match dataset_name:
        case "cnn_dailymail":
            cnn_data_loader = CNNDataLoader(dataset_size, max_sample_len=max_sample_len, prefix_size=prefix_size,
                train_fraction=train_fraction, eval_fraction=eval_fraction, test_fraction=test_fraction)
        case _:
            raise ValueError(f"Dataset {dataset_name} not supported yet")
    return cnn_data_loader

def choose_generator(model_name: str, gen_params, device: str):
    
    gen_loader = GenLoader(model_name, gen_params, device)
    gen, gen_model, gen_config = gen_loader.load()
    
    return gen, gen_model, gen_config

def choose_watermarking_scheme(cfg: DictConfig, watermarking_scheme_name: str, gen, model_config):
    
    algorithm_config = cfg.watermark

    watermarking_scheme = AutoWatermark.load(watermarking_scheme_name,
                    algorithm_config=algorithm_config,
                    gen_model=gen,
                    model_config=model_config)
    
    return watermarking_scheme

def choose_attack(cfg: DictConfig, attack_type: str, gen_model, model_config, max_sample_len, 
                  watermarking_scheme_logits_processor=None, paraphraser_model=None,
                  paraphraser_config=None):
    
    attack_loader = AttackLoader(cfg, attack_type, gen_model, model_config, max_sample_len, watermarking_scheme_logits_processor,
                                    paraphraser_model, paraphraser_config)
    attack = attack_loader.load()
        
    return attack
        

@hydra.main(version_base=None, config_path="conf", config_name="main")
def create_dataset(cfg: DictConfig):
    
    #print(OmegaConf.to_yaml(cfg))
    
    # generator parameters
    device = cfg.device
    batch_size = cfg.generation.batch_size
    
    # generation parameters
    dataset_size = cfg.generation.dataset_size
    max_sample_len = cfg.generation.max_sample_len
    prefix_size = cfg.generation.prefix_size
    dataset_name = cfg.generation.dataset_name
    max_new_tokens = cfg.generation.max_new_tokens
    min_new_tokens = cfg.generation.min_new_tokens
    generator_name = cfg.generation.generator_name
    attack_name = cfg.generation.attack_name
    attack_type = cfg.generation.attack_type
    skip_cache = cfg.generation.skip_cache
    skip_train_split = cfg.generation.skip_train_split
    
    # watermarking parameters
    watermarking_scheme_name = cfg.watermark.algorithm_name
    
    print(f"Creating dataset with the following parameters:")
    
    print("General parameters:")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print()
        
    print("Generation parameters:")
    for k, v in cfg.generation.items():
        print(f"{k}: {v}")
    print()
    
    print("Watermarking parameters:")
    for k, v in cfg.watermark.items():
        print(f"{k}: {v}")
    
    ### Data loader ###
    cnn_data_loader = choose_dataset(dataset_name, dataset_size, max_sample_len, prefix_size)

    ### Generator ###
    default_gen_params = {
        "max_new_tokens": 220,
        "min_new_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1,
        "do_sample": True,
        "top_k": 50
    }
    
    default_gen_params["max_new_tokens"] = max_new_tokens
    default_gen_params["min_new_tokens"] = min_new_tokens
    
    
    # load generator
    gen, gen_model, gen_config = choose_generator(generator_name, default_gen_params, device)

    ### Watermarking ###
    if watermarking_scheme_name == "no_watermark":
        watermarking_scheme = None
        watermarking_scheme_logits_processor = None
    else:
        watermarking_scheme = choose_watermarking_scheme(cfg, watermarking_scheme_name, gen, gen_config)
        watermarking_scheme_logits_processor = watermarking_scheme.logits_processor
    
    ### Prompt & Attack ###
    attack = choose_attack(cfg, attack_type, gen_model, gen_config, max_sample_len, watermarking_scheme_logits_processor)
    attack.set_attack_name(attack_name)
    attack.set_watermarking_scheme_name(watermarking_scheme_name)
    
    ### Pipeline ###
    experiment_path = f"data/generated_datasets/{dataset_name}/{attack_name}/{watermarking_scheme_name}"
    #experiment_path = "benchmark_saved_results"
    simple_test_watermark_pipeline = CreateDatasetPipeline(cfg, cnn_data_loader, attack,
        experiment_path, batch_size=batch_size, skip_cache=skip_cache, skip_train_split=skip_train_split)
    simple_test_watermark_pipeline.run_pipeline()
    

if __name__ == "__main__":
    
    
    create_dataset()