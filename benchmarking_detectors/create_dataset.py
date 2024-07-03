from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM,
                          ElectraForSequenceClassification, ElectraTokenizer, AutoConfig)
import torch
import argparse

from generation import GenParamsAttack, PromptAttack, PromptParaphrasingAttack, LLMGenerator, GenLoader
from utils.configs import ModelConfig, PromptConfig
from detector import BertDetector, WatermarkDetector, FastDetectGPT, DetectorLoader
from dataset_loader import CNNDataLoader, FakeTruePairsDataLoader
from pipeline import ExperimentTestPipeline, CreateDatasetPipeline, ExperimentTestPipelineFull
from watermark.auto_watermark import AutoWatermark


def choose_dataset(dataset_name: str, dataset_size: int, max_sample_len: int, prefix_size: int):
    match dataset_name:
        case "cnn_dailymail":
            cnn_data_loader = CNNDataLoader(dataset_size, max_sample_len=max_sample_len, prefix_size=prefix_size)
        case _:
            raise ValueError(f"Dataset {dataset_name} not supported yet")
    return cnn_data_loader

def choose_generator(model_name: str, gen_params, device: str):
    
    gen_loader = GenLoader(model_name, gen_params, device)
    gen, gen_model, gen_config = gen_loader.load()
    
    return gen, gen_model, gen_config

def choose_attack(attack_name: str, gen_model, model_config, max_sample_len,
                  watermarking_scheme_logits_processor=None, paraphraser_model=None,
                  paraphraser_config=None, paraphraser_prompt_config=None):
    
    match attack_name:
        case "no_attack":
            
            system_prompt = "You are a helpful assistant."
            user_prompt = "Continue writing the following news article starting with:"
            prompt_config = PromptConfig(system_prompt=system_prompt, user_prompt=user_prompt)
            
            attack = PromptAttack(gen_model, model_config,
                prompt_config, prompt_config, max_sample_len, watermarking_scheme_logits_processor)
            
        case "prompt_attack":
            
            # TODO: how to configure the attacks? Maybe have a config file for each attack?
            adversarial_system_prompt = "You are a tweeter user tweeting news information from news articles to your followers."
            adversarial_user_prompt = "Write a 500 characters news tweet starting with:"
            prompt_config = PromptConfig(system_prompt=adversarial_system_prompt, user_prompt=adversarial_user_prompt)
            
            attack = PromptAttack(gen_model, model_config,
                prompt_config, prompt_config, max_sample_len, watermarking_scheme_logits_processor)
            
        case "gen_params_attack":
            
            system_prompt = "You are a helpful assistant."
            user_prompt = "Continue writing the following news article starting with:"
            prompt_config = PromptConfig(system_prompt=system_prompt, user_prompt=user_prompt)
            
            adversarial_gen_params = model_config.gen_params
            adversarial_gen_params["temperature"] = 1.2
            
            attack = GenParamsAttack(gen_model, model_config, prompt_config, 
                adversarial_gen_params, max_sample_len, watermarking_scheme_logits_processor)
            
            
        case "prompt_paraphrasing_attack":
            assert (paraphraser_model is not None) and (paraphraser_config is not None), "Paraphraser model and config must be provided"
            
            system_paraphrasing_prompt = """You are a paraphraser. You are given an input passage ‘INPUT’. You should paraphrase ‘INPUT’ to print ‘OUTPUT’."
                "‘OUTPUT’ shoud be diverse and different as much as possible from ‘INPUT’ and should not copy any part verbatim from ‘INPUT’."
                "‘OUTPUT’ should preserve the meaning and content of ’INPUT’ while maintaining text quality and grammar."
                "‘OUTPUT’ should not be much longer than ‘INPUT’. You should print ‘OUTPUT’ and nothing else so that its easy for me to parse."""
            user_paraphrasing_prompt = "INPUT:"
            paraphraser_prompt_config = PromptConfig(system_prompt=system_paraphrasing_prompt, user_prompt=user_paraphrasing_prompt)

            system_prompt = "You are a helpful assistant."
            user_prompt = "Continue writing the following news article starting with:"
            gen_prompt_config = PromptConfig(system_prompt=system_prompt, user_prompt=system_prompt)
            
            # TODO: for now we use the same model for gen and paraphrasing, but this should be configurable
            paraphraser_model = gen_model
            paraphraser_config = model_config
            
            attack = PromptParaphrasingAttack(gen_model, model_config, gen_prompt_config, 
                paraphraser_model, paraphraser_config, paraphraser_prompt_config, max_sample_len,
                watermarking_scheme_logits_processor)
            
        case _:
            raise ValueError(f"Attack {attack_name} not supported yet")
        
    return attack
            

def choose_watermarking_scheme(watermarking_scheme_name: str, gen, model_config):
    
    match watermarking_scheme_name:
        case "kgw":
            watermarking_scheme = AutoWatermark.load('KGW', 
                                 algorithm_config='watermark/watermarking_config/KGW.json',
                                 gen_model=gen,
                                 model_config=model_config)
        case "sir":
            watermarking_scheme = AutoWatermark.load('SIR',
                                    algorithm_config='watermark/watermarking_config/SIR.json',
                                    gen_model=gen,
                                    model_config=model_config)
        case _:
            raise ValueError(f"Watermarking scheme {watermarking_scheme_name} not supported yet")
        
    return watermarking_scheme
            

def create_dataset(dataset_size: int, max_sample_len: int, prefix_size: int,
                        dataset_name: str, generator_name: str, attack_name: str,
                        watermarking_scheme_name: str, batch_size: int, device: str):
    
    print(f"Creating dataset with the following parameters:")
    print(f"Dataset size: {dataset_size}")
    print(f"Max sample length: {max_sample_len}")
    print(f"Prefix size: {prefix_size}")
    print(f"Dataset name: {dataset_name}")
    print(f"Generator name: {generator_name}")
    print(f"Attack name: {attack_name}")
    print(f"Watermarking scheme name: {watermarking_scheme_name}")
    
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
    device = device
    gen, gen_model, gen_config = choose_generator(generator_name, default_gen_params, device)

    ### Watermarking ###
    if watermarking_scheme_name == "":
        watermarking_scheme = None
        watermarking_scheme_logits_processor = None
    else:
        watermarking_scheme = choose_watermarking_scheme(watermarking_scheme_name, gen, gen_config)
        watermarking_scheme_logits_processor = watermarking_scheme.logits_processor
    
    ### Prompt & Attack ###
    
    attack = choose_attack(attack_name, gen_model, gen_config, max_sample_len, watermarking_scheme_logits_processor)
    attack.set_attack_name(attack_name)
    attack.set_watermarking_scheme_name(watermarking_scheme_name)
    
    ### Pipeline ###
    skip_cache = False
    experiment_path = "benchmark_saved_results"
    simple_test_watermark_pipeline = CreateDatasetPipeline(cnn_data_loader, attack,
        device, experiment_path, batch_size=batch_size, skip_cache=skip_cache)
    simple_test_watermark_pipeline.run_pipeline()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_size", type=int,
        help="Size of the dataset to create. Note that the total size will be 2 * dataset_size for fake and true samples and that the test/eval split will be 0.1 of the total size.",
        default=100)
    parser.add_argument("--max_sample_len", type=int, help="Maximum length of the samples in the dataset (in chars)", default=500)
    parser.add_argument("--prefix_size", type=int, help="Size of the prefix to use for the generation (in words)", default=10)
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to create", default="cnn_dailymail")
    parser.add_argument("--generator_name", type=str, help="Name of the generator to use", default="qwen2_chat_0_5B")
    parser.add_argument("--attack_name", type=str, help="Name of the attack to use", default="no_attack")
    parser.add_argument("--watermarking_scheme_name", type=str, help="Name of the watermarking scheme to use. Use "" for no watermarking.", default="")
    parser.add_argument("--batch_size", type=int, help="Batch size to use for the generation for all models", default=1)
    parser.add_argument("--device", type=str, help="Device to use for the generation", default="cuda")
    
    args = parser.parse_args()
    dataset_size = args.dataset_size
    max_sample_len = args.max_sample_len
    prefix_size = args.prefix_size
    dataset_name = args.dataset_name
    generator_name = args.generator_name
    attack_name = args.attack_name
    watermarking_scheme_name = args.watermarking_scheme_name
    batch_size = args.batch_size
    device = args.device
    
    #datasets = ["cnn_dailymail"]
    #generators = ["qwen2_chat_0_5B", "zephyr", "llama3_instruct"]
    #attacks = ["no_attack", "prompt_attack", "prompt_paraphrasing_attack", "gen_param_attack"]
    #watermarking_schemes = ["kgw", "sir"]
    
    create_dataset(dataset_size, max_sample_len, prefix_size, dataset_name, generator_name,
        attack_name, watermarking_scheme_name, batch_size, device)