
from omegaconf import DictConfig
from .generator import LLMGenerator
from .prompt_attack import PromptAttack
from .gen_params_attack import GenParamsAttack
from .prompt_paraphrasing_attack import PromptParaphrasingAttack
from utils.configs import ModelConfig, PromptConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
    
class AttackLoader:
        
    def __init__(self, cfg: DictConfig, attack_name: str, gen_model: LLMGenerator, model_config: ModelConfig, max_sample_len: int, 
                    watermarking_scheme_logits_processor: LogitsProcessor=None, paraphraser_model: LLMGenerator=None,
                    paraphraser_config: ModelConfig=None):
        
        self.cfg = cfg
        self.attack_name = attack_name
        self.gen_model = gen_model
        self.model_config = model_config
        self.max_sample_len = max_sample_len
        self.watermarking_scheme_logits_processor = watermarking_scheme_logits_processor
        self.paraphraser_model = paraphraser_model
        self.paraphraser_config = paraphraser_config
    
    def load(self):
        
        cfg = self.cfg
        attack_name = self.attack_name
        gen_model = self.gen_model
        model_config = self.model_config
        max_sample_len = self.max_sample_len
        watermarking_scheme_logits_processor = self.watermarking_scheme_logits_processor
        paraphraser_model = self.paraphraser_model
        paraphraser_config = self.paraphraser_config
        
        
        match attack_name:
            case "no_attack":
                
                system_prompt = cfg.generation.system_prompt
                user_prompt = cfg.generation.user_prompt
                prompt_config = PromptConfig(system_prompt=system_prompt, user_prompt=user_prompt)
                
                attack = PromptAttack(gen_model, model_config,
                    prompt_config, prompt_config, max_sample_len, watermarking_scheme_logits_processor)
                
            case "prompt_attack":
                
                adversarial_system_prompt = cfg.generation.user_prompt
                adversarial_user_prompt = cfg.generation.system_prompt
                prompt_config = PromptConfig(system_prompt=adversarial_system_prompt, user_prompt=adversarial_user_prompt)
                
                attack = PromptAttack(gen_model, model_config,
                    prompt_config, prompt_config, max_sample_len, watermarking_scheme_logits_processor)
                
            case "gen_params_attack": 
                
                system_prompt = "You are a helpful assistant."
                user_prompt = "Continue writing the following news article starting with:"
                prompt_config = PromptConfig(system_prompt=system_prompt, user_prompt=user_prompt)
                
                adversarial_gen_params = model_config.gen_params
                adversarial_gen_params["temperature"] = cfg.generation.temperature
                adversarial_gen_params["repetition_penalty"] = cfg.generation.repetition_penalty
                
                attack = GenParamsAttack(gen_model, model_config, prompt_config, 
                    adversarial_gen_params, max_sample_len, watermarking_scheme_logits_processor)
                
                
            case "prompt_paraphrasing_attack":
            
            # see TODO below
            # assert (paraphraser_model is not None) and (paraphraser_config is not None), "Paraphraser model and config must be provided"
                
                system_paraphrasing_prompt = cfg.generation.system_paraphrasing_prompt
                user_paraphrasing_prompt = cfg.generation.user_paraphrasing_prompt
                paraphraser_prompt_config = PromptConfig(system_prompt=system_paraphrasing_prompt, user_prompt=user_paraphrasing_prompt)

                system_prompt = cfg.generation.system_prompt
                user_prompt = cfg.generation.user_prompt
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
            