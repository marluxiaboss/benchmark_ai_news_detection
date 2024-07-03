import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .generator import LLMGenerator
from utils.configs import ModelConfig



class GenLoader:
    
    def __init__(self, model_name, gen_params, device) -> None:
        
        self.model_name = model_name
        self.gen_params = gen_params
        self.device = device

    def load(self):
        """
        Load the specifed generator model and tokenizer
        
        Parameters

            
        Returns
        LLMGenerator
            The loaded generator model
            
        """
        # set generation parameters
        default_gen_params = self.gen_params
        
        model_name = self.model_name
        device = self.device
            
        # load generator
        if model_name == "qwen2_chat_0_5B":
            
            gen_path = "Qwen/Qwen2-0.5B-Instruct"
            gen_tokenizer = AutoTokenizer.from_pretrained(
                gen_path,
                pad_token='<|extra_0|>',
                eos_token='<|endoftext|>',
                padding_side='left',
                trust_remote_code=True
            )

            gen = AutoModelForCausalLM.from_pretrained(
                gen_path,
                torch_dtype="auto",
                device_map="auto",
                pad_token_id=gen_tokenizer.pad_token_id,
            ).to(device)

            # config for chat template and gen parameters
            use_chat_template = True
            chat_template_type = "system_user"
            gen_config = ModelConfig(gen_tokenizer,
                use_chat_template=use_chat_template, chat_template_type=chat_template_type,
                gen_params=default_gen_params, model_name=model_name, device=device)

            gen_model = LLMGenerator(gen, gen_config)
            
        elif model_name == "zephyr":
            gen_path = "HuggingFaceH4/zephyr-7b-beta"
            
            gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
            gen_tokenizer.pad_token = gen_tokenizer.eos_token
            
            gen = AutoModelForCausalLM.from_pretrained(gen_path,
                torch_dtype=torch.bfloat16,
                device_map="auto").to(device)

            # config for chat template and gen parameters
            use_chat_template = True
            chat_template_type = "system_user"
            gen_config = ModelConfig(gen_tokenizer,
                use_chat_template=use_chat_template, chat_template_type=chat_template_type,
                gen_params=default_gen_params, model_name=model_name, device=device)
            
            gen_model = LLMGenerator(gen, gen_config)

        elif model_name == "llama3_instruct":
            gen_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
            gen_tokenizer.pad_token = '<|eot_id|>'
            gen_tokenizer.padding_side = "left"

            gen = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(device)

            # config for chat template and gen parameters
            use_chat_template = True
            chat_template_type = "system_user"
            
            # special for llama3
            terminators = [
                gen_tokenizer.eos_token_id,
                gen_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            gen_params = default_gen_params     
            gen_params["eos_token_id"] = terminators
            
            gen_config = ModelConfig(gen_tokenizer,
                use_chat_template=use_chat_template, chat_template_type=chat_template_type,
                gen_params=gen_params, model_name=model_name, device=device)

            gen_model = LLMGenerator(gen, gen_config)
            
        else:
            # no other generator is supported for now
            raise ValueError("Generator not supported")
        
        return gen, gen_model, gen_config