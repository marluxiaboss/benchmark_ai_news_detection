import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generator import LLMGenerator

def load_generator(model_name: str, device: str, access_token: str = None, temperature: float = 0.8,
                   repetition_penalty: float = 1.0, top_p: float = 0.95, top_k: int = 50, checkpoint_path: str = None) -> LLMGenerator:
    """
    Load the specifed generator model and tokenizer
    
    Parameters
    model_name: str
        Name of the model to load among ["qwen_chat", "qwen_0.5b", "gpt2", "gemma_2b_chat", "gemma_2b", "phi", "mistral", "zephyr", "llama3_instruct", "phi_news"]
    device: str
        Device to load the model on
    access_token: str, optional
        Access token required for loading some models, by default None
    temperature: float, optional
        Temperature for generation, by default 0.8
    repetition_penalty: float, optional
        Repetition penalty for generation, by default 1.0
    top_p: float, optional
        Top p for generation, by default 0.95
    top_k: int, optional
        Top k for generation, by default 50
    checkpoint_path: str, optional
        Path to the checkpoint, by default None
        
    Returns
    LLMGenerator
        The loaded generator model
        
    """
    
    # set generation parameters
    default_gen_params = {
        "max_length": 200,
        "max_new_tokens": None,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1,
        "do_sample": True,
        "min_new_tokens": 100,
        "top_k": 50
    }

    default_gen_params["temperature"] = temperature
    default_gen_params["repetition_penalty"] = repetition_penalty
    default_gen_params["top_p"] = top_p
    default_gen_params["top_k"] = top_k
        
    # load generator
    if model_name == "qwen_chat":
        gen_path = "Qwen/Qwen1.5-0.5B-Chat"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = True
        template_type ="system_user"

    elif model_name == "qwen_0.5b":
        gen_path = "Qwen/Qwen1.5-0.5B"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None        

    elif model_name == "gpt2":
        gen_path = "openai-community/gpt2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype="auto").to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True, padding_side="left")

        gen_params = default_gen_params
        #gen_params["repetition_penalty"] = 2.0
        
        # special for gpt2
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None
        
    elif model_name == "gemma_2b_chat":
        gen_path = "google/gemma-2b-it"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=access_token, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = True
        template_type ="user"

    elif model_name == "gemma_2b":
        gen_path = "google/gemma-2b"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path,  token=access_token, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path,  token=access_token)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        #template for chat
        use_chat_template = False
        template_type = None  

    elif model_name == "phi":
        gen_path = "microsoft/phi-2"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.float16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        # special for phi
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        #template for chat
        use_chat_template = False
        template_type = None  

    elif model_name == "mistral":
        gen_path = "mistralai/Mistral-7B-v0.1"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        #template for chat
        use_chat_template = False
        template_type = None  

    elif model_name == "zephyr":
        gen_path = "HuggingFaceH4/zephyr-7b-beta"
        
        # For zephyr, we can load from a checkpoint
        if checkpoint_path:
            gen_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).to(device)
        else:
            gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        #template for chat
        use_chat_template = True
        template_type ="system_user"
        
    elif model_name == "llama3_instruct":
        gen_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        
        
        #gen_tokenizer.eos_token ='<|eot_id|>'
        #gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.pad_token = '<|eot_id|>'
        gen_tokenizer.padding_side = "left"

        #template for chat
        use_chat_template = True
        template_type ="system_user"
        
        
        # special for llama3
        terminators = [
            gen_tokenizer.eos_token_id,
            gen_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        gen_params = default_gen_params     
        gen_params["eos_token_id"] = terminators
        
        generator = LLMGenerator(gen_model, gen_tokenizer, gen_params=default_gen_params)

    elif model_name == "phi_news":
        gen_path = "trained_models/phi-2-cnn_news_peft"
        base_path = "microsoft/phi-2"

        gen_model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.float16, local_files_only=True).to(device)
        gen_tokenizer = AutoTokenizer.from_pretrained(base_path)
        generator = LLMGenerator(gen_model, gen_tokenizer, device, gen_params=default_gen_params)

        # special for phi
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = 'left'

        #template for chat
        use_chat_template = False
        template_type = None  


    else:
        # no other generator is supported for now
        raise ValueError("Generator not supported")
    
    return generator, gen_tokenizer, use_chat_template, template_type