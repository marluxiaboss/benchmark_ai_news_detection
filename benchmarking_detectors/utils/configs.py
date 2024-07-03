from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM


@dataclass
class ModelConfig:
    def __init__(self, tokenizer, use_chat_template, chat_template_type, gen_params, model_name, device):
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.chat_template_type = chat_template_type
        self.gen_params = gen_params
        self.model_name = model_name
        self.device = device

@dataclass
class PromptConfig:
    def __init__(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt