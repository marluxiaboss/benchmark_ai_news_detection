from dataclasses import dataclass
import json
import os

import pandas as pd
import nltk.data
nltk.download('punkt')
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
        
        
def transform_chat_template_with_prompt(prefix: str, prompt: str, tokenizer: AutoTokenizer,
                                        use_chat_template: bool = False, template_type: str = None,
                                        system_prompt: str = "", forced_prefix: str = "") -> str:
    
    """
    Transform a prefix with a prompt into a chat template
    
    Parameters:
    prefix : str
        The prefix to use
    prompt : str
        The prompt to use
    tokenizer : AutoTokenizer
        The tokenizer to use
    use_chat_template : bool, optional
        Whether to use a chat template, by default False
    template_type : str, optional
        The type of template to use, by default None
    system_prompt : str, optional
        The system prompt to use, by default ""
        
    Returns:
    str
        The transformed prefix
    """
        

    if prefix != "":
        text_instruction = f"{prompt} {prefix}"
    else:
        text_instruction = prompt
        
    if use_chat_template:
        if system_prompt == "":
            sys_prompt = "You are a helpful assistant."
        else:
            sys_prompt = system_prompt
        match template_type:
            case "system_user":
                messages = [
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{text_instruction}"},
                ]
            case "user":
                messages = [
                {"role": "user", "content": f"{text_instruction}"},
                ]
            case _:
                raise ValueError("Template type not supported")

        text_template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # force prefix on the generated response
        #text_template = f"{text_template}\n{forced_prefix}"
        text_template = f"{text_template} {forced_prefix}"

    else:
        text_template = text_instruction

    return text_template


def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None
    
def load_json_as_list(input_file: str) -> list:
    """Load a JSON file as a list of dictionaries."""
    res = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        res.append(d)
    return res


def create_directory_for_file(file_path) -> None:
    """Create the directory for the specified file path if it does not already exist."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)