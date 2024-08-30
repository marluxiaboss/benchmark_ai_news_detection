from dataclasses import dataclass
from typing import Optional

import json

from transformers import AutoTokenizer

@dataclass
class ModelConfig:
    def __init__(self, tokenizer: AutoTokenizer, gen_params: Optional[dict]=None, model_name: Optional[str]="",
                 use_chat_template: Optional[bool]=True, chat_template_type: str="system_user", device: str="cuda"):
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.chat_template_type = chat_template_type
        self.gen_params = gen_params
        self.model_name = model_name
        self.device = device
        
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