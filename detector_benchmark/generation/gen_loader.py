import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .generator import LLMGenerator
from ..utils.configs import ModelConfig


class GenLoader:
    def __init__(
        self, model_name: str, gen_params: dict, device: str, gen_tokenizer_only: bool = False
    ) -> None:
        """
        Class for loading a generator model (LLMGenerator class) and tokenizer from Huggingface.

        Parameters:
        ----------
            model_name: str
                The name of the model to load.
            gen_params: dict
                The generation parameters to use for generation.
            device: str
                The device to use for generation.
        """

        self.model_name = model_name
        self.gen_params = gen_params
        self.device = device
        self.gen_tokenizer_only = gen_tokenizer_only

    def load(self) -> tuple[torch.nn.Module, LLMGenerator, ModelConfig]:
        """
        Load the specifed generator model (from init) and tokenizer

        Returns:
        -------
            torch.nn.Module: The loaded generator model
            LLMGenerator: The loaded generator model
            ModelConfig: The configuration of the generator model
        """
        # set generation parameters
        default_gen_params = self.gen_params

        model_name = self.model_name
        device = self.device

        # load generator
        match model_name:

            case "qwen2_0_5B":
                gen_path = "Qwen/Qwen2-0.5B"
                gen_tokenizer = AutoTokenizer.from_pretrained(
                    gen_path,
                    pad_token="<|extra_0|>",
                    eos_token="<|endoftext|>",
                    padding_side="left",
                    trust_remote_code=True,
                )
                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path,
                        torch_dtype="auto",
                        device_map="auto",
                        pad_token_id=gen_tokenizer.pad_token_id,
                    )

                # config for chat template and gen parameters
                use_chat_template = False
                chat_template_type = None
                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=default_gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case "qwen2_chat_0_5B":
                gen_path = "Qwen/Qwen2-0.5B-Instruct"
                gen_tokenizer = AutoTokenizer.from_pretrained(
                    gen_path,
                    pad_token="<|extra_0|>",
                    eos_token="<|endoftext|>",
                    padding_side="left",
                    trust_remote_code=True,
                )
                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path,
                        torch_dtype="auto",
                        device_map="auto",
                        pad_token_id=gen_tokenizer.pad_token_id,
                    )

                # config for chat template and gen parameters
                use_chat_template = True
                chat_template_type = "system_user"
                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=default_gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case "qwen2_chat_7B":
                gen_path = "Qwen/Qwen2-7B-Instruct"
                gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
                gen_tokenizer.pad_token = gen_tokenizer.eos_token

                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path, torch_dtype=torch.bfloat16, device_map="auto"
                    )

                # config for chat template and gen parameters
                use_chat_template = True
                chat_template_type = "system_user"
                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=default_gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case "zephyr":
                gen_path = "HuggingFaceH4/zephyr-7b-beta"
                gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
                gen_tokenizer.pad_token = gen_tokenizer.eos_token

                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path, torch_dtype=torch.bfloat16, device_map="auto"
                    )

                # config for chat template and gen parameters
                use_chat_template = True
                chat_template_type = "system_user"
                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=default_gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case "llama3_instruct_3":
                gen_path = "meta-llama/Meta-Llama-3-8B-Instruct"
                gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
                gen_tokenizer.pad_token = "<|eot_id|>"
                gen_tokenizer.padding_side = "left"

                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path, torch_dtype=torch.bfloat16, device_map="auto"
                    )

                # config for chat template and gen parameters
                use_chat_template = True
                chat_template_type = "system_user"

                # special for llama3
                terminators = [
                    gen_tokenizer.eos_token_id,
                    gen_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                gen_params = default_gen_params
                gen_params["eos_token_id"] = terminators

                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case "llama_instruct_3.1":
                gen_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
                gen_tokenizer.pad_token = "<|eot_id|>"
                gen_tokenizer.padding_side = "left"

                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path, torch_dtype=torch.bfloat16, device_map="auto"
                    )

                # config for chat template and gen parameters
                use_chat_template = True
                chat_template_type = "system_user"

                # special for llama3
                terminators = [
                    gen_tokenizer.eos_token_id,
                    gen_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                gen_params = default_gen_params
                gen_params["eos_token_id"] = terminators

                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case "llama_3.1":
                gen_path = "meta-llama/Meta-Llama-3.1-8B"
                gen_tokenizer = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
                gen_tokenizer.pad_token = "<|eot_id|>"
                gen_tokenizer.padding_side = "left"

                if self.gen_tokenizer_only:
                    gen = None
                else:
                    gen = AutoModelForCausalLM.from_pretrained(
                        gen_path, torch_dtype=torch.bfloat16, device_map="auto"
                    )

                # config for chat template and gen parameters
                use_chat_template = False
                chat_template_type = None

                # special for llama3
                terminators = [
                    gen_tokenizer.eos_token_id,
                    gen_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                gen_params = default_gen_params
                gen_params["eos_token_id"] = terminators

                gen_config = ModelConfig(
                    gen_tokenizer,
                    use_chat_template=use_chat_template,
                    chat_template_type=chat_template_type,
                    gen_params=gen_params,
                    model_name=model_name,
                    device=device,
                )

                gen_model = LLMGenerator(gen, gen_config)

            case _:
                # no other generator is supported for now
                raise ValueError(f"Generator {model_name} not supported yet")

        return gen, gen_model, gen_config
