from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

import os

from .bert_detector import BertDetector
from .fast_detect_gpt import FastDetectGPT
from .gpt_zero_detector import GPTZero
from .watermark_detector import WatermarkDetector
from .detector import Detector
from ..watermark.auto_watermark import AutoWatermark
from ..generation import GenLoader


class DetectorLoader:
    def __init__(
        self,
        cfg: dict,
        detector_name: str,
        device: str,
        weights_checkpoint: str = None,
        local_weights: bool = False,
    ) -> None:
        """
        Class used to load a detector based on the given configuration.

        Parameters:
        ----------
            cfg: dict
                The configuration dictionary (hydra config)
            detector_name: str
                The name of the detector to load
            device: str
                The device to use for the detector
            weights_checkpoint: str
                The path to the weights checkpoint. Default is None.
            local_weights: bool
                Whether to load the weights locally. Default is False.
        """

        self.cfg = cfg
        self.detector_name = detector_name
        self.device = device
        self.weights_checkpoint = weights_checkpoint
        self.local_weights = local_weights

    def load(self) -> Detector:
        """
        Load the detector based on the given configuration (init).

        Returns:
        ----------
            Detector
                The loaded detector
        """

        detector_name = self.detector_name
        device = self.device

        match detector_name:

            case "electra_large":
                assert (
                    self.local_weights and self.weights_checkpoint is not None
                ), "This detector requires a weights checkpoint"

                detector_path = "google/electra-large-discriminator"
                config = AutoConfig.from_pretrained(detector_path)
                detector_model = ElectraForSequenceClassification(config)
                detector_tokenizer = ElectraTokenizer.from_pretrained(detector_path)

                model_path = self.weights_checkpoint
                detector_model.load_state_dict(torch.load(model_path))
                detector_model.to(device)

                detector = BertDetector(detector_model, detector_tokenizer, device)

            case "fast_detect_gpt":

                # TODO: add more config options for fast_detect_gpt
                match self.cfg.detection.ref_model:
                    case "gpt2":
                        ref_model_path = "openai-community/gpt2"
                    case "gpt-neo":
                        ref_model_path = "EleutherAI/gpt-neo-2.7B"
                    case "gpt-j":
                        ref_model_path = "EleutherAI/gpt-j-6B"
                    case _:
                        raise ValueError("Reference model not supported yet")

                ref_model = AutoModelForCausalLM.from_pretrained(
                    ref_model_path, torch_dtype="auto"
                ).to(device)
                ref_tokenizer = AutoTokenizer.from_pretrained(
                    ref_model_path, trust_remote_code=True, padding_side="left"
                )

                # special for gpt2
                ref_tokenizer.pad_token = ref_tokenizer.eos_token
                ref_tokenizer.padding_side = "left"

                scoring_model = ref_model
                scoring_tokenizer = ref_tokenizer

                detector = FastDetectGPT(
                    ref_model, scoring_model, ref_tokenizer, scoring_tokenizer, device
                )

            case "gpt_zero":

                debug_mode = self.cfg.detection.debug_mode
                api_key = os.environ.get("GPT_ZERO_API_KEY", None)
                detector = GPTZero(api_key, debug_mode)

            case "watermark_detector":
                cfg = self.cfg
                model_name = cfg.generation.generator_name

                # Note: we should adapt these parameters to the attack since we can assume that
                # if the attacker has changed the parameter, we will know it
                default_gen_params = {
                    "max_new_tokens": 220,
                    "min_new_tokens": 200,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "repetition_penalty": 1,
                    "do_sample": True,
                    "top_k": 50,
                }
                gen_params = default_gen_params

                if cfg.generation.get("temperature", None) is not None:
                    gen_params["temperature"] = cfg.generation.temperature

                if cfg.generation.get("repetition_penalty", None) is not None:
                    gen_params["repetition_penalty"] = cfg.generation.repetition_penalty

                if cfg.generation.get("max_new_tokens", None) is not None:
                    gen_params["max_new_tokens"] = cfg.generation.max_new_tokens

                if cfg.generation.get("min_new_tokens", None) is not None:
                    gen_params["min_new_tokens"] = cfg.generation.min_new_tokens

                if cfg.watermark.get("algorithm_name", None) == "SWEET":
                    # maybe specify the name of a smaller surrogate model, like in the paper
                    gen_loader = GenLoader(model_name, gen_params, device, gen_tokenizer_only=False)
                    gen, _, gen_config = gen_loader.load()

                else:
                    gen_loader = GenLoader(model_name, gen_params, device, gen_tokenizer_only=True)
                    _, _, gen_config = gen_loader.load()
                    gen = None

                watemark_scheme = AutoWatermark.load(
                    self.cfg.watermark.algorithm_name,
                    algorithm_config=self.cfg.watermark,
                    gen_model=gen,
                    model_config=gen_config,
                )
                detector = WatermarkDetector(watemark_scheme, self.cfg.watermark.z_threshold)

            case _:
                raise ValueError(f"Detector {detector_name} not supported yet")

        return detector
