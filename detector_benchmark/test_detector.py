from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    AutoConfig,
)
import torch

# hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
import json

from .generation import (
    GenParamsAttack,
    PromptAttack,
    PromptParaphrasingAttack,
    LLMGenerator,
    GenLoader,
)
from .utils.configs import ModelConfig, PromptConfig
from .detector import BertDetector, WatermarkDetector, FastDetectGPT, DetectorLoader
from .dataset_loader import CNNDataLoader, FakeTruePairsDataLoader
from .pipeline import ExperimentTestDetectorPipeline
from .watermark.auto_watermark import AutoWatermark


@hydra.main(version_base=None, config_path="conf", config_name="main")
def test_detector(cfg: DictConfig):

    # dataset parameters
    dataset_name = cfg.generation.dataset_name
    attack_name = cfg.generation.attack_name

    watermarking_scheme_name = cfg.watermark.algorithm_name
    dataset_experiment_path = (
        f"data/generated_datasets/{dataset_name}/{attack_name}/{watermarking_scheme_name}"
    )
    dataset_path = f"{dataset_experiment_path}/{cfg.generation.generator_name}_{cfg.generation.experiment_name}"
    print(f"Dataset path: {dataset_path}")

    # set the watermark config as the config used for generation
    with open(f"{dataset_path}_test.json", "r") as f:
        json_data = json.load(f)
        watermark_config = json_data["watermark_config"]["0"]

        # check that we have the same watermark algorithm
        assert watermark_config["algorithm_name"] == watermarking_scheme_name

        # check if there is a key called "threshold" in the watermark_config
        if "threshold" in watermark_config:
            # change it to "z_threshold"
            watermark_config["z_threshold"] = watermark_config["threshold"]

        # modify all values of cfg.watermark to the values in watermark_config
        for key, value in cfg.watermark.items():
            cfg.watermark[key] = watermark_config[key]

    # detection parameters
    test_res_dir = cfg.detection.test_res_dir
    detector_name = cfg.detection.detector_name
    batch_size = cfg.detection.batch_size
    weights_checkpoint = cfg.detection.weights_checkpoint

    print("Detector parameters:")
    for key, value in cfg.detection.items():
        print(f"{key}: {value}")

    if weights_checkpoint == "":
        weights_checkpoint = None
        local_weights = False
    else:
        local_weights = True

    # general parameters
    device = cfg.device

    print(f"Testing detector {detector_name} on dataset {dataset_path}")

    # Load detector
    detector_loader = DetectorLoader(cfg, detector_name, device, weights_checkpoint, local_weights)
    detector = detector_loader.load()

    experiment_path = (
        f"{test_res_dir}/{detector_name}_{watermarking_scheme_name}/{dataset_name}/{attack_name}"
    )
    non_attack_experiment_path = (
        f"{test_res_dir}/{detector_name}_{watermarking_scheme_name}/{dataset_name}/no_attack"
    )
    simple_test_detector_pipeline = ExperimentTestDetectorPipeline(
        cfg,
        detector,
        experiment_path,
        non_attack_experiment_path,
        dataset_experiment_path,
        batch_size,
    )
    simple_test_detector_pipeline.run_pipeline()


def main():
    test_detector()


if __name__ == "__main__":

    main()
