from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM,
                          ElectraForSequenceClassification, ElectraTokenizer, AutoConfig)
import torch

# hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

from generation import GenParamsAttack, PromptAttack, PromptParaphrasingAttack, LLMGenerator, GenLoader
from utils.configs import ModelConfig, PromptConfig
from detector import BertDetector, WatermarkDetector, FastDetectGPT, DetectorLoader
from dataset_loader import CNNDataLoader, FakeTruePairsDataLoader
from pipeline import ExperimentTestDetectorPipeline
from watermark.auto_watermark import AutoWatermark


@hydra.main(version_base=None, config_path="conf", config_name="main")
def test_detector(cfg: DictConfig):

    # dataset parameters
    dataset_name = cfg.general.dataset_name
    attack_name = cfg.general.attack_name
    watermarking_scheme_name = cfg.general.watermarking_scheme_name
    dataset_experiment_path = f"data/generated_datasets/{dataset_name}/{attack_name}/{watermarking_scheme_name}"

    # detection parameters
    test_res_dir = cfg.test_res_dir
    detector_name = cfg.detection.detector_name
    batch_size = cfg.detection.batch_size
    detection_threshold = cfg.detection.detection_threshold
    weights_checkpoint = cfg.detection.weights_checkpoint
    
    if weights_checkpoint == "":
        weights_checkpoint = None
        local_weights = False
    
    # general parameters
    device = cfg.device
    
    print(f"Testing detector {detector_name} on dataset {dataset_name}")
    
    # Load detector
    detector_loader = DetectorLoader(detector_name, detection_threshold, device,
                 weights_checkpoint, local_weights)
    detector = detector_loader.load()

    #experiment_path = f"{test_res_dir}/{detector_name}/{dataset_name}/{attack_name}/{watermarking_scheme_name}"
    experiment_path = f"{test_res_dir}/{dataset_name}/{attack_name}/{watermarking_scheme_name}"
    simple_test_watermark_pipeline = ExperimentTestDetectorPipeline(cfg, detector, device, experiment_path,
        dataset_experiment_path, batch_size)
    simple_test_watermark_pipeline.run_pipeline()
if __name__ == "__main__":
    
    
    test_detector()