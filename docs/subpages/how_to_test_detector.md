# How to test a detector

cf. `bash_scripts/test_detectors/test_zero_shot_detectors/test_fast_detect_gpt_detector.sh` (see other scripts in `bash_scripts/test_detectors` for other examples)

## Test a zero-shot or trained detector
```sh
generator_name="qwen2_chat_0_5B"
attack="generation_base"
batch_size=128
gen_experiment_name="compare_watermark"
detection_experiment_name="compare_detectors"
test_res_dir="detection_test_results"

# detector specification
detector="fast_detect_gpt"

python test_detector.py detection=$detector generation=$attack generation.generator_name=$generator_name generation.experiment_name=$gen_experiment_name \
        detection.experiment_name=$detection_experiment_name  \
        detection.test_res_dir=$test_res_dir detection.batch_size=$batch_size \
        detection.weights_checkpoint=$weights_checkpoint detection.detector_name=$detector_name   
```

```{admonition} Configure the benchmark
:class: hint

This section outlines the parameters used for testing a detector with the specified configurations. See the configuration files under `detector_benchmark/conf/detection` for the complete list.
Note that each detector may have its own specific parameters that we don't cover here, look at the individual configuration files for each detector for more details.

- `attack`: See [how to generate page](how_to_generate.md). Note that we don't assume that we know the attack, we specify the attack here because it allow us to track the data folder name for the corresponding generated dataset. Default value: "generation_base"

- `batch_size`: Specifies the number of samples to be process in parallel by the GPU. Default value: 2

- `gen_experiment_name`: Specifies the experiment name for generating the data. The name should match the corresponding `experiment_name` value when generating the benchmark (see the `experiment_name` field in [how to generate page](how_to_generate.md)). Default value: base

- `detection_experiment_name`: Specifies the experiment name for the detection test - used to identify a specific experiment in the result folders (similar to `gen_experiment_name`). Default value: base

- `detector`: Specifies the hydra configuration file for the detector. See [the list of supported detector](description_lists/supported_detectors.md) for the list of supported detectors. Default value: "detection_base"

- `generator_name`: See [how to generate page](how_to_generate.md). Note that we don't assume that we know the LLM used for generation, we specify the generator here because it allow us to track the data folder name for the corresponding generated dataset. Default value: "qwen2_chat_0_5B"

- `test_res_dir`: Specify the base folder where the detection results should be saved. Default value: "detection_test_results"

- `weights_checkpoint`: Sets the .pt file for the folder used to load the weights of the detector, in case we fine-tuned a detector on a specific dataset locally.
```

## Test a watermark detector

```sh
generator_name="qwen2_chat_0_5B"
attack="generation_base"
watermark_scheme="watermark_kgw"
detector="watermark_detector"
batch_size=128
gen_experiment_name="compare_watermark"
detection_experiment_name="compare_watermark"
test_res_dir="detection_test_results"

test_detector detection=$detector generation=$attack generation.generator_name=generator_name watermark=$watermark_scheme generation.experiment_name=$gen_experiment_name \
        detection.experiment_name=$detection_experiment_name  \
        detection.test_res_dir=$test_res_dir detection.batch_size=$batch_size             
```

```{admonition} Configure the benchmark
:class: hint

The parameters here are the same as above except for the following:

- `watermark_scheme`: Specify the hydra configuration file for the watermarking scheme that was used for generation. It allows us to load the watermark detector with the correct configuration that will be tested. Default value: watermark_kgw. Default value: "watermark_base"
Note: We need to set the `detector` field to "watermark_detector" to be able to test the watermark detector.
```


## Specific parameters for each detectors
- **TODO**