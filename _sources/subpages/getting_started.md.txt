# Getting Started

Welcome to the Getting Started guide for My Project. Here, you'll find all the information you need to get up and running.

## Installation

Clone the repository for running the benchmark generation and detection tests:

```sh
git clone git@github.com:marluxiaboss/benchmark_ai_news_detection.git
```

## Creating the environment
- For all scripts, except the ones using BiGGen-Bench (`bash_scripts/big_gen_bench`): Create the conda environment llm_detector by running `bash_scripts/create_envs/create_llm_detector_env.sh`.
- For scripts using BiGGen-Bench: Create the conda environment big_gen_bench by running `bash_scripts/create_envs/create_big_gen_bench_env.sh`.

## Generate the benchmark

``` sh
attack="generation_base"
watermark_scheme="watermark_base"

python create_dataset.py generation=$attack watermark=$watermark_scheme
```

Choose the attack in the available list [here](description_lists/supported_attacks.md).  
See [here](how_to_generate.md) for all the other parameters for generating the data.

## Test a detector on the created benchmark

```sh
detector="fast_detect_gpt"

python test_detector.py detection=$detector 
```

Choose the detector in the available list [here](description_lists/supported_detectors.md).  
See [here](how_to_test_detector.md) for all the other parameters for testing a detector on the generated benchmark.
