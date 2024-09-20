# Getting Started

## Installation

0. Create a conda environment (highly recommended to avoid compatibility issues) and activate it

``` sh
conda create -n "llm_detector" python=3.10.12 ipython
conda activate llm_detector
```

1. Install pytorch with a version compatible with your CUDA driver

For CUDA version 11.8 (check your version with nvidia-smi and see [PyTorch's website](https://pytorch.org/)):
``` sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Clone and install the package

```sh
git clone git@github.com:marluxiaboss/benchmark_ai_news_detection.git
pip install -e .
```
**TODO** Add to PyPI

## Generate the benchmark

``` sh
attack="generation_base"
watermark_scheme="watermark_base"

create_dataset generation=$attack watermark=$watermark_scheme
```

Choose the attack in the available list [here](description_lists/supported_attacks.md).  
See [here](how_to_generate.md) for all the other parameters for generating the data.

## Test a detector on the created benchmark

```sh
detector="fast_detect_gpt"

test_detector detection=$detector 
```

Choose the detector in the available list [here](description_lists/supported_detectors.md).  
See [here](how_to_test_detector.md) for all the other parameters for testing a detector on the generated benchmark.
