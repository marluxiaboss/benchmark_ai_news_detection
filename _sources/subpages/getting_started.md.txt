# Getting Started

Welcome to the Getting Started guide for My Project. Here, you'll find all the information you need to get up and running.

## Installation

Follow these steps to install the project:

```sh
pip install my_project
```

## Generate the benchmark

```
attack="generation_base"
watermark_scheme="watermark_base"

python create_dataset.py generation=$attack watermark=$watermark_scheme
```

Choose the attack in the available list [here](description_lists/supported_attacks.md).  
See [here](how_to_generate.md) for all the other parameters for generating the data.

## Test a detector
