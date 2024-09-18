---
hide-toc: true
---

# LLM Detector benchmark

Documentation for [detector benchmark](https://github.com/marluxiaboss/benchmark_ai_news_detection).

## Features
- Generating an (adversarial) benchmark with a specific configuration, used for testing detectors.
- Detectors and watermark detection benchmarking (adversarial + non-adversarial)
- **Modularity**: possible to add new datasets, detectors, attacks and watermarking schemes without much effort

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started
---
subpages/getting_started.md
```

```{toctree}
:hidden:
:maxdepth: 3
:caption: Contents

subpages/how_to_generate.md
subpages/how_to_test_detector.md
subpages/how_to_add_index.md
subpages/supported_list_index.md
subpages/details.md
```