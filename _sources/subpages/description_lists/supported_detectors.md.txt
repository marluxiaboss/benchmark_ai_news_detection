# List of supported detectors
See `detector_benchmark/detector/detector_loader.py` and `detector_benchmark/conf/detector`

```{admonition} Supported LLM detectors
- "electra_large": Electra large from ... Need to provide the .pt file of weights for the trained model.
- "fast_detect_gpt": Fast-DetectGPT zero-shot detector from ...
- "gpt_zero": GPTZero detector. Need to set the API key as an environment variable called "GPT_ZERO_API_KEY".
- "watermark_detector": Detector used for detecting a watermark (the detector depends on the watermarking scheme used for generation).
```