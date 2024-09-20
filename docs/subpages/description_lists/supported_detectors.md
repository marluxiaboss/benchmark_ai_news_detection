# List of supported detectors
See `detector_benchmark/detector/detector_loader.py` and `detector_benchmark/conf/detector`

```{admonition} Supported LLM detectors
- "electra_large": Electra large from ... Need to provide the .pt file of weights for the trained model.
- "fast_detect_gpt": Fast-DetectGPT zero-shot detector from [[2310.05130\]Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature (arxiv.org)](http://arxiv.org/abs/2310.05130).
- "gpt_zero": GPTZero detector. Need to set the API key as an environment variable called "GPT_ZERO_API_KEY".
- "watermark_detector": Detector used for detecting a watermark (the detector depends on the watermarking scheme used for generation).
```