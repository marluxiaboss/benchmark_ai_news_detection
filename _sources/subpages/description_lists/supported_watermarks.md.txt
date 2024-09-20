# List of supported watermarking schemes
See `detector_benchmark/watermark` and `detector_benchmark/conf/watermark` and <https://github.com/THU-BPM/MarkLLM> for the source of most of the watermarking schemes.


```{admonition} Supported Watermarking schemes
- "watermark_base": No watermarking
- "watermark_dip": DIP watermarking scheme from [\[2310.07710\] A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models (arxiv.org)](https://arxiv.org/abs/2310.07710)
- "watermark_exp": EXP watermarking scheme from <https://www.scottaaronson.com/talks/watermark.ppt>
- "watermark_kgw" Watermarking scheme from [\[2301.10226\] A Watermark for Large Language Models (arxiv.org)](https://arxiv.org/abs/2301.10226)
- "watermark_sir": SIR watermarking scheme from [\[2310.06356\] A Semantic Invariant Robust Watermark for Large Language Models (arxiv.org)](https://arxiv.org/abs/2310.06356).
- "watermark_sweet": SWEET watermarking scheme from [\[2305.15060\] Who Wrote this Code? Watermarking for Code Generation (arxiv.org)](https://arxiv.org/abs/2305.15060).
```

Note: Since the watermark code structure is taken from <https://github.com/THU-BPM/MarkLLM>. We can easily add a new watermarking scheme if there is a new one available in the provided link.