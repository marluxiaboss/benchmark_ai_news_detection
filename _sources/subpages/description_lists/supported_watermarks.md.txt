# List of supported LLMs for generation
See `detector_benchmark/watermark` and `detector_benchmark/conf/watermark`


```{admonition} Supported Watermarking schemes
- "watermark_base": No watermarking
- "watermark_dip": DIP watermarking scheme from ...
- "watermark_exp": EXP watermarking scheme from ...
- "watermark_kgw" Watermarking scheme from Kirchenbauer et al.
- "watermark_sir": SIR watermarking scheme.
- "watermark_sweet": SWEET watermarking scheme.
```

Note: Since the watermark structure is taken from <https://github.com/THU-BPM/MarkLLM>. We can easily add a new watermarking scheme if there is a new one available in the provided link.