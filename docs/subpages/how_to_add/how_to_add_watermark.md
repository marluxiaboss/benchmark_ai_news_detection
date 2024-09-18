# How to add a watermarking scheme on the benchmark
Credits to <https://github.com/THU-BPM/MarkLLM> for most of the watermarking code structure and classes.

To add a watermarking scheme, 4 files need to be added/modified inside the:
- add: a {watermarking_scheme}.py file inside its own `detector_benchmark/watermark/{watermarking_scheme}` folder.
- add: a corresponding `__init__.py` inside the same folder
- add: a configuration filde under `detector_benchmark/conf/watermark` to configure the watermarking scheme.
- modify: the `WATERMARK_MAPPING_NAMES` dictionary variable inside `detector_benchmark/watermark/auto_watermark.py`.

See examples of already added watermarking schemes to understand what functions the {watermarking_scheme}.py should implement. The core of the watermarking scheme is a class `{watermarking_scheme}` inheriting from LogitsProcessor having at least a `__init__` constructor method and a `__call__` method with the following signature:

``` python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
```

Taking as input a context (input_ids) and the logits (scores) as returned by the LLM. The watermarking scheme then modifies the logits and returns the new logits.