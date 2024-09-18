# How to add a new detector on the detection benchmark

To add a new detector, 3 files need to be added/modified:
- add: a .py file in `detector_benchmark/detector` containing the class for the new detector, inheriting the base detector class.
- add: a configuration file under `detector_benchmark/conf/detection` to configure the new detector.
- modify: `detector_benchmark/detector/detector_loader.py` to be able to load the detector from the .py file created above.

The added detector class should have at least a constructor and a `detect` function with the following signature:

``` python
def detect(self, texts: list, batch_size: int, detection_threshold: float) -> tuple[list[int], list[float], list[int]]:
```

Where `texts` is a list of text to be detected as LLM-written or not and it should return the following variables:
- `preds`: a list of 0s (human-written) and 1s (LLM-generated) with the predicted labels computed as argmax of the logits of both classes
- `logits_pos_class`: the list of logits or probabilities for the positive class (softmaxed or not) 
- `preds_at_threshold`: same as `preds` but where the prediction is made using a threshold on the logits rather than argmax  

You can look at already added detectors for guiding examples.