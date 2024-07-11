# LLM-generated news benchmark  :mag: 📰
The purpose of this benchmark is to evaluate LLM detectors, especially against evasion attacks. So far, the benchmark is based on the detection of short LLM-generated news articles, but it can be extended to cover different detection tasks.  
The main consideration is to make the benchmark easy to extend with different datasets, detectors and evasion attacks

## How to test a new detector on the benchmark
To add a new detectors, 3 files needs to be added/modified:
- a .py file in `detector` containing the class for the new detector, extending the base detector class.
- `detector/detector_loader.py` needs to be modified to be able to load the detector from the .py file created above.
- a configuration file under `conf/detection` needs to be added to configure the new detector.

The added detector class should have at least a constructor and a `detect` function with the following signature:
```
def detect(self, texts: list, batch_size: int, detection_threshold: float) -> tuple[list[int], list[float], list[int]]:
```
Where `texts` is a list of text to be detected as LLM-written or not and it should return the following variables:
- `preds`: a list of 0s (human-written) and 1s (LLM-generated) with the predicted labels computed as argmax of the logits of both classes
- `logits_pos_class`: the list of logits or probabilities for the positive class (softmaxed or not) 
- `preds_at_threshold`: same as `preds` but where the prediction is made using a threshold on the logits rather than argmax  

You can look at already added detectors for guiding examples.

## How to add a dataset

## How to add an attack

## Class descriptions


## Future work


