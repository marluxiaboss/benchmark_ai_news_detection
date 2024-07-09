# LLM-generated news benchmark
The purpose of this benchmark is to evaluate LLM detectors, especially against evasion attacks. So far, the benchmark is based on the detection of short LLM-generated news articles,
but it can be expanded to cover different detection tasks.  
The main consideration is to make the benchmark easy to extend with different datasets, detectors and evasion attacks

## How to test a new detector on the benchmark
To add a new detectors, 3 files needs to be added/modified:
- a .py file in `detector` containing the class for the new detector, extending the base detector class.
- `detector/detector_loader.py` needs to be modified to be able to load the detector from the .py file created above
- a configuration file under `conf/detection` needs to be added to configure the new detector.

The added detector class should have at leas a `detect` function with the following signature:
```
def detect(self, texts: list, batch_size: int, detection_threshold: int) -> list:
```
Where texts

## How to add a dataset

## How to add an attack


## Class descriptions


## Future work


