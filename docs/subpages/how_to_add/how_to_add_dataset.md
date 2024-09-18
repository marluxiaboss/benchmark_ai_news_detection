# How to add a dataset
First, let's describe the datasets format used for testing the detectors. The datasets are huggingface Datasets where each samples has the following fields:
- `label`: the label of the text (0 for human written, 1 for AI-written)
- `text`: contains the full text of the sample
- `prefix`: prefix from the human written texts used to generated the AI-written text. For each prefix, we always have the corresponding true sample (label 0) and the fake one (label 1) sharing the same prefix.
- `generation_config`: the config (data from the config file in `conf`) used to generate the text AI-generated text for the dataset
- `watermark_config`: the watermarking config used to generate the text (which watermarking algorithm if any,...)

Now to add a different dataset than the existing ones, we need to add dataset loader class inheriting the base `FakeTruePairsDataLoader` class inside the `dataset_loader` folder.
This class should implement the `load_data` function with the following signature:

```python
def load_data(self) -> DatasetDict:
```

This function should return a DatasetDict (huggingface dataset format) with a train, eval and test split and respecting the dataset format. 
To see how to apply the correct format to the dataset, see the existing dataset loaders.

Note: the load_data returns a dataset where the fake samples (label 1) have an empty `text` and no `generation_config` nor `watermark_config` fields since the AI texts have not been generated yet. The only fields that should be fully field for AI texts are the label and the prefix.