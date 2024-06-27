import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import adapters


class AugmentedRobertaClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config):
      super().__init__()
      self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
      classifier_dropout = (
          config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
      )
      self.dropout = nn.Dropout(classifier_dropout)
      self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
      self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

  def forward(self, features, **kwargs):
      x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
      x = self.dropout(x)
      x = self.dense1(x)
      x = torch.tanh(x)
      x = self.dropout(x)
      x = self.dense2(x)
      x = torch.tanh(x)
      x = self.dropout(x)
      x = self.out_proj(x)
      return x

class LLMDetector(nn.Module):
  def __init__(self, bert_model, tokenizer, num_classes, freeze_bert=False, device=None, add_more_layers=False):
    super().__init__()

    self.tokenizer = tokenizer

    # bert should already be trained
    self.bert = bert_model
    
    if device is None:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = device

    # set num_classes
    self.num_classes = num_classes

    if freeze_bert:
      # freeze the base model
      self.freeze_base(self.bert)

    if add_more_layers:
      # add more layers to the classifier
      self.bert = self.add_more_layers(self.bert)

  def forward(self, text):

    # tokenize text using the tokenizer
    output = self.tokenizer(text, return_tensors="pt")
    input_ids = output["input_ids"].to(self.device)
    logits = self.bert(input_ids)["logits"]

    # apply sigmoid to get probabilities of each class
    output = torch.sigmoid(logits)
    return output
  
  @staticmethod
  def freeze_base(bert_model):
    for name, param in bert_model.named_parameters():
      if not name.startswith("classifier") and not name.startswith("classification_head"):
        param.requires_grad = False

  @staticmethod
  def add_more_layers(bert_model):
    """
    bert_model.classifier = nn.Sequential(
      nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size),
      nn.Dropout(0.1),
      nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size),
      nn.Dropout(0.1),
      nn.Linear(bert_model.config.hidden_size, num_classes)
    )
    """
    bert_model.classifier = AugmentedRobertaClassificationHead(bert_model.config)
    return bert_model
  
  @staticmethod
  def use_adapter(bert_model, device, train_adapter=False):
    adapters.init(bert_model)
    config = adapters.BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
    bert_model.add_adapter("fake_true_detection", config=config)

    if train_adapter:
      bert_model.train_adapter("fake_true_detection")
    else:
      bert_model.set_active_adapters("fake_true_detection")
    bert_model.to(device)


