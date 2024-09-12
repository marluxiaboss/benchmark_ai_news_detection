import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json
from time import gmtime, strftime
import logging
import sys
import os
from datasets import load_from_disk
from abc import ABC, abstractmethod
import pandas as pd

from .pipeline_utils import *


class ExperimentPipeline(ABC):
    """
    Abstract class for an experiment pipeline.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run_pipeline(self):
        pass
