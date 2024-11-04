# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# detector.py
# Description: Implementation of SynthID watermark detectors
# ============================================

import os
import abc
import tqdm
import json
import torch
import numpy as np
from .detector_bayesian_torch import RawBayesianDetector

# from evaluation.dataset import C4Dataset


class SynthIDDetector(abc.ABC):
    """Base class for SynthID watermark detectors.

    This class defines the interface that all SynthID watermark detectors must implement.
    Subclasses should override the detect() method to implement specific detection algorithms.
    """

    def __init__(self):
        """Initialize the detector."""
        pass

    @abc.abstractmethod
    def detect(self, g_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect watermark presence in the given g-values.

        Args:
            g_values: Array of shape [batch_size, seq_len, watermarking_depth] containing
                the g-values computed from the text.
            mask: Binary array of shape [batch_size, seq_len] indicating which g-values
                should be used in detection. g-values with mask value 0 are discarded.

        Returns:
            Array of shape [batch_size] containing detection scores, where higher values
            indicate stronger evidence of watermarking.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement detect()")


class MeanDetector(SynthIDDetector):
    def detect(self, g_values, mask):
        """
        Args:
            g_values: shape [batch_size, seq_len, watermarking_depth]
            mask: shape [batch_size, seq_len]
        Returns:
            scores: shape [batch_size]
        """
        watermarking_depth = g_values.shape[-1]
        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
            watermarking_depth * num_unmasked
        )


class WeightedMeanDetector(SynthIDDetector):
    def detect(
        self,
        g_values: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """Computes the Weighted Mean score.

        Args:
            g_values: g-values of shape [batch_size, seq_len, watermarking_depth]
            mask: A binary array shape [batch_size, seq_len] indicating which g-values
                should be used. g-values with mask value 0 are discarded
            weights: array of non-negative floats, shape [watermarking_depth]. The
                weights to be applied to the g-values. If not supplied, defaults to
                linearly decreasing weights from 10 to 1

        Returns:
            Weighted Mean scores, of shape [batch_size]. This is the mean of the
            unmasked g-values, re-weighted using weights.
        """
        watermarking_depth = g_values.shape[-1]

        if weights is None:
            weights = np.linspace(start=10, stop=1, num=watermarking_depth)

        # Normalise weights so they sum to watermarking_depth
        weights *= watermarking_depth / np.sum(weights)

        # Apply weights to g-values
        g_values = g_values * np.expand_dims(weights, axis=(0, 1))

        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
            watermarking_depth * num_unmasked
        )


def get_detector(detector_name: str, logits_processor):
    if detector_name == "mean":
        return MeanDetector()
    elif detector_name == "weighted_mean":
        return WeightedMeanDetector()
    else:
        raise ValueError(f"Detector {detector_name} not found.")
