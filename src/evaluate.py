"""Evaluation utilities for classification performance.

This module will calculate standard metrics and support both known and unknown
threat evaluations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


class Evaluator:
    """Compute classification metrics and confusion details."""

    def __init__(self):
        pass

    def compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        """Return a dictionary of common classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    def summarize_unknown_detection(self, actual_labels: pd.Series, predicted_unknown: pd.Series) -> dict[str, float]:
        """Compute a simple unknown-threat summary.

        TODO:
            - provide precision, recall, and detection rates for unknown traffic
        """
        return {
            "unknown_detection_rate": 0.0,
            "false_positive_unknown_rate": 0.0,
        }
