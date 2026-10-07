"""Anomaly detection module.

This component is responsible for identifying unknown or anomalous cyber
threat patterns using Isolation Forest trained primarily on normal traffic.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Placeholder anomaly detection wrapper around Isolation Forest."""

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, normal_data: pd.DataFrame):
        """Train the Isolation Forest on predominantly normal traffic."""
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(normal_data)
        return self.model

    def score_samples(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores for the provided samples."""
        if self.model is None:
            raise ValueError("Anomaly detector must be fitted before scoring.")
        scores = self.model.decision_function(X)
        return pd.Series(scores)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict inlier vs outlier labels."""
        if self.model is None:
            raise ValueError("Anomaly detector must be fitted before prediction.")
        predictions = self.model.predict(X)
        return pd.Series(predictions)
