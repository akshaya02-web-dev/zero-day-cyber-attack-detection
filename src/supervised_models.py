"""Supervised model definitions for known attack detection.

This file contains placeholder model trainers for Logistic Regression,
Random Forest, and XGBoost.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class SupervisedModelTrainer:
    """Placeholder trainer for known attack detection models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series):
        """Train a logistic regression classifier."""
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X, y)
        return model

    def train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """Train a random forest classifier."""
        model = RandomForestClassifier(random_state=self.random_state, n_estimators=200)
        model.fit(X, y)
        return model

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """Train an XGBoost classifier."""
        model = XGBClassifier(random_state=self.random_state, n_estimators=200, eval_metric="logloss")
        model.fit(X, y)
        return model
