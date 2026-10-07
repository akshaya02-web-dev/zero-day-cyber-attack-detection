"""Train-only Random Forest feature selection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:
    """Rank numeric features with a Random Forest fitted on training data."""

    def __init__(self, top_n_features: int = 20, random_state: int = 42):
        if top_n_features < 1:
            raise ValueError("top_n_features must be at least 1.")
        self.top_n_features = top_n_features
        self.random_state = random_state
        self.selected_features: list[str] = []
        self.importance_report = pd.DataFrame()
        self.model: RandomForestClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Fit importance ranking using training data only."""
        if X.empty:
            raise ValueError("Cannot select features from an empty dataframe.")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        importances = np.asarray(self.model.feature_importances_, dtype=float)
        self.importance_report = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances,
        }).sort_values("Importance", ascending=False, ignore_index=True)
        selected_count = min(self.top_n_features, len(self.importance_report))
        self.selected_features = self.importance_report.head(selected_count)["Feature"].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep the selected feature columns in their fitted order."""
        if not self.selected_features:
            raise ValueError("FeatureSelector must be fitted before transform.")
        missing = sorted(set(self.selected_features) - set(X.columns))
        if missing:
            raise ValueError(f"Selected features are missing from input: {missing}")
        return X[self.selected_features].copy()

    def get_selected_features(self) -> list[str]:
        """Return selected feature names ordered by Random Forest importance."""
        return self.selected_features
