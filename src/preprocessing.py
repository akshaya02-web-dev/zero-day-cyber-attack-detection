"""Preprocessing utilities for cybersecurity data.

This module cleans and prepares network traffic data for downstream ML tasks,
while preserving original attack labels for known-versus-unknown experiments.
The preprocessing is intentionally modular and safe so it can support known
attack detection and unknown threat detection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import ProjectConfig, build_config_from_dict, get_default_config
from src.utils import configure_logger


class Preprocessor:
    """Clean and normalize cybersecurity data for downstream modeling tasks."""

    def __init__(self, config: ProjectConfig | dict[str, Any] | None = None):
        if config is None:
            self.config = get_default_config()
        elif isinstance(config, dict):
            self.config = build_config_from_dict(config)
        else:
            self.config = config

        self.logger = configure_logger("cyber_threat_preprocessor", logging.INFO)
        self.feature_columns: list[str] | None = None
        self._resolved_label_column: str | None = None

    def _clean_column_name(self, column_name: Any) -> str:
        """Normalize dataframe column names by stripping whitespace and collapsing repeated spaces."""
        if not isinstance(column_name, str):
            return str(column_name)

        cleaned = column_name.strip()
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _make_unique_column_names(self, columns: list[Any]) -> list[str]:
        """Ensure column names are unique after cleaning."""
        seen: dict[str, int] = {}
        unique_columns: list[str] = []

        for column in columns:
            cleaned = self._clean_column_name(column)
            if cleaned in seen:
                seen[cleaned] += 1
                cleaned = f"{cleaned}_{seen[cleaned]}"
            else:
                seen[cleaned] = 0
            unique_columns.append(cleaned)

        return unique_columns

    def _normalize_attack_category(self, value: Any) -> str:
        """Normalize an attack label while preserving the human-readable category."""
        if pd.isna(value):
            return "UNKNOWN"
        cleaned = str(value).strip()
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate the input dataframe before preprocessing."""
        if df is None:
            raise ValueError("Input dataframe is None.")
        if df.empty:
            raise ValueError("Input dataset is empty.")

        label_column = self.config.data.label_column
        normalized_columns = {
            self._clean_column_name(column).lower(): column for column in df.columns
        }
        matched_label = normalized_columns.get(label_column.lower())

        if matched_label is None:
            matches = [
                column for column in df.columns
                if self._clean_column_name(column).lower() == label_column.lower()
            ]
            if not matches:
                raise ValueError(
                    f"Configured label column '{label_column}' is missing from the dataset. "
                    f"Available columns: {list(df.columns)}"
                )
            matched_label = matches[0]

        if matched_label != label_column:
            df = df.rename(columns={matched_label: label_column})

        self._resolved_label_column = label_column

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply safe, non-destructive cleaning to a copy of the dataset."""
        cleaned = df.copy()

        cleaned.columns = self._make_unique_column_names(list(cleaned.columns))

        def normalize_string_value(value: Any) -> Any:
            if isinstance(value, str):
                return value.strip()
            return value

        cleaned = cleaned.apply(lambda col: col.map(normalize_string_value))
        cleaned = cleaned.replace([float("inf"), float("-inf")], float("nan"))

        return cleaned

    def prepare(self, df: pd.DataFrame) -> dict[str, Any]:
        """Prepare the cybersecurity dataset for ML by cleaning, validating and labeling it.

        Returns a dictionary with:
        - X: feature matrix excluding the label and attack-category columns
        - binary_y: binary target (0 for normal, 1 for attack)
        - attack_categories: normalized attack labels preserved for later use
        - metadata: summary information about cleaning and label distribution
        """
        self._validate_input(df)
        df = df.rename(columns={
            column: self.config.data.label_column if self._clean_column_name(column).lower() == self.config.data.label_column.lower() else column
            for column in df.columns
        })

        original_shape = df.shape
        numeric_input = df.select_dtypes(include=[np.number])
        infinity_values_replaced = int(np.isinf(numeric_input.to_numpy()).sum()) if not numeric_input.empty else 0
        cleaned = self.clean_data(df)

        missing_before = int(cleaned.isna().sum().sum())
        duplicate_count = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        label_column = self.config.data.label_column
        attack_column = self.config.data.attack_column

        if attack_column not in cleaned.columns:
            cleaned[attack_column] = cleaned[label_column].copy()

        cleaned[attack_column] = cleaned[attack_column].map(self._normalize_attack_category)

        if label_column not in cleaned.columns:
            raise ValueError(f"Configured label column '{label_column}' is missing after cleaning.")

        cleaned[label_column] = cleaned[label_column].map(self._normalize_attack_category)

        normal_names = {str(name).strip().lower() for name in self.config.data.normal_label_names}
        binary_y = cleaned[label_column].astype(str).apply(
            lambda value: 0 if value.strip().lower() in normal_names else 1
        )

        if cleaned.empty:
            raise ValueError("Resulting dataset is empty after preprocessing.")
        if binary_y.nunique() < 2:
            raise ValueError(
                "Only one class remains after binary label mapping. "
                "The dataset must contain both normal and attack samples."
            )

        feature_columns = [
            column for column in cleaned.columns if column not in {label_column, attack_column}
        ]
        if not feature_columns:
            raise ValueError(
                "No feature columns remain after excluding the configured label and attack columns."
            )

        X = cleaned[feature_columns].copy()
        coerced_values = 0
        for column in feature_columns:
            original_values = X[column].copy()
            converted = pd.to_numeric(original_values, errors="coerce")
            coerced_values += int((original_values.notna() & converted.isna()).sum())
            X[column] = converted

        all_null_columns = [column for column in X.columns if X[column].isna().all()]
        if all_null_columns:
            X = X.drop(columns=all_null_columns)

        if X.empty:
            raise ValueError("No usable numeric feature columns remain after cleaning.")

        feature_columns = X.columns.tolist()
        attack_categories = cleaned[attack_column].copy()

        metadata = {
            "original_shape": original_shape,
            "final_shape": cleaned.shape,
            "duplicates_removed": duplicate_count,
            "infinity_values_replaced": infinity_values_replaced,
            "missing_value_count": int(X.isna().sum().sum()),
            "missing_values_before_handling": missing_before,
            "non_numeric_values_coerced": coerced_values,
            "all_null_features_removed": all_null_columns,
            "normal_samples": int((binary_y == 0).sum()),
            "attack_samples": int((binary_y == 1).sum()),
            "attack_category_distribution": cleaned[attack_column].value_counts().to_dict(),
            "label_distribution": binary_y.value_counts().to_dict(),
            "label_column": label_column,
            "attack_column": attack_column,
            "feature_columns": feature_columns,
        }

        self.logger.info("Original dataset shape: %s", original_shape)
        self.logger.info("Final dataset shape: %s", cleaned.shape)
        self.logger.info("Duplicate rows removed: %s", duplicate_count)
        self.logger.info("Missing value count: %s", missing_before)
        self.logger.info("Normal samples: %s", metadata["normal_samples"])
        self.logger.info("Attack samples: %s", metadata["attack_samples"])
        self.logger.info("Original attack distribution: %s", metadata["attack_category_distribution"])

        return {
            "X": X,
            "binary_y": binary_y,
            "attack_categories": attack_categories,
            "metadata": metadata,
        }

    def preprocess(self, df: pd.DataFrame) -> dict[str, Any]:
        """Alias for prepare() to keep the API readable and reusable."""
        return self.prepare(df)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit metadata required for downstream preprocessing steps."""
        self.feature_columns = [
            column for column in df.columns if column not in {self.config.data.label_column, self.config.data.attack_column}
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured cleaning logic to a dataframe without altering the original data."""
        return self.clean_data(df)

    def fit_transform(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit and transform the dataset in one step."""
        self.fit(df)
        return self.prepare(df)
