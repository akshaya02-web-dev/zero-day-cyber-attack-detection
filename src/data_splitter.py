"""Dataset splitting utilities for the cybersecurity ML pipeline.

This module handles the data preparation stage by dividing the cleaned dataset
into training, validation, and testing subsets while preserving class balance,
attack categories, and strict no-leakage guarantees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ProjectConfig, get_default_config
from src.utils import configure_logger


@dataclass
class SplitResult:
    """Structured result for train/validation/test separation."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    attack_categories_train: pd.Series
    attack_categories_val: pd.Series
    attack_categories_test: pd.Series
    metadata: dict


class DataSplitter:
    """Create leakage-safe train, validation, and test datasets."""

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or get_default_config()
        self.random_state = self.config.splitting.random_state
        self.train_ratio = self.config.splitting.train_ratio
        self.validation_ratio = self.config.splitting.validation_ratio
        self.test_ratio = self.config.splitting.test_ratio
        self.logger = configure_logger("cyber_threat_splitter", logging.INFO)

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series, attack_categories: pd.Series) -> None:
        """Check that the inputs are valid and aligned for splitting."""
        if X is None or y is None or attack_categories is None:
            raise ValueError("X, y, and attack_categories must all be provided.")
        if len(X) != len(y) or len(X) != len(attack_categories):
            raise ValueError(
                "X, y, and attack_categories must have the same number of rows. "
                f"Got lengths {len(X)}, {len(y)}, and {len(attack_categories)}."
            )
        if X.empty:
            raise ValueError("Input feature matrix is empty.")
        if y.nunique() < 2:
            raise ValueError("Binary target must contain both classes for stratified splitting.")

    def _assert_no_overlap(self, left_index: pd.Index, right_index: pd.Index, left_name: str, right_name: str) -> None:
        """Verify that two datasets have no overlapping indices."""
        overlap = left_index.intersection(right_index)
        if len(overlap) > 0:
            raise ValueError(f"Data leakage detected: {left_name} and {right_name} share indices: {list(overlap[:10])}")

    def _assert_row_alignment(self, X: pd.DataFrame, y: pd.Series, attack_categories: pd.Series, label: str) -> None:
        """Ensure aligned row counts for a split."""
        if len(X) != len(y):
            raise ValueError(f"{label}: feature matrix and target length mismatch ({len(X)} vs {len(y)}).")
        if len(X) != len(attack_categories):
            raise ValueError(f"{label}: feature matrix and attack category length mismatch ({len(X)} vs {len(attack_categories)}).")

    def _compute_split_counts(self, total_samples: int) -> tuple[int, int, int]:
        """Compute 70/15/15 sample counts while preserving approximations."""
        train_count = int(total_samples * self.train_ratio)
        remaining = total_samples - train_count
        val_count = int(total_samples * self.validation_ratio)
        test_count = total_samples - train_count - val_count

        if test_count < 1:
            test_count = 1
            val_count = remaining - 1 if remaining > 1 else 0

        return train_count, val_count, test_count

    def split(self, X: pd.DataFrame, y: pd.Series, attack_categories: pd.Series) -> SplitResult:
        """Split the preprocessed dataset into train, validation, and test sets.

        The split is performed in two phases:
        1. 70% train, 30% temporary
        2. 50/50 split of the temporary set into validation/test
        """
        self._validate_inputs(X, y, attack_categories)

        X = X.copy()
        y = pd.Series(y).copy()
        attack_categories = pd.Series(attack_categories).copy()

        X.index = pd.RangeIndex(len(X))
        y.index = X.index
        attack_categories.index = X.index

        total_samples = len(X)
        train_count, val_count, test_count = self._compute_split_counts(total_samples)

        X_train, X_temp, y_train, y_temp, attack_train, attack_temp = train_test_split(
            X,
            y,
            attack_categories,
            train_size=self.train_ratio,
            random_state=self.random_state,
            stratify=y,
        )

        X_val, X_test, y_val, y_test, attack_val, attack_test = train_test_split(
            X_temp,
            y_temp,
            attack_temp,
            test_size=0.5,
            random_state=self.random_state,
            stratify=y_temp,
        )

        self._assert_row_alignment(X_train, y_train, attack_train, "TRAIN")
        self._assert_row_alignment(X_val, y_val, attack_val, "VALIDATION")
        self._assert_row_alignment(X_test, y_test, attack_test, "TEST")

        self._assert_no_overlap(X_train.index, X_val.index, "TRAIN", "VALIDATION")
        self._assert_no_overlap(X_train.index, X_test.index, "TRAIN", "TEST")
        self._assert_no_overlap(X_val.index, X_test.index, "VALIDATION", "TEST")

        self.logger.info("================ DATA SPLITTING ================")
        self.logger.info("Total Samples: %s", total_samples)
        self.logger.info("TRAINING SET - Samples: %s, Normal: %s, Attack: %s", len(X_train), int((y_train == 0).sum()), int((y_train == 1).sum()))
        self.logger.info("VALIDATION SET - Samples: %s, Normal: %s, Attack: %s", len(X_val), int((y_val == 0).sum()), int((y_val == 1).sum()))
        self.logger.info("TEST SET - Samples: %s, Normal: %s, Attack: %s", len(X_test), int((y_test == 0).sum()), int((y_test == 1).sum()))
        self.logger.info("Original Attack Categories: %s", attack_categories.value_counts().to_dict())

        print("\n================ DATA SPLITTING ================")
        print(f"Total Samples: {total_samples}")
        print("\nTRAINING SET")
        print(f"Samples: {len(X_train)}")
        print(f"Normal Samples: {int((y_train == 0).sum())}")
        print(f"Attack Samples: {int((y_train == 1).sum())}")
        print("\nVALIDATION SET")
        print(f"Samples: {len(X_val)}")
        print(f"Normal Samples: {int((y_val == 0).sum())}")
        print(f"Attack Samples: {int((y_val == 1).sum())}")
        print("\nTEST SET")
        print(f"Samples: {len(X_test)}")
        print(f"Normal Samples: {int((y_test == 0).sum())}")
        print(f"Attack Samples: {int((y_test == 1).sum())}")
        print("\nOriginal Attack Categories:")
        print("TRAIN:")
        print(attack_train.value_counts().to_dict())
        print("\nVALIDATION:")
        print(attack_val.value_counts().to_dict())
        print("\nTEST:")
        print(attack_test.value_counts().to_dict())
        print("\nNO DATA LEAKAGE DETECTED")
        print("DATA SPLITTING COMPLETED SUCCESSFULLY")

        metadata = {
            "total_samples": total_samples,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "normal_train": int((y_train == 0).sum()),
            "attack_train": int((y_train == 1).sum()),
            "normal_val": int((y_val == 0).sum()),
            "attack_val": int((y_val == 1).sum()),
            "normal_test": int((y_test == 0).sum()),
            "attack_test": int((y_test == 1).sum()),
            "train_attack_distribution": attack_train.value_counts().to_dict(),
            "val_attack_distribution": attack_val.value_counts().to_dict(),
            "test_attack_distribution": attack_test.value_counts().to_dict(),
            "random_state": self.random_state,
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
        }

        return SplitResult(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            attack_categories_train=attack_train,
            attack_categories_val=attack_val,
            attack_categories_test=attack_test,
            metadata=metadata,
        )

    def split_train_test(self, X: pd.DataFrame, y: pd.Series, attack_categories: pd.Series):
        """Backward-compatible helper for simple train/test splitting."""
        return self.split(X, y, attack_categories)
