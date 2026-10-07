"""Tests for the dataset splitting stage."""

from __future__ import annotations

import pandas as pd

from src.config import ProjectConfig
from src.data_splitter import DataSplitter


def test_data_splitter_produces_stratified_no_leakage_splits():
    """Ensure the dataset is split into train/val/test sets with no overlap."""
    config = ProjectConfig()
    splitter = DataSplitter(config)

    X = pd.DataFrame(
        {
            "feature_1": list(range(100)),
            "feature_2": list(range(100, 200)),
        }
    )
    y = pd.Series([0, 1] * 50)
    attack_categories = pd.Series(["BENIGN" if label == 0 else "DoS" for label in y])

    result = splitter.split(X, y, attack_categories)

    assert len(result.X_train) == 70
    assert len(result.X_val) == 15
    assert len(result.X_test) == 15

    assert len(result.X_train) == len(result.y_train)
    assert len(result.X_val) == len(result.y_val)
    assert len(result.X_test) == len(result.y_test)

    assert len(result.attack_categories_train) == len(result.X_train)
    assert len(result.attack_categories_val) == len(result.X_val)
    assert len(result.attack_categories_test) == len(result.X_test)

    train_idx = set(result.X_train.index)
    val_idx = set(result.X_val.index)
    test_idx = set(result.X_test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    assert result.y_train.value_counts().to_dict().get(0, 0) > 0
    assert result.y_train.value_counts().to_dict().get(1, 0) > 0
    assert result.y_val.value_counts().to_dict().get(0, 0) > 0
    assert result.y_val.value_counts().to_dict().get(1, 0) > 0
    assert result.y_test.value_counts().to_dict().get(0, 0) > 0
    assert result.y_test.value_counts().to_dict().get(1, 0) > 0
