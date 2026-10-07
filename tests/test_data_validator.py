"""Tests for the dataset validation and reporting stage."""

from __future__ import annotations

import pandas as pd

from src.data_splitter import DataSplitter
from src.data_validator import DataValidator


def test_data_validator_confirms_clean_and_split_dataset():
    """Ensure the validator accepts a valid processed dataset and splits."""
    X = pd.DataFrame(
        {
            "feature_1": list(range(100)),
            "feature_2": list(range(100, 200)),
        },
        index=pd.RangeIndex(100),
    )
    binary_y = pd.Series([0, 1] * 50, index=X.index)
    attack_categories = pd.Series(["BENIGN" if value == 0 else "DoS" for value in binary_y], index=X.index)

    splitter = DataSplitter()
    split = splitter.split(X, binary_y, attack_categories)

    validator = DataValidator()
    validation = validator.validate_full_pipeline(
        X=X,
        binary_y=binary_y,
        attack_categories=attack_categories,
        X_train=split.X_train,
        X_val=split.X_val,
        X_test=split.X_test,
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
        attack_categories_train=split.attack_categories_train,
        attack_categories_val=split.attack_categories_val,
        attack_categories_test=split.attack_categories_test,
        preprocessing_metadata={
            "duplicates_removed": 0,
            "infinity_values_replaced": 0,
            "rows_removed_due_to_missing_target": 0,
            "original_shape": (100, 2),
            "final_shape": (100, 2),
        },
    )

    assert validation["is_valid"] is True
    assert validation["class_distribution"]["full"]["NORMAL"] > 0
    assert validation["class_distribution"]["full"]["ATTACK"] > 0
    assert validation["data_leakage"]["train_vs_validation"] == "PASSED"
    assert validation["data_leakage"]["train_vs_test"] == "PASSED"
    assert validation["data_leakage"]["validation_vs_test"] == "PASSED"
