"""Tests for Step 8 validation-only evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model_validation_pipeline import validate_models
from src.multi_model_training import train_all_models


def _write_split(directory: Path, name: str, offset: int) -> None:
    rows = []
    for index in range(20):
        row = {f"feature_{feature_index}": offset + index + feature_index for feature_index in range(20)}
        row["Label"] = "BENIGN" if index % 2 == 0 else "DoS"
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / name, index=False)


def test_validates_saved_models_using_only_validation_data(tmp_path: Path):
    """Evaluate saved models without requiring a test directory or retraining."""
    splits_directory = tmp_path / "splits"
    train_directory = splits_directory / "train"
    validation_directory = splits_directory / "validation"
    train_directory.mkdir(parents=True)
    validation_directory.mkdir(parents=True)
    _write_split(train_directory, "first.csv", 0)
    _write_split(validation_directory, "first.csv", 100)
    models_directory = tmp_path / "models"
    train_all_models(splits_directory, models_directory)

    result = validate_models(validation_directory, models_directory, tmp_path / "results" / "validation")

    assert result["validation_samples"] == 20
    assert len(result["features"]) == 20
    assert set(result["results"]) == {"random_forest", "xgboost", "lightgbm"}
    for model_name in result["results"]:
        assert (tmp_path / "results" / "validation" / f"{model_name}_classification_report.txt").is_file()
        assert (tmp_path / "results" / "validation" / f"{model_name}_confusion_matrix.csv").is_file()
    assert (tmp_path / "results" / "validation" / "model_comparison.csv").is_file()
    assert result["report_path"].is_file()
