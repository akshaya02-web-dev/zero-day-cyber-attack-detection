"""Tests for Step 9 final-test evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.final_test_pipeline import evaluate_final_test
from src.multi_model_training import train_all_models


def _write_split(directory: Path, name: str, offset: int) -> None:
    rows = []
    for index in range(20):
        row = {f"feature_{feature_index}": offset + index + feature_index for feature_index in range(20)}
        row["Label"] = "BENIGN" if index % 2 == 0 else "DoS"
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / name, index=False)


def test_evaluates_saved_models_on_test_data_only(tmp_path: Path):
    """Evaluate saved models without retraining or requiring validation data."""
    splits_directory = tmp_path / "splits"
    train_directory = splits_directory / "train"
    test_directory = splits_directory / "test"
    train_directory.mkdir(parents=True)
    test_directory.mkdir(parents=True)
    _write_split(train_directory, "first.csv", 0)
    _write_split(test_directory, "first.csv", 100)
    models_directory = tmp_path / "models"
    train_all_models(splits_directory, models_directory)

    result = evaluate_final_test(test_directory, models_directory, tmp_path / "results" / "final_test")

    assert result["test_samples"] == 20
    assert len(result["features"]) == 20
    assert set(result["results"]) == {"random_forest", "xgboost", "lightgbm"}
    assert result["report_path"].is_file()
    assert (tmp_path / "results" / "final_test" / "model_comparison.csv").is_file()