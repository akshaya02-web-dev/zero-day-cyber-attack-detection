"""Tests for Step 7 multi-model supervised training."""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd

from src.multi_model_training import train_all_models


def _write_split(directory: Path, name: str, offset: int) -> None:
    rows = []
    for index in range(12):
        row = {f"feature_{feature_index}": offset + index + feature_index for feature_index in range(20)}
        row["Label"] = "BENIGN" if index % 2 == 0 else "DoS"
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / name, index=False)


def test_trains_all_models_from_train_and_validation_only(tmp_path: Path):
    """Save three models with the same feature contract and learned classes."""
    train_directory = tmp_path / "splits" / "train"
    train_directory.mkdir(parents=True)
    _write_split(train_directory, "first.csv", 0)

    result = train_all_models(tmp_path / "splits", tmp_path / "models")

    assert len(result["feature_columns"]) == 20
    assert len(result["results"]) == 3
    assert all(item["training_samples"] == 12 for item in result["results"])
    assert all(item["features"] == 20 for item in result["results"])
    assert all(item["model_path"].is_file() for item in result["results"])
    classes = json.loads(result["classes_path"].read_text(encoding="utf-8"))
    assert classes["random_forest"] == ["BENIGN", "DoS"]
    assert classes["xgboost"] == ["BENIGN", "DoS"]
    assert classes["lightgbm"] == ["BENIGN", "DoS"]
    assert result["report_path"].is_file()
