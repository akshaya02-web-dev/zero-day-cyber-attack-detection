"""Tests for the independent Step 6 dataset splitting stage."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.dataset_splitting_pipeline import split_normalized_datasets


def test_splits_normalized_datasets_independently_and_stratifies(tmp_path: Path):
    """Split two normalized datasets without fitting or changing the scaler."""
    normalized_directory = tmp_path / "normalized"
    splits_directory = tmp_path / "splits"
    normalized_directory.mkdir()
    scaler_path = tmp_path / "standard_scaler.joblib"
    report_path = tmp_path / "reports" / "splitting_report.txt"

    first = pd.DataFrame({"feature_a": range(20), "feature_b": range(20, 40), "Label": ["BENIGN", "DoS"] * 10})
    second = pd.DataFrame({"feature_a": range(100, 120), "feature_b": range(120, 140), "Label": ["BENIGN", "Bot"] * 10})
    first.to_csv(normalized_directory / "first.csv", index=False)
    second.to_csv(normalized_directory / "second.csv", index=False)
    scaler = StandardScaler().fit(pd.concat([first[["feature_a", "feature_b"]], second[["feature_a", "feature_b"]]]))
    joblib.dump(scaler, scaler_path)
    original_mean = scaler.mean_.copy()

    result = split_normalized_datasets(normalized_directory, splits_directory, scaler_path, report_path)

    assert result["total_counts"] == {"train": 28, "validation": 6, "test": 6}
    assert result["features"] == ["feature_a", "feature_b"]
    assert result["report_path"].is_file()
    assert joblib.load(scaler_path).mean_.tolist() == original_mean.tolist()

    for dataset_name in ("first.csv", "second.csv"):
        train = pd.read_csv(splits_directory / "train" / dataset_name)
        validation = pd.read_csv(splits_directory / "validation" / dataset_name)
        test = pd.read_csv(splits_directory / "test" / dataset_name)
        assert train.columns.tolist() == ["feature_a", "feature_b", "Label"]
        assert len(train) == 14
        assert len(validation) == 3
        assert len(test) == 3
        assert set(train.Label) == set(validation.Label) == set(test.Label)

    report = report_path.read_text(encoding="utf-8")
    assert "No overlap detected" in report
    assert "class distribution:" in report
