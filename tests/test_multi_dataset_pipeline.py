"""Tests for the dataset-isolated end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.multi_dataset_pipeline import run_multi_dataset_pipeline


def test_runs_all_stages_without_cross_dataset_pooling(tmp_path: Path):
    """Keep one dataset's preprocessing and model artifacts under its own identity."""
    raw_directory = tmp_path / "raw"
    output_directory = tmp_path / "output"
    raw_directory.mkdir()
    rows = []
    for index in range(40):
        row = {
            "Total Fwd Packets": index + 1,
            "Total Backward Packets": index + 2,
            "Total Length of Fwd Packets": index + 10,
            "Total Length of Bwd Packets": index + 20,
            "Flow Duration": index + 100,
            "Label": "BENIGN" if index % 2 == 0 else "DoS",
        }
        row.update({f"base_{feature_index}": index + feature_index for feature_index in range(20)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(raw_directory / "sample.csv", index=False)

    result = run_multi_dataset_pipeline(raw_directory, output_directory)

    record = result["datasets"][0]
    assert record["name"] == "sample"
    assert (output_directory / "processed" / "sample" / "cleaned.csv").is_file()
    assert (output_directory / "processed" / "sample" / "extracted_features.csv").is_file()
    assert (output_directory / "processed" / "sample" / "selected_features.csv").is_file()
    assert (output_directory / "processed" / "sample" / "normalized.csv").is_file()
    assert (output_directory / "processed" / "sample" / "feature_selector.pkl").is_file()
    assert (output_directory / "processed" / "sample" / "scaler.pkl").is_file()
    assert (output_directory / "splits" / "sample" / "train" / "sample.csv").is_file()
    assert (output_directory / "models" / "sample" / "random_forest.pkl").is_file()
    assert (output_directory / "models" / "sample" / "xgboost.pkl").is_file()
    assert (output_directory / "models" / "sample" / "lightgbm.pkl").is_file()
    assert (output_directory / "results" / "sample" / "validation_results.csv").is_file()
    assert (output_directory / "results" / "sample" / "test_results.csv").is_file()
    assert result["limitations"] == []
