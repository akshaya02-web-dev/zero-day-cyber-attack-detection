"""Tests for the train-pool-only Step 5 normalization stage."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.normalization_pipeline import normalize_selected_datasets


FEATURES = ["feature_a", "feature_b"]


def test_fits_one_scaler_and_reuses_it_for_each_dataset(tmp_path: Path):
    """Normalize each selected dataset with one scaler fitted on the temporary pool."""
    selected_directory = tmp_path / "selected"
    output_directory = tmp_path / "normalized"
    models_directory = tmp_path / "models"
    report_directory = tmp_path / "reports"
    selected_directory.mkdir()
    names_path = tmp_path / "selected_feature_names.txt"
    names_path.write_text("\n".join(FEATURES) + "\n", encoding="utf-8")

    first = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [10.0, 20.0], "Label": ["BENIGN", "DoS"]})
    second = pd.DataFrame({"feature_a": [3.0, 4.0], "feature_b": [30.0, 40.0], "Label": ["Bot", "BENIGN"]})
    first.to_csv(selected_directory / "first.csv", index=False)
    second.to_csv(selected_directory / "second.csv", index=False)

    result = normalize_selected_datasets(
        selected_directory,
        output_directory,
        models_directory,
        report_directory,
        names_path,
    )

    assert result["features_before"] == 2
    assert result["features_after"] == 2
    assert result["scaler_path"].is_file()
    assert result["report_path"].is_file()
    scaler = joblib.load(result["scaler_path"])
    assert scaler.n_features_in_ == 2
    assert scaler.mean_.tolist() == [2.5, 25.0]

    first_normalized = pd.read_csv(output_directory / "first.csv")
    second_normalized = pd.read_csv(output_directory / "second.csv")
    assert first_normalized.columns.tolist() == FEATURES + ["Label"]
    assert second_normalized.columns.tolist() == FEATURES + ["Label"]
    assert first_normalized["Label"].tolist() == ["BENIGN", "DoS"]
    assert second_normalized["Label"].tolist() == ["Bot", "BENIGN"]
    assert "Training data used for fitting:" in result["report_path"].read_text(encoding="utf-8")
