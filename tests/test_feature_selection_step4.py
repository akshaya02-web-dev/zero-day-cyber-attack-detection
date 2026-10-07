"""Tests for the train-only Step 4 feature selection stage."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.feature_selection_pipeline import select_features_from_datasets


def test_fits_one_selector_and_reuses_it_for_each_dataset(tmp_path: Path):
    """Fit on the temporary training pool and preserve dataset-specific labels."""
    extracted_directory = tmp_path / "extracted"
    output_directory = tmp_path / "selected"
    artifact_directory = tmp_path / "artifacts"
    extracted_directory.mkdir()

    base = {
        "feature_a": [0, 1, 0, 1],
        "feature_b": [1, 0, 1, 0],
        "feature_c": [0, 0, 1, 1],
        "feature_d": [1, 1, 0, 0],
    }
    pd.DataFrame({**base, "Label": ["BENIGN", "DoS", "BENIGN", "DoS"]}).to_csv(
        extracted_directory / "first.csv", index=False
    )
    pd.DataFrame({**base, "Label": ["Bot", "BENIGN", "Bot", "BENIGN"]}).to_csv(
        extracted_directory / "second.csv", index=False
    )

    result = select_features_from_datasets(
        extracted_directory,
        output_directory,
        artifact_directory,
        top_n_features=2,
    )

    assert result["features_before"] == 4
    assert result["features_after"] == 2
    assert len(result["selected_features"]) == 2
    assert result["selector_path"].is_file()
    assert result["selected_names_path"].is_file()
    assert result["report_path"].is_file()
    fitted_selector = joblib.load(result["selector_path"])
    assert fitted_selector.get_selected_features() == result["selected_features"]

    first = pd.read_csv(output_directory / "first.csv")
    second = pd.read_csv(output_directory / "second.csv")
    assert first.columns[:-1].tolist() == result["selected_features"]
    assert second.columns[:-1].tolist() == result["selected_features"]
    assert first["Label"].tolist() == ["BENIGN", "DoS", "BENIGN", "DoS"]
    assert second["Label"].tolist() == ["Bot", "BENIGN", "Bot", "BENIGN"]
