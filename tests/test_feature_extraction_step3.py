"""Tests for the independent Step 3 feature extraction stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_extraction import extract_all_datasets


BASE_COLUMNS = {
    "Total Fwd Packets": [2, 4],
    "Total Backward Packets": [1, 2],
    "Total Length of Fwd Packets": [20, 40],
    "Total Length of Bwd Packets": [10, 20],
    "Flow Duration": [10, 20],
}


def test_extracts_each_cleaned_dataset_independently(tmp_path: Path):
    """Use one deterministic extractor while keeping labels outside its input."""
    cleaned_directory = tmp_path / "cleaned"
    output_directory = tmp_path / "extracted"
    report_directory = tmp_path / "reports"
    cleaned_directory.mkdir()

    first = pd.DataFrame({**BASE_COLUMNS, "Label": ["BENIGN", "DoS"]})
    second = pd.DataFrame({**BASE_COLUMNS, "Label": ["Bot", "BENIGN"]})
    first.to_csv(cleaned_directory / "first.csv", index=False)
    second.to_csv(cleaned_directory / "second.csv", index=False)

    summaries = extract_all_datasets(cleaned_directory, output_directory, report_directory)

    assert len(summaries) == 2
    assert summaries[0]["original_feature_count"] == 5
    assert summaries[0]["extracted_feature_count"] == 12
    assert summaries[0]["generated_features"] == [
        "Total Packets",
        "Total Bytes",
        "Fwd/Bwd Packet Ratio",
        "Fwd/Bwd Byte Ratio",
        "Derived Packet Rate",
        "Derived Byte Rate",
        "Average Packet Size",
    ]
    assert summaries[0]["generated_features"] == summaries[1]["generated_features"]

    first_output = pd.read_csv(output_directory / "first.csv")
    second_output = pd.read_csv(output_directory / "second.csv")
    assert first_output["Label"].tolist() == ["BENIGN", "DoS"]
    assert second_output["Label"].tolist() == ["Bot", "BENIGN"]
    assert first_output.columns[-1] == "Label"
    assert (report_directory / "first_feature_extraction_report.txt").is_file()
    assert (report_directory / "second_feature_extraction_report.txt").is_file()
