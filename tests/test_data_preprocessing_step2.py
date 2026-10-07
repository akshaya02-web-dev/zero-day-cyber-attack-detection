"""Tests for the independent Step 2 preprocessing stage."""

from __future__ import annotations

from pathlib import Path

from src.data_preprocessing import preprocess_all_datasets


def test_preprocesses_each_raw_dataset_independently(tmp_path: Path):
    """Cleaned datasets and reports retain identity, labels, and raw-file safety."""
    raw_directory = tmp_path / "raw"
    output_directory = tmp_path / "cleaned"
    report_directory = tmp_path / "reports"
    raw_directory.mkdir()

    first = raw_directory / "first.csv"
    second = raw_directory / "second.csv"
    first.write_text(
        "feature_a,feature_b, Label \n1,,BENIGN\ninf,2,DoS\n1,,BENIGN\n",
        encoding="utf-8",
    )
    second.write_text("feature_c,Label\n3,PortScan\n4,BENIGN\n", encoding="utf-8")
    original_first = first.read_bytes()
    original_second = second.read_bytes()

    summaries = preprocess_all_datasets(raw_directory, output_directory, report_directory)

    assert [summary["dataset"] for summary in summaries] == ["first.csv", "second.csv"]
    assert (output_directory / "first.csv").is_file()
    assert (output_directory / "second.csv").is_file()
    assert (report_directory / "first_preprocessing_report.txt").is_file()
    assert (report_directory / "second_preprocessing_report.txt").is_file()

    first_output = (output_directory / "first.csv").read_text(encoding="utf-8")
    first_report = (report_directory / "first_preprocessing_report.txt").read_text(encoding="utf-8")
    assert first_output.splitlines()[0] == "feature_a,feature_b,Label"
    assert "BENIGN" in first_output and "DoS" in first_output
    assert "Original Rows: 3" in first_report
    assert "Duplicate Rows Removed: 1" in first_report
    assert "Infinite Values Handled: 1" in first_report
    assert "Label Distribution:" in first_report
    assert first.read_bytes() == original_first
    assert second.read_bytes() == original_second
