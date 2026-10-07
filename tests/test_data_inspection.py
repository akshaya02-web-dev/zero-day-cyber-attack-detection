"""Tests for the Step 1 data inspection stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_inspection import inspect_all_datasets


def test_inspects_each_csv_without_merging_or_modifying_raw_files(tmp_path: Path):
    """Each raw CSV gets its own complete report and label summary entry."""
    raw_directory = tmp_path / "raw"
    report_directory = tmp_path / "reports"
    raw_directory.mkdir()

    first = raw_directory / "first.csv"
    second = raw_directory / "second.csv"
    first.write_text("feature_a,feature_b,Label\n1,,BENIGN\ninf,2,DoS\n1,,BENIGN\n", encoding="utf-8")
    second.write_text("feature_c,Label\n3,PortScan\n4,BENIGN\n", encoding="utf-8")
    original_first = first.read_bytes()
    original_second = second.read_bytes()

    reports = inspect_all_datasets(raw_directory, report_directory)

    assert len(reports) == 2
    assert (report_directory / "first_inspection_report.txt").is_file()
    assert (report_directory / "second_inspection_report.txt").is_file()
    assert (report_directory / "attack_labels_summary.txt").is_file()
    first_report = (report_directory / "first_inspection_report.txt").read_text(encoding="utf-8")
    assert "Rows:\n3" in first_report
    assert "Feature Names:\n- feature_a\n- feature_b" in first_report
    assert f"{'feature_b':<35}2" in first_report
    assert f"{'feature_a':<35}1" in first_report
    assert "- DoS: 1" in (report_directory / "attack_labels_summary.txt").read_text(encoding="utf-8")
    assert first.read_bytes() == original_first
    assert second.read_bytes() == original_second
    assert pd.read_csv(first).shape == (3, 3)