"""Tests for the preprocessing stage."""

from __future__ import annotations

import pandas as pd

from src.config import DataConfig, ProjectConfig
from src.preprocessing import Preprocessor


def test_preprocessor_builds_binary_targets_and_metadata():
    """Ensure preprocessing cleans data and preserves attack labels safely."""
    config = ProjectConfig(
        data=DataConfig(
            dataset_path="dummy.csv",
            label_column="label",
            attack_column="attack_category",
            normal_label_names=("NORMAL", "BENIGN", "normal"),
            known_attack_categories=("DoS", "DDoS", "PortScan", "Bot", "Web Attack"),
            held_out_unknown_attack_categories=("Web Attack",),
        )
    )

    df = pd.DataFrame(
        {
            "label": ["NORMAL", " DoS ", "Bot", "NORMAL", " DoS "],
            "attack_category": ["BENIGN", "DoS", "Bot", "BENIGN", "DoS"],
            "src_port": [10, 20, 30, 10, 20],
            "dst_port": [5, 7, 9, 5, 7],
            "bytes": [100.0, float("inf"), 80.0, 100.0, float("-inf")],
            "notes": [" x ", " y ", " z ", " x ", " y "],
        }
    )
    df = pd.concat([df, df.iloc[[0]].copy()], ignore_index=True)

    prepared = Preprocessor(config).prepare(df)

    assert "X" in prepared
    assert "binary_y" in prepared
    assert "attack_categories" in prepared
    assert "metadata" in prepared
    assert prepared["binary_y"].nunique() == 2
    assert prepared["binary_y"].isin([0, 1]).all()
    assert set(prepared["attack_categories"].dropna().unique()) >= {"BENIGN", "DoS", "Bot"}
    assert prepared["metadata"]["original_shape"][0] == len(df)
    assert prepared["metadata"]["duplicates_removed"] >= 1
    assert prepared["metadata"]["missing_value_count"] >= 0
    assert "final_shape" in prepared["metadata"]
    assert "normal_samples" in prepared["metadata"]
    assert "attack_samples" in prepared["metadata"]


def test_preprocessor_accepts_benign_and_ddos_labels():
    """Ensure real-world CIC-style labels are mapped to binary targets."""
    config = ProjectConfig(
        data=DataConfig(
            dataset_path="dummy.csv",
            label_column="Label",
            attack_column="attack_category",
            normal_label_names=("NORMAL", "BENIGN", "normal", "benign"),
            known_attack_categories=("DoS", "DDoS", "PortScan", "Bot", "Web Attack"),
            held_out_unknown_attack_categories=("Web Attack",),
        )
    )

    df = pd.DataFrame(
        {
            "Label": ["BENIGN", "DDoS", "BENIGN", "DDoS"],
            "src_port": [1, 2, 3, 4],
            "dst_port": [5, 6, 7, 8],
            "bytes": [10.0, 20.5, 30.0, 40.5],
        }
    )

    prepared = Preprocessor(config).prepare(df)

    assert prepared["binary_y"].tolist() == [0, 1, 0, 1]
    assert (prepared["binary_y"] == 0).sum() == 2
    assert (prepared["binary_y"] == 1).sum() == 2


def test_run_pipeline_handles_single_class_dataset_gracefully(monkeypatch, capsys):
    """The pipeline should stop cleanly when a user selects a single-class sample dataset."""
    import run_pipeline

    single_class_df = pd.DataFrame({
        "Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN"],
        "src_port": [1, 2, 3, 4],
        "dst_port": [5, 6, 7, 8],
        "bytes": [10.0, 20.0, 30.0, 40.0],
    })

    class DummyLoader:
        def __init__(self, file_path):
            self.file_path = file_path
            self.resolved_file_path = "sample_network_traffic_100.csv"

        def choose_csv_file(self, directory):
            return str(directory)

        def load_from_csv(self, chosen_csv):
            return single_class_df

    monkeypatch.setattr(run_pipeline, "DataLoader", DummyLoader)

    run_pipeline.main()

    captured = capsys.readouterr()
    assert "single class" in captured.out.lower()
    assert "choose a dataset with both normal and attack samples" in captured.out.lower()
