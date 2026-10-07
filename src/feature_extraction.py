"""Step 3: independent feature extraction for cleaned datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.utils import ensure_directory


def _find_label_column(columns: pd.Index) -> str:
    """Find the target column without changing the input dataframe."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column was found in the cleaned dataset.")
    return matches[0]


def extract_dataset(csv_path: Path, output_directory: Path) -> dict[str, Any]:
    """Extract the same flow features from one cleaned dataset."""
    dataframe = pd.read_csv(csv_path)
    label_column = _find_label_column(dataframe.columns)
    labels = dataframe[label_column].copy()
    feature_frame = dataframe.drop(columns=[label_column]).copy()
    original_feature_count = len(feature_frame.columns)

    engineered_features = FeatureEngineer().transform(feature_frame)
    generated_features = [
        column for column in engineered_features.columns
        if column not in feature_frame.columns
    ]
    extracted = engineered_features.copy()
    extracted["Label"] = labels.to_numpy()

    output_path = ensure_directory(output_directory) / csv_path.name
    extracted.to_csv(output_path, index=False)
    return {
        "dataset": csv_path.name,
        "original_feature_count": original_feature_count,
        "extracted_feature_count": len(extracted.columns) - 1,
        "generated_features": generated_features,
        "output_path": output_path,
    }


def format_extraction_report(summary: dict[str, Any]) -> str:
    """Format one dataset's feature extraction result."""
    return "\n".join([
        "=" * 62,
        "FEATURE EXTRACTION",
        "=" * 62,
        f"Dataset: {summary['dataset']}",
        f"Original Feature Count: {summary['original_feature_count']}",
        f"Extracted Feature Count: {summary['extracted_feature_count']}",
        "Newly Generated Features:",
        *[f"  - {feature}" for feature in summary["generated_features"]],
        "=" * 62,
    ])


def extract_all_datasets(
    cleaned_directory: Path,
    output_directory: Path,
    report_directory: Path,
) -> list[dict[str, Any]]:
    """Extract features independently from every cleaned CSV."""
    csv_files = sorted(
        (path for path in cleaned_directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )
    if not csv_files:
        raise FileNotFoundError(f"No cleaned CSV datasets found under '{cleaned_directory}'.")

    ensure_directory(report_directory)
    summaries = []
    for csv_path in csv_files:
        summary = extract_dataset(csv_path, output_directory)
        summaries.append(summary)
        report_path = report_directory / f"{csv_path.stem}_feature_extraction_report.txt"
        report_path.write_text(format_extraction_report(summary) + "\n", encoding="utf-8")
    return summaries