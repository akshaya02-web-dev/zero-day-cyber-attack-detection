"""Step 2: CIC-IDS2017 dataset preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import Preprocessor
from src.utils import ensure_directory


def _find_label_column(columns: pd.Index) -> str:
    """Find the Label column without changing the source dataset."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column was found in the dataset.")
    return matches[0]


def preprocess_dataset(csv_path: Path, output_directory: Path) -> dict[str, Any]:
    """Clean one CSV while preserving its complete column structure."""
    original = pd.read_csv(csv_path)
    original_rows, original_columns = original.shape
    original_label_column = _find_label_column(original.columns)
    missing_values_before = int(original.isna().sum().sum())

    numeric_original = original.select_dtypes(include=[np.number])
    infinite_values_handled = int(np.isinf(numeric_original.to_numpy()).sum()) if not numeric_original.empty else 0

    # Reuse the repository's existing column-name and infinity-cleaning logic.
    cleaned = Preprocessor().clean_data(original.copy())
    label_column = _find_label_column(cleaned.columns)
    cleaned = cleaned.rename(columns={label_column: "Label"})
    if original_label_column != label_column:
        cleaned["Label"] = original[original_label_column].copy()

    duplicate_rows_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Apply the existing median/mode strategy without changing label values.
    for column in [column for column in cleaned.columns if column != "Label"]:
        if not cleaned[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            replacement = cleaned[column].median()
        else:
            modes = cleaned[column].mode(dropna=True)
            replacement = modes.iloc[0] if not modes.empty else "UNKNOWN"
        cleaned[column] = cleaned[column].fillna(0 if pd.isna(replacement) else replacement)

    output_path = ensure_directory(output_directory) / csv_path.name
    cleaned.to_csv(output_path, index=False)

    return {
        "dataset": csv_path.name,
        "original_rows": original_rows,
        "original_columns": original_columns,
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_values_handled": missing_values_before,
        "infinite_values_handled": infinite_values_handled,
        "final_rows": len(cleaned),
        "final_columns": len(cleaned.columns),
        "ml_features": len(cleaned.columns) - 1,
        "label_column": "Label",
        "label_distribution": {
            "<MISSING>" if pd.isna(label) else str(label): int(count)
            for label, count in cleaned["Label"].value_counts(dropna=False).items()
        },
        "output_path": output_path,
    }


def format_preprocessing_report(summary: dict[str, Any]) -> str:
    """Format one dataset result using the requested Step 2 layout."""
    return "\n".join([
        "=" * 62,
        "PREPROCESSING",
        "=" * 62,
        f"Dataset: {summary['dataset']}",
        f"Original Rows: {summary['original_rows']}",
        f"Original Columns: {summary['original_columns']}",
        f"Duplicate Rows Removed: {summary['duplicate_rows_removed']}",
        f"Missing Values Handled: {summary['missing_values_handled']}",
        f"Infinite Values Handled: {summary['infinite_values_handled']}",
        f"Final Rows: {summary['final_rows']}",
        f"Final Columns: {summary['final_columns']}",
        f"ML Features: {summary['ml_features']}",
        f"Label Column: {summary['label_column']}",
        "Label Distribution:",
        *[f"  {label}: {count}" for label, count in summary["label_distribution"].items()],
        "=" * 62,
    ])


def preprocess_all_datasets(
    raw_directory: Path,
    output_directory: Path,
    report_directory: Path,
) -> list[dict[str, Any]]:
    """Preprocess every raw CSV independently and save one report per dataset."""
    csv_files = sorted(
        (path for path in raw_directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV datasets found under '{raw_directory}'.")

    ensure_directory(report_directory)
    summaries = []
    for csv_path in csv_files:
        summary = preprocess_dataset(csv_path, output_directory)
        summaries.append(summary)
        report_path = report_directory / f"{csv_path.stem}_preprocessing_report.txt"
        report_path.write_text(format_preprocessing_report(summary) + "\n", encoding="utf-8")
    return summaries