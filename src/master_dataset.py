"""Step 4A: combine compatible cleaned CIC-IDS2017 datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_directory


def _label_column(columns: pd.Index) -> str | None:
    """Return the existing target column, allowing harmless header whitespace."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    return matches[0] if matches else None


def _column_signature(columns: pd.Index) -> tuple[str, ...]:
    """Compare columns using their source names, including their order."""
    return tuple(str(column) for column in columns)


def create_master_dataset(input_directory: Path, output_directory: Path) -> dict[str, Any]:
    """Merge compatible cleaned datasets and globally deduplicate them."""
    csv_files = sorted(input_directory.rglob("*.csv"), key=lambda path: path.name.lower())
    if not csv_files:
        raise FileNotFoundError(f"No cleaned CSV datasets found under '{input_directory}'.")

    datasets: list[tuple[Path, pd.DataFrame]] = []
    rejected: list[dict[str, str]] = []
    expected_signature: tuple[str, ...] | None = None
    expected_label: str | None = None

    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        label_column = _label_column(dataframe.columns)
        if label_column is None:
            rejected.append({"dataset": csv_path.name, "reason": "No Label column found."})
            continue

        signature = _column_signature(dataframe.columns)
        if expected_signature is None:
            expected_signature = signature
            expected_label = label_column
        if signature != expected_signature:
            rejected.append({
                "dataset": csv_path.name,
                "reason": "Column structure mismatch.",
                "columns": ", ".join(signature),
            })
            continue

        datasets.append((csv_path, dataframe))

    if not datasets:
        raise ValueError("No structurally compatible datasets with a Label column were found.")

    combined = pd.concat([dataframe for _, dataframe in datasets], ignore_index=True)
    duplicate_rows_before = int(combined.duplicated().sum())
    master = combined.drop_duplicates(ignore_index=True)
    duplicate_rows_removed = len(combined) - len(master)

    output_directory = ensure_directory(output_directory)
    master_path = output_directory / "master_cleaned_dataset.csv"
    summary_path = output_directory / "master_dataset_summary.csv"
    master.to_csv(master_path, index=False)

    assert expected_label is not None
    feature_count = len(master.columns) - 1
    summary_rows = [{
        "datasets_combined": len(datasets),
        "datasets_rejected": len(rejected),
        "total_rows_before_deduplication": len(combined),
        "total_columns": len(master.columns),
        "ml_features": feature_count,
        "target": "Label",
        "unique_classes": int(master[expected_label].nunique(dropna=False)),
        "duplicate_rows_before_global_deduplication": duplicate_rows_before,
        "duplicate_rows_removed": duplicate_rows_removed,
        "final_rows": len(master),
        "rejected_datasets": "; ".join(item["dataset"] for item in rejected) or "None",
    }]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "datasets_combined": len(datasets),
        "datasets_rejected": rejected,
        "total_rows": len(combined),
        "total_columns": len(master.columns),
        "ml_features": feature_count,
        "target": "Label",
        "unique_classes": int(master[expected_label].nunique(dropna=False)),
        "duplicate_rows_before": duplicate_rows_before,
        "duplicate_rows_removed": duplicate_rows_removed,
        "final_rows": len(master),
        "master_path": master_path,
        "summary_path": summary_path,
    }


def format_master_report(summary: dict[str, Any]) -> str:
    """Format the requested Step 4A console output."""
    return "\n".join([
        "=" * 62,
        "MASTER DATASET",
        "=" * 62,
        f"Datasets Combined: {summary['datasets_combined']}",
        f"Total Rows: {summary['total_rows']}",
        f"Total Columns: {summary['total_columns']}",
        f"ML Features: {summary['ml_features']}",
        f"Target: {summary['target']}",
        f"Unique Classes: {summary['unique_classes']}",
        f"Duplicate Rows Before Global Deduplication: {summary['duplicate_rows_before']}",
        f"Duplicate Rows Removed: {summary['duplicate_rows_removed']}",
        f"Final Rows: {summary['final_rows']}",
        "=" * 62,
    ])