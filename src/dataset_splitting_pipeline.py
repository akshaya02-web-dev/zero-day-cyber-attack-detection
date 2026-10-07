"""Step 6: independent train, validation, and test splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_directory


def _dataset_files(directory: Path) -> list[Path]:
    """Return normalized datasets in a deterministic order."""
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )


def _find_label_column(columns: pd.Index) -> str:
    """Find the target column without changing the input dataframe."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column was found in the normalized dataset.")
    return matches[0]


def _distribution(labels: pd.Series, total: int) -> list[dict[str, Any]]:
    """Return counts and percentages for every class in stable order."""
    counts = labels.value_counts(dropna=False)
    return [
        {
            "label": "<MISSING>" if pd.isna(label) else str(label),
            "count": int(count),
            "percentage": round(int(count) / total * 100, 2) if total else 0.0,
        }
        for label, count in counts.items()
    ]


def _split_dataset(dataframe: pd.DataFrame, label_column: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified 70/15/15 partitions from one dataset."""
    train, remainder = train_test_split(
        dataframe,
        train_size=0.70,
        random_state=random_state,
        stratify=dataframe[label_column],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder[label_column],
    )
    train_indices = set(train.index)
    validation_indices = set(validation.index)
    test_indices = set(test.index)
    if train_indices & validation_indices or train_indices & test_indices or validation_indices & test_indices:
        raise ValueError("Data overlap detected between train, validation, and test partitions.")
    return train, validation, test


def split_normalized_datasets(
    normalized_directory: Path,
    splits_directory: Path,
    scaler_path: Path,
    report_path: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split every normalized dataset independently without fitting transformations."""
    csv_files = _dataset_files(normalized_directory)
    if not csv_files:
        raise FileNotFoundError(f"No normalized CSV datasets found under '{normalized_directory}'.")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"The fitted scaler was not found: '{scaler_path}'.")

    scaler = joblib.load(scaler_path)
    summaries: list[dict[str, Any]] = []
    expected_features: list[str] | None = None
    output_paths: list[Path] = []
    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        label_column = _find_label_column(dataframe.columns)
        feature_columns = [column for column in dataframe.columns if column != label_column]
        if expected_features is None:
            expected_features = feature_columns
        elif feature_columns != expected_features:
            raise ValueError(f"Feature order mismatch in '{csv_path.name}'.")
        if getattr(scaler, "n_features_in_", len(feature_columns)) != len(feature_columns):
            raise ValueError(f"Scaler feature count does not match '{csv_path.name}'.")

        train, validation, test = _split_dataset(dataframe, label_column, random_state)
        partitions = {"train": train, "validation": validation, "test": test}
        partition_summary: dict[str, Any] = {}
        for partition_name, partition in partitions.items():
            partition_directory = ensure_directory(splits_directory / partition_name)
            output_path = partition_directory / csv_path.name
            partition.to_csv(output_path, index=False)
            output_paths.append(output_path)
            partition_summary[partition_name] = {
                "samples": len(partition),
                "features": len(feature_columns),
                "class_distribution": _distribution(partition[label_column], len(partition)),
                "output_path": output_path,
            }
        summaries.append({"dataset": csv_path.name, "partitions": partition_summary})

    report_lines = [
        "=" * 62,
        "DATASET SPLITTING",
        "=" * 62,
        "Input: normalized datasets from Step 5",
        f"Scaler reused for validation: {scaler_path}",
        "No scaler or other transformation was fitted during splitting.",
        "Split ratios: train=70%, validation=15%, test=15%",
        f"Random state: {random_state}",
        "",
    ]
    total_counts = {"train": 0, "validation": 0, "test": 0}
    for summary in summaries:
        report_lines.append(f"Dataset: {summary['dataset']}")
        for partition_name in ("train", "validation", "test"):
            partition = summary["partitions"][partition_name]
            total_counts[partition_name] += partition["samples"]
            report_lines.extend([
                f"  {partition_name.title()} samples: {partition['samples']}",
                f"  {partition_name.title()} features: {partition['features']}",
                f"  {partition_name.title()} output: {partition['output_path']}",
                f"  {partition_name.title()} class distribution:",
                *[
                    f"    {entry['label']}: {entry['count']} ({entry['percentage']}%)"
                    for entry in partition["class_distribution"]
                ],
            ])
        report_lines.append("")
    report_lines.extend([
        "TOTAL SAMPLES",
        f"Train: {total_counts['train']}",
        f"Validation: {total_counts['validation']}",
        f"Test: {total_counts['test']}",
        "No overlap detected between partitions within any dataset.",
        "=" * 62,
    ])
    ensure_directory(report_path.parent)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "datasets": summaries,
        "features": expected_features or [],
        "total_counts": total_counts,
        "report_path": report_path,
        "output_paths": output_paths,
    }