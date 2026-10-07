"""Read-only, per-dataset inspection of CSV datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_directory


def _find_label_column(columns: pd.Index) -> str:
    """Find the label column without changing the source dataframe."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column was found in the dataset.")
    return matches[0]


def _format_counts(title: str, counts: pd.Series, value_label: str) -> list[str]:
    """Format a count for every feature, including zero-count features."""
    lines = [f"{title}:", f"Total: {int(counts.sum())}", ""]
    lines.extend([f"{'Feature':<35}{value_label}", "-" * 50])
    lines.extend(f"{str(name):<35}{int(count)}" for name, count in counts.items())
    return lines


def inspect_dataset(csv_path: Path) -> str:
    """Inspect one CSV and return its beginner-friendly report text."""
    dataframe = pd.read_csv(csv_path)
    label_column = _find_label_column(dataframe.columns)
    feature_columns = [column for column in dataframe.columns if column != label_column]

    missing_counts = dataframe[feature_columns].isna().sum()

    numeric_features = dataframe[feature_columns].select_dtypes(include=[np.number])
    if numeric_features.empty:
        infinite_counts = pd.Series(dtype="int64")
    else:
        infinite_counts = pd.Series(
            np.isinf(numeric_features.to_numpy()).sum(axis=0),
            index=numeric_features.columns,
        )
    infinite_counts = infinite_counts.reindex(feature_columns, fill_value=0)

    class_counts = dataframe[label_column].value_counts(dropna=False)
    class_counts.index = class_counts.index.map(lambda value: "<MISSING>" if pd.isna(value) else str(value))

    lines = [
        "=" * 62,
        "DATA INSPECTION",
        "=" * 62,
        "",
        f"Dataset:\n{csv_path.name}",
        "",
        f"Rows:\n{len(dataframe)}",
        "",
        f"Total Columns:\n{len(dataframe.columns)}",
        "",
        f"Target Column:\n{str(label_column).strip()}",
        "",
        f"ML Features:\n{len(feature_columns)}",
        "",
        "Feature Names:",
        *[f"- {feature}" for feature in feature_columns],
        "",
        f"Duplicate Rows:\n{int(dataframe.duplicated().sum())}",
        "",
    ]

    lines.extend(_format_counts("Missing Values", missing_counts, "Missing Count"))
    lines.append("")

    lines.extend(_format_counts("Infinite Values", infinite_counts, "Infinite Count"))
    lines.extend(["", "Class Distribution:", f"{'Class':<35}Count", "-" * 50])
    lines.extend(f"{str(label):<35}{int(count)}" for label, count in class_counts.items())
    lines.extend([
        "",
        f"Unique Attack/Normal Labels:\n{dataframe[label_column].nunique(dropna=False)}",
        "",
        "=" * 62,
    ])
    return "\n".join(lines)


def inspect_all_datasets(
    raw_directory: Path,
    report_directory: Path,
    summary_path: Path | None = None,
) -> list[str]:
    """Inspect every CSV independently and save one report per dataset."""
    csv_files = sorted(
        (path for path in raw_directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV datasets found under '{raw_directory}'.")

    reports = []
    label_summary = ["STEP 1 ATTACK LABEL SUMMARY", "", f"Raw data folder: {raw_directory}", ""]
    ensure_directory(report_directory)
    for csv_path in csv_files:
        try:
            report = inspect_dataset(csv_path)
            reports.append(report)
            report_file = report_directory / f"{csv_path.stem}_inspection_report.txt"
            report_file.write_text(report + "\n", encoding="utf-8")

            header = pd.read_csv(csv_path, nrows=0)
            label_column = _find_label_column(header.columns)
            labels = pd.read_csv(csv_path, usecols=[label_column])[label_column]
            label_summary.extend([f"Dataset: {csv_path.name}", "Labels:"])
            label_summary.extend(
                f"- {('<MISSING>' if pd.isna(label) else str(label))}: {int(count)}"
                for label, count in labels.value_counts(dropna=False).items()
            )
            label_summary.append("")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            report = "\n".join([
                "=" * 62,
                "DATA INSPECTION",
                "=" * 62,
                "",
                f"Dataset:\n{csv_path.name}",
                "",
                f"Inspection failed: {exc}",
                "=" * 62,
            ])
            reports.append(report)
            (report_directory / f"{csv_path.stem}_inspection_report.txt").write_text(report + "\n", encoding="utf-8")
            label_summary.extend([f"Dataset: {csv_path.name}", f"Inspection failed: {exc}", ""])

    if summary_path is None:
        summary_path = report_directory / "attack_labels_summary.txt"
    ensure_directory(summary_path.parent)
    summary_path.write_text("\n".join(label_summary).rstrip() + "\n", encoding="utf-8")
    return reports