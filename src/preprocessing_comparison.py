"""Step 3: compare raw and cleaned CIC-IDS2017 datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_directory


SUMMARY_COLUMNS = [
    "Dataset", "Rows Before", "Rows After", "Columns Before", "Columns After",
    "ML Features Before", "ML Features After", "Duplicates Before", "Duplicates After",
    "Missing Before", "Missing After", "Infinite Before", "Infinite After", "Rows Removed",
]

ATTACK_COLUMNS = [
    "Dataset", "Class", "Before Count", "After Count", "Difference",
    "Before Percentage", "After Percentage",
]


def _label_column(dataframe: pd.DataFrame) -> str:
    return next(
        column for column in dataframe.columns
        if str(column).strip().lower() == "label"
    )


def _metrics(dataframe: pd.DataFrame) -> dict[str, Any]:
    label_column = _label_column(dataframe)
    feature_columns = [column for column in dataframe.columns if column != label_column]
    numeric_features = dataframe[feature_columns].select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(numeric_features.to_numpy()).sum()) if not numeric_features.empty else 0
    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "ml_features": len(feature_columns),
        "duplicates": int(dataframe.duplicated().sum()),
        "missing": int(dataframe.isna().sum().sum()),
        "infinite": infinite_count,
        "label": label_column,
    }


def _class_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    label_column = _label_column(dataframe)
    counts = dataframe[label_column].value_counts(dropna=False)
    return {"<MISSING>" if pd.isna(label) else str(label).strip(): int(count) for label, count in counts.items()}


def _affected_features(dataframe: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    label_column = _label_column(dataframe)
    features = [column for column in dataframe.columns if column != label_column]
    missing = dataframe[features].isna().sum()
    missing = {str(column): int(count) for column, count in missing[missing > 0].items()}
    numeric = dataframe[features].select_dtypes(include=[np.number])
    infinite = {}
    if not numeric.empty:
        counts = pd.Series(np.isinf(numeric.to_numpy()).sum(axis=0), index=numeric.columns)
        infinite = {str(column): int(count) for column, count in counts[counts > 0].items()}
    return missing, infinite


def _class_rows(dataset: str, before: dict[str, int], after: dict[str, int], before_rows: int, after_rows: int) -> list[dict[str, Any]]:
    rows = []
    for class_name in sorted(set(before) | set(after)):
        before_count = before.get(class_name, 0)
        after_count = after.get(class_name, 0)
        rows.append({
            "Dataset": dataset,
            "Class": class_name,
            "Before Count": before_count,
            "After Count": after_count,
            "Difference": after_count - before_count,
            "Before Percentage": round(before_count / before_rows * 100, 2) if before_rows else 0.0,
            "After Percentage": round(after_count / after_rows * 100, 2) if after_rows else 0.0,
        })
    return rows


def _format_distribution(counts: dict[str, int]) -> list[str]:
    return [f"  {name}: {count}" for name, count in sorted(counts.items())]


def generate_preprocessing_comparison(raw_directory: Path, cleaned_directory: Path, report_directory: Path) -> dict[str, Any]:
    """Generate Step 3 CSV and text reports from raw and cleaned datasets."""
    raw_files = {path.name: path for path in raw_directory.rglob("*.csv")}
    cleaned_files = {path.name: path for path in cleaned_directory.rglob("*.csv")}
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV datasets found under '{raw_directory}'.")
    missing_cleaned = sorted(set(raw_files) - set(cleaned_files))
    if missing_cleaned:
        raise FileNotFoundError(f"Missing cleaned datasets: {', '.join(missing_cleaned)}")

    summary_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    narratives: list[str] = []

    for dataset_name in sorted(raw_files):
        before_frame = pd.read_csv(raw_files[dataset_name])
        after_frame = pd.read_csv(cleaned_files[dataset_name])
        before = _metrics(before_frame)
        after = _metrics(after_frame)
        before_classes = _class_counts(before_frame)
        after_classes = _class_counts(after_frame)
        missing_features, infinite_features = _affected_features(before_frame)

        summary_rows.append({
            "Dataset": dataset_name,
            "Rows Before": before["rows"], "Rows After": after["rows"],
            "Columns Before": before["columns"], "Columns After": after["columns"],
            "ML Features Before": before["ml_features"], "ML Features After": after["ml_features"],
            "Duplicates Before": before["duplicates"], "Duplicates After": after["duplicates"],
            "Missing Before": before["missing"], "Missing After": after["missing"],
            "Infinite Before": before["infinite"], "Infinite After": after["infinite"],
            "Rows Removed": before["rows"] - after["rows"],
        })
        attack_rows.extend(_class_rows(dataset_name, before_classes, after_classes, before["rows"], after["rows"]))

        difference_lines = [
            f"Dataset: {dataset_name}",
            "Actions: duplicate rows removed; missing values handled; infinite values handled; original columns preserved.",
            f"Rows: {before['rows']} -> {after['rows']} (difference: {after['rows'] - before['rows']})",
            f"Total Columns: {before['columns']} -> {after['columns']} (difference: {after['columns'] - before['columns']})",
            f"ML Features: {before['ml_features']} -> {after['ml_features']} (difference: {after['ml_features'] - before['ml_features']})",
            f"Duplicates: {before['duplicates']} -> {after['duplicates']}",
            f"Missing Values: {before['missing']} -> {after['missing']}",
            f"Infinite Values: {before['infinite']} -> {after['infinite']}",
            f"Rows Removed: {before['rows'] - after['rows']}",
            "Final Class Distribution:",
            *_format_distribution(after_classes),
            "Features with missing values prior to preprocessing:",
            *([f"  {name}: {count}" for name, count in sorted(missing_features.items())] or ["  None"]),
            "Features with infinite values prior to preprocessing:",
            *([f"  {name}: {count}" for name, count in sorted(infinite_features.items())] or ["  None"]),
        ]
        narratives.append("\n".join(difference_lines))

    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    attack_frame = pd.DataFrame(attack_rows, columns=ATTACK_COLUMNS)
    ensure_directory(report_directory)
    summary_frame.to_csv(report_directory / "preprocessing_summary.csv", index=False)
    attack_frame.to_csv(report_directory / "attack_distribution_summary.csv", index=False)

    action_lines = [
        "PREPROCESSING ACTIONS",
        "====================",
        *[
            f"{row['Dataset']}: Duplicates removed={row['Duplicates Before'] - row['Duplicates After']}; "
            f"Missing values handled={row['Missing Before'] - row['Missing After']}; "
            f"Infinite values handled={row['Infinite Before'] - row['Infinite After']}; "
            f"Rows removed={row['Rows Removed']}"
            for row in summary_rows
        ],
    ]
    class_lines = ["ATTACK CLASS DISTRIBUTION", "=========================", *[
        f"{row['Dataset']} | {row['Class']}: {row['Before Count']} -> {row['After Count']} (difference: {row['Difference']})"
        + (" WARNING: class disappears." if row["Before Count"] > 0 and row["After Count"] == 0 else "")
        for row in attack_rows
    ]]
    table_lines = [
        "BEFORE -> AFTER TABLE",
        "=====================",
        "Dataset | Metric | BEFORE | AFTER | DIFFERENCE",
        *[
            f"{row['Dataset']} | {metric} | {row[before_key]} | {row[after_key]} | {row[after_key] - row[before_key]}"
            for row in summary_rows
            for metric, before_key, after_key in [
                ("Rows", "Rows Before", "Rows After"),
                ("Total Columns", "Columns Before", "Columns After"),
                ("ML Features", "ML Features Before", "ML Features After"),
                ("Duplicates", "Duplicates Before", "Duplicates After"),
                ("Missing Values", "Missing Before", "Missing After"),
                ("Infinite Values", "Infinite Before", "Infinite After"),
            ]
        ],
    ]
    report_text = "\n\n".join([
        "BEFORE -> AFTER PREPROCESSING",
        "============================",
        *narratives,
        "\n".join(table_lines),
        "\n".join(action_lines),
        "\n".join(class_lines),
        "FINAL STATUS",
        "============",
        "STEP 1 ✓",
        "STEP 2 ✓",
        "STEP 3 ✓",
        f"Reports saved to: {report_directory}",
        "STEP 3 COMPLETED",
    ])
    (report_directory / "preprocessing_report.txt").write_text(report_text, encoding="utf-8")
    return {"summary": summary_frame, "attack_distribution": attack_frame, "report": report_text}