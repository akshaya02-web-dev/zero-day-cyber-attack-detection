"""Step 4E: leakage and split validation for Step 4D outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.attack_configuration import configure_attacks


def _label_column(dataframe: pd.DataFrame) -> str | None:
    return next(
        (column for column in dataframe.columns if str(column).strip().lower() == "label"),
        None,
    )


def _duplicate_intersection(left: pd.DataFrame, right: pd.DataFrame) -> int:
    """Count exact records present in both dataframes using row hashes."""
    left_hashes = set(pd.util.hash_pandas_object(left, index=False).tolist())
    right_hashes = set(pd.util.hash_pandas_object(right, index=False).tolist())
    return len(left_hashes.intersection(right_hashes))


def validate_splits(split_directory: Path, master_path: Path) -> dict[str, Any]:
    """Run all requested Step 4E checks and return report values."""
    train = pd.read_csv(split_directory / "training_known_attacks.csv")
    validation = pd.read_csv(split_directory / "validation_known_attacks.csv")
    test = pd.read_csv(split_directory / "final_test.csv")
    known_attacks, unknown_attacks, _ = configure_attacks(master_path)

    train_label = _label_column(train)
    validation_label = _label_column(validation)
    test_label = _label_column(test)
    labels = {
        "TRAIN": train[train_label].astype(str).str.strip() if train_label else pd.Series(dtype=str),
        "VALIDATION": validation[validation_label].astype(str).str.strip() if validation_label else pd.Series(dtype=str),
        "TEST": test[test_label].astype(str).str.strip() if test_label else pd.Series(dtype=str),
    }

    unknown_set = set(unknown_attacks)
    unknown_counts = {name: int(values.isin(unknown_set).sum()) for name, values in labels.items()}
    duplicate_counts = {
        "TRAIN ↔ VALIDATION": _duplicate_intersection(train, validation),
        "TRAIN ↔ TEST": _duplicate_intersection(train, test),
        "VALIDATION ↔ TEST": _duplicate_intersection(validation, test),
    }

    columns_match = list(train.columns) == list(validation.columns) == list(test.columns)
    label_pass = train_label is not None and validation_label is not None and test_label is not None
    unknown_withheld = (
        unknown_counts["TRAIN"] == 0
        and unknown_counts["VALIDATION"] == 0
        and unknown_counts["TEST"] > 0
        and all(set(unknown_attacks).issubset(set(values)) for name, values in labels.items() if name == "TEST")
    )

    known_presence: dict[str, dict[str, bool]] = {}
    for attack in known_attacks:
        known_presence[attack] = {
            "TRAIN": attack in set(labels["TRAIN"]),
            "VALIDATION": attack in set(labels["VALIDATION"]),
            "TEST": attack in set(labels["TEST"]),
        }
    known_presence_pass = all(all(checks.values()) for checks in known_presence.values())

    failures: list[str] = []
    if not unknown_withheld:
        failures.append("Unknown attack withholding or TEST coverage failed.")
    if any(duplicate_counts.values()):
        failures.append(f"Duplicate records found across splits: {duplicate_counts}.")
    if not columns_match:
        failures.append("Column structures are not identical across TRAIN, VALIDATION, and TEST.")
    if not label_pass:
        failures.append("Label column is missing from one or more splits.")
    if not known_presence_pass:
        failures.append(f"Known attack coverage failed: {known_presence}.")

    return {
        "unknown_counts": unknown_counts,
        "duplicate_counts": duplicate_counts,
        "columns_match": columns_match,
        "label_pass": label_pass,
        "unknown_withheld": unknown_withheld,
        "known_presence_pass": known_presence_pass,
        "failures": failures,
    }


def format_validation_report(result: dict[str, Any]) -> str:
    """Format the required Step 4E report."""
    status = "PASS" if not result["failures"] else "FAIL"
    lines = [
        "=" * 62,
        "LEAKAGE CHECK",
        "=" * 62,
        f"Unknown attacks in TRAIN: {result['unknown_counts']['TRAIN']}",
        f"Unknown attacks in VALIDATION: {result['unknown_counts']['VALIDATION']}",
        f"Unknown attacks in TEST: {result['unknown_counts']['TEST']}",
        f"TRAIN ↔ VALIDATION duplicates: {result['duplicate_counts']['TRAIN ↔ VALIDATION']}",
        f"TRAIN ↔ TEST duplicates: {result['duplicate_counts']['TRAIN ↔ TEST']}",
        f"VALIDATION ↔ TEST duplicates: {result['duplicate_counts']['VALIDATION ↔ TEST']}",
        f"Column structure: {'PASS' if result['columns_match'] else 'FAIL'}",
        f"Label column: {'PASS' if result['label_pass'] else 'FAIL'}",
        f"Unknown attack withheld correctly: {'PASS' if result['unknown_withheld'] else 'FAIL'}",
        "=" * 62,
        "",
        f"FINAL STATUS: {status}",
    ]
    if result["failures"]:
        lines.extend(["", "Failure summary:", *[f"- {failure}" for failure in result["failures"]]])
    return "\n".join(lines)