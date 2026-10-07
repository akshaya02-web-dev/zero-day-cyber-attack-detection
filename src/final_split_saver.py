"""Step 4F: save final train, validation, and test datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.attack_configuration import configure_attacks
from src.utils import ensure_directory


def _label_column(dataframe: pd.DataFrame) -> str:
    return next(
        column for column in dataframe.columns
        if str(column).strip().lower() == "label"
    )


def save_final_splits(split_directory: Path, master_path: Path) -> dict[str, Any]:
    """Save the already-created Step 4D splits under their final filenames."""
    source_train = pd.read_csv(split_directory / "training_known_attacks.csv")
    source_validation = pd.read_csv(split_directory / "validation_known_attacks.csv")
    source_test = pd.read_csv(split_directory / "final_test.csv")
    known_attacks, unknown_attacks, _ = configure_attacks(master_path)

    label_column = _label_column(source_train)
    train_labels = source_train[label_column].astype(str).str.strip()
    validation_labels = source_validation[label_column].astype(str).str.strip()
    test_labels = source_test[label_column].astype(str).str.strip()
    unknown_set = set(unknown_attacks)

    unknown_train = int(train_labels.isin(unknown_set).sum())
    unknown_validation = int(validation_labels.isin(unknown_set).sum())
    unknown_test = int(test_labels.isin(unknown_set).sum())
    leakage_passed = unknown_train == 0 and unknown_validation == 0 and unknown_test > 0

    for left, right in [
        (source_train, source_validation),
        (source_train, source_test),
        (source_validation, source_test),
    ]:
        left_hashes = set(pd.util.hash_pandas_object(left, index=False).tolist())
        right_hashes = set(pd.util.hash_pandas_object(right, index=False).tolist())
        leakage_passed = leakage_passed and not left_hashes.intersection(right_hashes)

    output_directory = ensure_directory(split_directory)
    source_train.to_csv(output_directory / "train.csv", index=False)
    source_validation.to_csv(output_directory / "validation.csv", index=False)
    source_test.to_csv(output_directory / "test.csv", index=False)

    summary = pd.DataFrame([{
        "known_attack_classes": "; ".join(known_attacks),
        "unknown_attack_classes": "; ".join(unknown_attacks),
        "training_sample_count": len(source_train),
        "validation_sample_count": len(source_validation),
        "known_test_sample_count": len(source_test) - unknown_test,
        "unknown_test_sample_count": unknown_test,
        "final_test_sample_count": len(source_test),
        "unknown_samples_in_training": unknown_train,
        "unknown_samples_in_validation": unknown_validation,
        "unknown_samples_in_test": unknown_test,
        "leakage_check_status": "PASSED" if leakage_passed else "FAILED",
    }])
    summary.to_csv(output_directory / "known_unknown_split_summary.csv", index=False)

    return {
        "known_attacks": known_attacks,
        "unknown_attacks": unknown_attacks,
        "train": source_train,
        "validation": source_validation,
        "test": source_test,
        "unknown_train": unknown_train,
        "unknown_validation": unknown_validation,
        "unknown_test": unknown_test,
        "leakage_passed": leakage_passed,
    }


def _classes(dataframe: pd.DataFrame) -> str:
    label_column = _label_column(dataframe)
    return " ".join(str(label) for label in dataframe[label_column].astype(str).str.strip().drop_duplicates())


def format_final_split_report(result: dict[str, Any]) -> str:
    """Format the exact requested Step 4F output."""
    return "\n".join([
        "=" * 62,
        "FINAL DATA SPLITS",
        "=" * 62,
        f"TRAIN: {len(result['train'])} samples",
        f"VALIDATION: {len(result['validation'])} samples",
        f"TEST: {len(result['test'])} samples",
        "-" * 62,
        f"TRAIN CLASSES: {_classes(result['train'])}",
        f"VALIDATION CLASSES: {_classes(result['validation'])}",
        f"TEST CLASSES: {_classes(result['test'])}",
        "-" * 62,
        f"Unknown attack in TRAIN: {result['unknown_train']}",
        f"Unknown attack in VALIDATION: {result['unknown_validation']}",
        f"Unknown attack in TEST: {result['unknown_test']}",
        f"Leakage: {'PASSED' if result['leakage_passed'] else 'FAILED'}",
        "=" * 62,
        "STEP 4 COMPLETED",
        "=" * 62,
    ])