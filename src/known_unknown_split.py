"""Step 4D: known-attack train/validation/test data splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.attack_configuration import configure_attacks
from src.utils import ensure_directory


def _label_column(columns: pd.Index) -> str:
    """Find the preserved Label column despite CIC header whitespace."""
    for column in columns:
        if str(column).strip().lower() == "label":
            return column
    raise ValueError("No Label column found in the master dataset.")


def _split_known_data(known_data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 70/15/15 splits from known data only, stratified by attack class."""
    stratify = known_data[label_column] if known_data[label_column].value_counts().min() >= 2 else None
    training, temporary = train_test_split(
        known_data,
        train_size=0.70,
        random_state=42,
        stratify=stratify,
    )
    temporary_stratify = temporary[label_column] if temporary[label_column].value_counts().min() >= 2 else None
    validation, known_test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=42,
        stratify=temporary_stratify,
    )
    return training.reset_index(drop=True), validation.reset_index(drop=True), known_test.reset_index(drop=True)


def create_known_unknown_splits(master_path: Path, output_directory: Path) -> dict[str, Any]:
    """Build known-only train/validation data and a final test with unknown attacks."""
    dataframe = pd.read_csv(master_path)
    label_column = _label_column(dataframe.columns)
    known_attacks, unknown_attacks, _ = configure_attacks(master_path)

    labels = dataframe[label_column].astype(str).str.strip()
    benign_mask = labels.str.lower().isin({"benign", "normal"})
    known_mask = labels.isin(known_attacks)
    unknown_mask = labels.isin(unknown_attacks)
    categorized_mask = benign_mask | known_mask | unknown_mask
    uncategorized = sorted(labels[~categorized_mask].unique().tolist())
    if uncategorized:
        raise ValueError(f"Labels are not configured as known or unknown attacks: {uncategorized}")

    known_data = dataframe[benign_mask | known_mask].copy()
    unknown_data = dataframe[unknown_mask].copy()
    training, validation, known_test = _split_known_data(known_data, label_column)
    final_test = pd.concat([known_test, unknown_data], ignore_index=True)

    output_directory = ensure_directory(output_directory)
    exports = {
        "training_known_attacks.csv": training,
        "validation_known_attacks.csv": validation,
        "known_test.csv": known_test,
        "unknown_test.csv": unknown_data.reset_index(drop=True),
        "final_test.csv": final_test,
    }
    for filename, split in exports.items():
        split.to_csv(output_directory / filename, index=False)

    return {
        "known_attacks": known_attacks,
        "unknown_attacks": unknown_attacks,
        "training": training,
        "validation": validation,
        "known_test": known_test,
        "unknown_test": unknown_data,
        "final_test": final_test,
        "output_directory": output_directory,
    }


def format_split_summary(result: dict[str, Any], label_column: str = "Label") -> str:
    """Format the required Step 4D summary and validation checks."""
    training = result["training"]
    validation = result["validation"]
    known_test = result["known_test"]
    unknown_test = result["unknown_test"]
    final_test = result["final_test"]

    def count_unknown(split: pd.DataFrame) -> int:
        labels = split[label_column] if label_column in split.columns else split[next(c for c in split.columns if str(c).strip().lower() == "label")]
        return int(labels.astype(str).str.strip().isin(result["unknown_attacks"]).sum())

    lines = [
        "=" * 62,
        "KNOWN / UNKNOWN DATA SPLIT",
        "=" * 62,
        "Known attacks:",
        *[f"- {label}" for label in result["known_attacks"]],
        "Unknown attacks:",
        *[f"- {label}" for label in result["unknown_attacks"]],
        "",
        f"Training samples: {len(training)}",
        f"Validation samples: {len(validation)}",
        f"Known Test samples: {len(known_test)}",
        f"Unknown Test samples: {len(unknown_test)}",
        f"Final Test samples: {len(final_test)}",
        "",
        "Unknown Attack Checks:",
        f"Unknown samples in TRAIN: {count_unknown(training)}",
        f"Unknown samples in VALIDATION: {count_unknown(validation)}",
        f"Unknown samples in TEST: {count_unknown(final_test)}",
        "",
        "STEP 4D STATUS: Known attack training/validation/test created successfully. STOP HERE.",
        "Do not perform clustering.",
    ]
    return "\n".join(lines)