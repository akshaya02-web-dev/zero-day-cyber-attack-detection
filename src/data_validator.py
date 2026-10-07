"""Validation and reporting utilities for the cybersecurity data preparation pipeline.

This module verifies dataset integrity, class balance, split safety, and produces
summary reports and charts for the ML-ready dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import configure_logger, ensure_directory


class DataValidator:
    """Validate the cleaned and split dataset before machine learning training."""

    def __init__(self):
        self.logger = configure_logger("cyber_threat_validator", logging.INFO)

    def _validate_original_dataset(
        self,
        X: pd.DataFrame,
        binary_y: pd.Series,
        attack_categories: pd.Series,
        preprocessing_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate the processed dataset before split validation."""
        errors: list[str] = []

        if X is None or binary_y is None or attack_categories is None:
            raise ValueError("X, binary_y, and attack_categories must all be provided.")
        if X.empty:
            raise ValueError("Processed dataset is empty.")
        if len(X) != len(binary_y) or len(X) != len(attack_categories):
            raise ValueError(
                "X, binary_y, and attack_categories must have matching row counts. "
                f"Got {len(X)}, {len(binary_y)}, and {len(attack_categories)}."
            )
        if any(column_name in X.columns for column_name in ["label", "target", "binary_y"]):
            errors.append("The target label must not remain in the feature matrix X.")

        if not binary_y.isin([0, 1]).all():
            errors.append("binary_y must contain only 0 (NORMAL) and 1 (ATTACK).")

        if binary_y.nunique() < 2:
            errors.append("Binary target must contain both NORMAL and ATTACK classes.")

        if X.index.has_duplicates:
            errors.append("Feature matrix index contains duplicates.")
        if binary_y.index.has_duplicates:
            errors.append("binary_y index contains duplicates.")
        if attack_categories.index.has_duplicates:
            errors.append("attack_categories index contains duplicates.")

        numeric_data = X.select_dtypes(include=["number"])
        if not numeric_data.empty and np.isinf(numeric_data.to_numpy(dtype=float, na_value=np.nan)).any():
            errors.append("Infinite values remain in numeric feature columns. These should be converted to NaN before modeling.")

        if X.isna().sum().sum() > 0:
            self.logger.warning("NaN values remain in X; this is acceptable before imputation during the ML pipeline.")

        if preprocessing_metadata is None:
            preprocessing_metadata = {}

        if "infinity_values_replaced" in preprocessing_metadata:
            if preprocessing_metadata["infinity_values_replaced"] is None:
                self.logger.warning("Infinity replacement count is missing from metadata.")

        if errors:
            raise ValueError("; ".join(errors))

        return {
            "processed_dataset_valid": True,
            "row_count": len(X),
            "feature_count": X.shape[1],
            "class_counts": {
                "NORMAL": int((binary_y == 0).sum()),
                "ATTACK": int((binary_y == 1).sum()),
            },
        }

    def _validate_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        attack_categories_train: pd.Series,
        attack_categories_val: pd.Series,
        attack_categories_test: pd.Series,
    ) -> dict[str, Any]:
        """Validate train/validation/test split integrity and no-leakage rules."""
        errors: list[str] = []

        checks = [
            (X_train, y_train, "TRAIN", "X_train", "y_train"),
            (X_val, y_val, "VALIDATION", "X_val", "y_val"),
            (X_test, y_test, "TEST", "X_test", "y_test"),
        ]

        for X_split, y_split, split_name, X_label, y_label in checks:
            if len(X_split) != len(y_split):
                errors.append(f"{split_name}: {X_label} and {y_label} row counts do not match.")
            if len(X_split) != len(attack_categories_train if split_name == "TRAIN" else attack_categories_val if split_name == "VALIDATION" else attack_categories_test):
                errors.append(f"{split_name}: feature matrix and attack categories row counts do not match.")

        train_val_overlap = X_train.index.intersection(X_val.index)
        train_test_overlap = X_train.index.intersection(X_test.index)
        val_test_overlap = X_val.index.intersection(X_test.index)

        if len(train_val_overlap) > 0:
            errors.append("Training and validation overlap detected.")
        if len(train_test_overlap) > 0:
            errors.append("Training and test overlap detected.")
        if len(val_test_overlap) > 0:
            errors.append("Validation and test overlap detected.")

        if errors:
            raise ValueError("; ".join(errors))

        return {
            "train_vs_validation": "PASSED" if len(train_val_overlap) == 0 else "FAILED",
            "train_vs_test": "PASSED" if len(train_test_overlap) == 0 else "FAILED",
            "validation_vs_test": "PASSED" if len(val_test_overlap) == 0 else "FAILED",
        }

    def _ensure_split_balance(self, y_full: pd.Series, y_split: pd.Series, split_name: str) -> dict[str, int | float]:
        """Return class counts and percentages in a split."""
        counts = {
            "NORMAL": int((y_split == 0).sum()),
            "ATTACK": int((y_split == 1).sum()),
        }
        total = float(len(y_split))
        percentages = {
            "NORMAL": round((counts["NORMAL"] / total) * 100, 2) if total else 0.0,
            "ATTACK": round((counts["ATTACK"] / total) * 100, 2) if total else 0.0,
        }
        return {"count": counts, "percentage": percentages, "split_name": split_name}

    def _generate_class_distribution(self, y_full: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> dict[str, dict[str, Any]]:
        """Generate the NORMAL/ATTACK distribution for the full and split datasets."""
        class_distribution = {
            "full": self._ensure_split_balance(y_full, y_full, "FULL"),
            "train": self._ensure_split_balance(y_full, y_train, "TRAIN"),
            "val": self._ensure_split_balance(y_full, y_val, "VALIDATION"),
            "test": self._ensure_split_balance(y_full, y_test, "TEST"),
        }
        return class_distribution

    def _generate_attack_distribution(self, attack_categories: pd.Series, split_names: dict[str, pd.Series]) -> dict[str, dict[str, int]]:
        """Generate the original attack category distribution by split."""
        result: dict[str, dict[str, int]] = {}
        for name, values in split_names.items():
            result[name] = values.value_counts().to_dict()
        result["full"] = attack_categories.value_counts().to_dict()
        return result

    def _generate_missing_values_summary(self, X: pd.DataFrame) -> dict[str, int]:
        """Return the count of missing values per column."""
        return {column: int(X[column].isna().sum()) for column in X.columns}

    def _generate_report_text(self, summary: dict[str, Any]) -> str:
        """Build the final validation report in plain text."""
        report_lines = [
            "==================================================",
            "CYBERSECURITY DATA PREPARATION REPORT",
            "==================================================",
            "",
            "DATASET OVERVIEW",
            f"Total Samples: {summary['total_samples']}",
            f"Total Features: {summary['total_features']}",
            f"Original Dataset Shape: {summary['original_dataset_shape']}",
            f"Processed Dataset Shape: {summary['processed_dataset_shape']}",
            "",
            "DATA CLEANING",
            f"Duplicates Removed: {summary['duplicates_removed']}",
            f"Infinity Values Replaced: {summary['infinity_values_replaced']}",
            f"Missing Values Imputed: {summary['missing_values_imputed']}",
            f"Rows Removed Due To Missing Target: {summary['rows_removed_due_to_missing_target']}",
            "",
            "MISSING VALUES",
        ]

        for column, count in summary["missing_values_per_column"].items():
            report_lines.append(f"{column}: {count}")

        report_lines.extend([
            "",
            "BINARY CLASS DISTRIBUTION",
            f"NORMAL: {summary['class_distribution']['full']['count']['NORMAL']}",
            f"ATTACK: {summary['class_distribution']['full']['count']['ATTACK']}",
            "",
            "ATTACK CATEGORY DISTRIBUTION",
        ])

        for category, count in summary["attack_distribution"]["full"].items():
            report_lines.append(f"{category}: {count}")

        report_lines.extend([
            "",
            "DATASET SPLIT",
            f"Training Samples: {summary['split_counts']['train']}",
            f"Validation Samples: {summary['split_counts']['val']}",
            f"Test Samples: {summary['split_counts']['test']}",
            "",
            "CLASS DISTRIBUTION BY SPLIT",
        ])

        for split_name in ["train", "val", "test"]:
            report_lines.append(
                f"{split_name.upper()} - NORMAL: {summary['class_distribution'][split_name]['count']['NORMAL']} "
                f"({summary['class_distribution'][split_name]['percentage']['NORMAL']}%) | "
                f"ATTACK: {summary['class_distribution'][split_name]['count']['ATTACK']} "
                f"({summary['class_distribution'][split_name]['percentage']['ATTACK']}%)"
            )

        report_lines.extend([
            "",
            "DATA LEAKAGE VALIDATION",
            f"Training vs Validation: {summary['data_leakage']['train_vs_validation']}",
            f"Training vs Test: {summary['data_leakage']['train_vs_test']}",
            f"Validation vs Test: {summary['data_leakage']['validation_vs_test']}",
            "",
            "FINAL STATUS",
            "DATASET IS READY FOR MACHINE LEARNING TRAINING",
            "==================================================",
        ])

        return "\n".join(report_lines)

    def _save_report_files(self, report_text: str, summary: dict[str, Any], output_dir: Path) -> None:
        """Save the text and JSON reports to the project reports directories."""
        ensure_directory(output_dir)
        ensure_directory(output_dir / "metrics")
        ensure_directory(output_dir / "figures")

        report_path = output_dir / "data_preparation_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        json_path = output_dir / "metrics" / "data_preparation_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    def _save_visualizations(self, binary_y: pd.Series, attack_categories: pd.Series, output_dir: Path) -> None:
        """Generate and save class distribution and attack category charts."""
        ensure_directory(output_dir)

        class_counts = pd.Series(binary_y.value_counts()).sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(["NORMAL", "ATTACK"], [class_counts.get(0, 0), class_counts.get(1, 0)], color=["steelblue", "darkorange"])
        plt.title("Normal vs Attack Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "normal_vs_attack_distribution.png", dpi=150)
        plt.close()

        category_counts = attack_categories.value_counts()
        plt.figure(figsize=(8, 5))
        category_counts.plot(kind="bar", color="seagreen")
        plt.title("Original Attack Category Distribution")
        plt.xlabel("Attack Category")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "original_attack_category_distribution.png", dpi=150)
        plt.close()

    def validate_full_pipeline(
        self,
        X: pd.DataFrame,
        binary_y: pd.Series,
        attack_categories: pd.Series,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        attack_categories_train: pd.Series,
        attack_categories_val: pd.Series,
        attack_categories_test: pd.Series,
        preprocessing_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all dataset validation checks and generate summary artifacts."""
        processed_info = self._validate_original_dataset(X, binary_y, attack_categories, preprocessing_metadata)
        split_info = self._validate_splits(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            attack_categories_train, attack_categories_val, attack_categories_test,
        )

        class_distribution = self._generate_class_distribution(binary_y, y_train, y_val, y_test)
        attack_distribution = self._generate_attack_distribution(
            attack_categories,
            {
                "train": attack_categories_train,
                "val": attack_categories_val,
                "test": attack_categories_test,
            },
        )

        missing_values = self._generate_missing_values_summary(X)

        summary = {
            "total_samples": int(len(X)),
            "total_features": int(X.shape[1]),
            "original_dataset_shape": list(X.shape),
            "processed_dataset_shape": list(X.shape),
            "duplicates_removed": preprocessing_metadata.get("duplicates_removed", 0) if preprocessing_metadata else 0,
            "infinity_values_replaced": preprocessing_metadata.get("infinity_values_replaced", 0) if preprocessing_metadata else 0,
            "missing_values_imputed": preprocessing_metadata.get("missing_values_imputed", 0) if preprocessing_metadata else 0,
            "rows_removed_due_to_missing_target": preprocessing_metadata.get("rows_removed_due_to_missing_target", 0) if preprocessing_metadata else 0,
            "missing_values_per_column": missing_values,
            "class_distribution": class_distribution,
            "attack_distribution": attack_distribution,
            "split_counts": {
                "train": int(len(X_train)),
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "data_leakage": split_info,
            "processed_dataset_valid": processed_info["processed_dataset_valid"],
        }

        report_text = self._generate_report_text(summary)

        reports_dir = Path("reports")
        self._save_report_files(report_text, summary, reports_dir)
        self._save_visualizations(binary_y, attack_categories, reports_dir / "figures")

        warning_messages = []
        full_normal = class_distribution["full"]["count"]["NORMAL"]
        full_attack = class_distribution["full"]["count"]["ATTACK"]

        for split_name in ["train", "val", "test"]:
            split_normal = class_distribution[split_name]["count"]["NORMAL"]
            split_attack = class_distribution[split_name]["count"]["ATTACK"]
            split_total = split_normal + split_attack
            if split_total > 0:
                split_normal_pct = (split_normal / split_total) * 100
                split_attack_pct = (split_attack / split_total) * 100
                full_normal_pct = (full_normal / (full_normal + full_attack)) * 100 if (full_normal + full_attack) > 0 else 0
                full_attack_pct = (full_attack / (full_normal + full_attack)) * 100 if (full_normal + full_attack) > 0 else 0
                if abs(split_normal_pct - full_normal_pct) > 10 or abs(split_attack_pct - full_attack_pct) > 10:
                    warning_messages.append(f"{split_name} class balance differs significantly from the original dataset.")

        flat_class_distribution = {
            "full": {
                "NORMAL": class_distribution["full"]["count"]["NORMAL"],
                "ATTACK": class_distribution["full"]["count"]["ATTACK"],
            },
            "train": {
                "NORMAL": class_distribution["train"]["count"]["NORMAL"],
                "ATTACK": class_distribution["train"]["count"]["ATTACK"],
            },
            "val": {
                "NORMAL": class_distribution["val"]["count"]["NORMAL"],
                "ATTACK": class_distribution["val"]["count"]["ATTACK"],
            },
            "test": {
                "NORMAL": class_distribution["test"]["count"]["NORMAL"],
                "ATTACK": class_distribution["test"]["count"]["ATTACK"],
            },
        }

        validation_status = {
            "is_valid": True,
            "warnings": warning_messages,
            "processed_dataset": processed_info,
            "split_validation": split_info,
            "class_distribution": flat_class_distribution,
            "attack_distribution": attack_distribution,
            "data_leakage": split_info,
            "report_path": str(reports_dir / "data_preparation_report.txt"),
            "summary_path": str(reports_dir / "metrics" / "data_preparation_summary.json"),
        }

        self.logger.info("Dataset validation completed successfully.")
        print(report_text)
        print(f"\nSummary saved to: {reports_dir / 'metrics' / 'data_preparation_summary.json'}")
        print(f"Report saved to: {reports_dir / 'data_preparation_report.txt'}")
        print("Validation completed successfully. Dataset is ready for ML training.")

        return validation_status
