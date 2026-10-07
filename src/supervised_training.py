"""Step 5: supervised Random Forest training for known attacks only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.attack_configuration import configure_attacks
from src.config import get_default_config
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.utils import ensure_directory


def _label_column(dataframe: pd.DataFrame) -> str:
    return next(
        column for column in dataframe.columns
        if str(column).strip().lower() == "label"
    )


def _unknown_counts(dataframe: pd.DataFrame, label_column: str, unknown_attacks: list[str]) -> int:
    return int(dataframe[label_column].astype(str).str.strip().isin(set(unknown_attacks)).sum())


def _prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Extract flow features and fit numeric cleanup values on train only."""
    feature_columns = [column for column in train.columns if column != label_column]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding Label.")
    if feature_columns != [column for column in validation.columns if column != label_column]:
        raise ValueError("Training and validation feature columns are not identical.")
    if feature_columns != [column for column in test.columns if column != label_column]:
        raise ValueError("Training and test feature columns are not identical.")

    train_features = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    validation_features = validation[feature_columns].apply(pd.to_numeric, errors="coerce")
    test_features = test[feature_columns].apply(pd.to_numeric, errors="coerce")
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    validation_features = validation_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)
    medians = train_features.median().fillna(0.0)
    return (
        train_features.fillna(medians).fillna(0.0),
        validation_features.fillna(medians).fillna(0.0),
        test_features.fillna(medians).fillna(0.0),
        feature_columns,
    )


def _extract_and_clean_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Extract features independently, then share the training imputation values."""
    engineer = FeatureEngineer()
    extracted = [engineer.transform(frame) for frame in (train, validation, test)]
    feature_columns = [column for column in extracted[0].columns if column != label_column]
    if not feature_columns:
        raise ValueError("No feature columns remain after feature extraction.")

    numeric_frames = [
        frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for frame in extracted
    ]
    medians = numeric_frames[0].median().fillna(0.0)
    cleaned = [frame.fillna(medians).fillna(0.0) for frame in numeric_frames]
    return cleaned[0], cleaned[1], cleaned[2], feature_columns


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def train_known_attack_model(split_directory: Path, master_path: Path, output_directory: Path) -> dict[str, Any]:
    """Train, validate, evaluate known test data, and save Step 5 artifacts."""
    train = pd.read_csv(split_directory / "train.csv")
    validation = pd.read_csv(split_directory / "validation.csv")
    test = pd.read_csv(split_directory / "test.csv")
    known_attacks, unknown_attacks, _ = configure_attacks(master_path)
    label_column = _label_column(train)

    train_labels = train[label_column].astype(str).str.strip()
    validation_labels = validation[label_column].astype(str).str.strip()
    test_labels = test[label_column].astype(str).str.strip()
    unknown_train = _unknown_counts(train, label_column, unknown_attacks)
    unknown_validation = _unknown_counts(validation, label_column, unknown_attacks)
    if unknown_train:
        raise ValueError("ERROR: Unknown attack leakage detected in training data.")
    if unknown_validation:
        raise ValueError("ERROR: Unknown attack leakage detected in validation data.")

    train_features, validation_features, test_features, extracted_features = _extract_and_clean_features(
        train, validation, test, label_column
    )
    config = get_default_config()
    selector = FeatureSelector(
        top_n_features=config.feature_selection.top_n_features,
        random_state=config.splitting.random_state,
    )
    selector.fit(train_features, train_labels)
    selected_train = selector.transform(train_features)
    selected_validation = selector.transform(validation_features)
    selected_test = selector.transform(test_features)

    scaler = StandardScaler()
    normalized_train = pd.DataFrame(
        scaler.fit_transform(selected_train), columns=selector.selected_features, index=selected_train.index
    )
    normalized_validation = pd.DataFrame(
        scaler.transform(selected_validation), columns=selector.selected_features, index=selected_validation.index
    )
    normalized_test = pd.DataFrame(
        scaler.transform(selected_test), columns=selector.selected_features, index=selected_test.index
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(normalized_train, train_labels)

    validation_metrics = _metrics(validation_labels, model.predict(normalized_validation))
    known_test_mask = ~test_labels.isin(set(unknown_attacks))
    known_test_features = normalized_test.loc[known_test_mask].reset_index(drop=True)
    known_test_labels = test_labels.loc[known_test_mask].reset_index(drop=True)
    known_test_metrics = _metrics(known_test_labels, model.predict(known_test_features))
    unknown_test = test.loc[~known_test_mask].copy()

    models_directory = ensure_directory(output_directory / "models")
    results_directory = ensure_directory(output_directory / "results")
    model_path = models_directory / "random_forest_known_attacks.pkl"
    feature_path = models_directory / "feature_columns.pkl"
    scaler_path = models_directory / "standard_scaler.pkl"
    selected_features_path = models_directory / "selected_features.pkl"
    joblib.dump(model, model_path)
    joblib.dump(selector.selected_features, feature_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(selector.selected_features, selected_features_path)
    selector.importance_report.to_csv(results_directory / "feature_importance_report.csv", index=False)

    results = pd.DataFrame([{
        "Model": "Random Forest",
        "Training Samples": len(train),
        "Validation Samples": len(validation),
        "Known Test Samples": len(known_test_labels),
        "Extracted Features": len(extracted_features),
        "Number of Features": len(selector.selected_features),
        "Accuracy": validation_metrics["accuracy"],
        "Precision": validation_metrics["precision"],
        "Recall": validation_metrics["recall"],
        "F1 Score": validation_metrics["f1"],
    }])
    results.to_csv(results_directory / "supervised_training_results.csv", index=False)
    report_text = "\n".join([
        "VALIDATION CLASSIFICATION REPORT",
        validation_metrics["classification_report"],
        "VALIDATION CONFUSION MATRIX",
        str(validation_metrics["confusion_matrix"]),
        "",
        "KNOWN TEST CLASSIFICATION REPORT",
        known_test_metrics["classification_report"],
        "KNOWN TEST CONFUSION MATRIX",
        str(known_test_metrics["confusion_matrix"]),
    ])
    (results_directory / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame([{
        "features_before_extraction": len(train.columns) - 1,
        "features_after_extraction": len(extracted_features),
        "selected_features": len(selector.selected_features),
        "train_shape": str(normalized_train.shape),
        "validation_shape": str(normalized_validation.shape),
        "test_shape": str(normalized_test.shape),
        "unknown_train": unknown_train,
        "unknown_validation": unknown_validation,
        "unknown_test": len(unknown_test),
        "leakage_check": "PASSED",
    }]).to_csv(results_directory / "feature_pipeline_report.csv", index=False)

    return {
        "known_attacks": known_attacks,
        "unknown_attacks": unknown_attacks,
        "training_samples": len(train),
        "validation_samples": len(validation),
        "known_test_samples": len(known_test_labels),
        "unknown_test_samples": len(unknown_test),
        "unknown_train": unknown_train,
        "unknown_validation": unknown_validation,
        "validation": validation_metrics,
        "known_test": known_test_metrics,
        "feature_count": len(selector.selected_features),
        "features_before_extraction": len(train.columns) - 1,
        "features_after_extraction": len(extracted_features),
        "selected_features": selector.selected_features,
        "train_shape": normalized_train.shape,
        "validation_shape": normalized_validation.shape,
        "test_shape": normalized_test.shape,
        "model_path": model_path,
        "feature_path": feature_path,
        "scaler_path": scaler_path,
        "importance_path": results_directory / "feature_importance_report.csv",
    }


def format_training_report(result: dict[str, Any]) -> str:
    """Format the requested Step 5 console output."""
    validation = result["validation"]
    known_test = result["known_test"]
    return "\n".join([
        "=" * 62,
        "TRAINING DATA",
        "=" * 62,
        f"Training samples: {result['training_samples']}",
        f"Features before extraction: {result['features_before_extraction']}",
        f"Features after extraction: {result['features_after_extraction']}",
        f"Selected features: {result['feature_count']}",
        f"Top selected features: {result['selected_features'][:10]}",
        "",
        "Classes used for training:",
        *[f"- {label}" for label in result["known_attacks"]],
        "- BENIGN",
        "",
        f"Unknown attack samples in training: {result['unknown_train']}",
        "=" * 62,
        "",
        "Validation samples: " + str(result["validation_samples"]),
        f"Unknown attack samples in validation: {result['unknown_validation']}",
        "=" * 62,
        "MODEL TRAINING",
        "=" * 62,
        "Algorithm:\nRandom Forest",
        f"Training samples:\n{result['training_samples']}",
        f"Features:\n{result['feature_count']}",
        f"Classes learned:\n{result['known_attacks'] + ['BENIGN']}",
        "Training completed successfully.",
        "=" * 62,
        "VALIDATION RESULTS",
        "=" * 62,
        f"Accuracy: {validation['accuracy'] * 100:.2f}%",
        f"Precision: {validation['precision'] * 100:.2f}%",
        f"Recall: {validation['recall'] * 100:.2f}%",
        f"F1-score: {validation['f1'] * 100:.2f}%",
        "",
        "Classification Report:",
        validation["classification_report"],
        "Confusion Matrix:",
        str(validation["confusion_matrix"]),
        "=" * 62,
        "KNOWN TEST RESULTS",
        "=" * 62,
        f"Known test samples: {result['known_test_samples']}",
        f"Accuracy: {known_test['accuracy'] * 100:.2f}%",
        f"Precision: {known_test['precision'] * 100:.2f}%",
        f"Recall: {known_test['recall'] * 100:.2f}%",
        f"F1-score: {known_test['f1'] * 100:.2f}%",
        "=" * 62,
        "UNKNOWN ATTACK TEST DATA",
        "=" * 62,
        f"Unknown attack classes:\n{result['unknown_attacks']}",
        f"Unknown attack samples: {result['unknown_test_samples']}",
        "Status:\nRESERVED FOR UNKNOWN ATTACK DETECTION",
        "=" * 62,
        "STEP 5 – SUPERVISED TRAINING",
        "=" * 62,
        "Algorithm: Random Forest",
        f"Known attacks: {result['known_attacks']}",
        f"Unknown attacks withheld: {result['unknown_attacks']}",
        f"Training samples: {result['training_samples']}",
        f"Validation samples: {result['validation_samples']}",
        f"Known test samples: {result['known_test_samples']}",
        f"Unknown test samples: {result['unknown_test_samples']}",
        f"Unknown attacks in training: {result['unknown_train']}",
        f"Unknown attacks in validation: {result['unknown_validation']}",
        f"Validation Accuracy: {validation['accuracy'] * 100:.2f}%",
        f"Validation F1: {validation['f1'] * 100:.2f}%",
        f"Known Test Accuracy: {known_test['accuracy'] * 100:.2f}%",
        f"Known Test F1: {known_test['f1'] * 100:.2f}%",
        f"Model saved: {result['model_path']}",
        f"Scaler saved: {result['scaler_path']}",
        "STATUS:\nSTEP 5 COMPLETED",
        "=" * 62,
    ])