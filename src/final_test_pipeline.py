"""Step 9: final evaluation on the held-out test dataset only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.multi_model_training import _load_split
from src.model_validation_pipeline import MODEL_NAMES
from src.utils import ensure_directory


def evaluate_final_test(
    test_directory: Path,
    models_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate saved models on test data without fitting or reading other splits."""
    class_names_path = models_directory / "model_classes.json"
    if not class_names_path.is_file():
        raise FileNotFoundError(f"Model class metadata was not found: '{class_names_path}'.")
    class_names_by_model = json.loads(class_names_path.read_text(encoding="utf-8"))
    test_features, test_labels, feature_columns = _load_split(test_directory)
    if len(feature_columns) != 20:
        raise ValueError("Test data must contain exactly 20 normalized selected features.")

    labels = test_labels.astype(str)
    class_names = class_names_by_model[MODEL_NAMES[0]]
    if sorted(labels.unique()) != sorted(class_names):
        raise ValueError("Test labels do not match the classes learned during training.")

    ensure_directory(output_directory)
    results: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        model_path = models_directory / f"{model_name}.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model was not found: '{model_path}'.")
        model = joblib.load(model_path)
        encoded_predictions = model.predict(test_features)
        if getattr(encoded_predictions, "ndim", 1) > 1:
            encoded_predictions = encoded_predictions.argmax(axis=1)
        predictions = pd.Series(encoded_predictions).map(lambda index: class_names[int(index)]).astype(str)
        model_classes = class_names_by_model[model_name]
        metrics = {
            "model_name": model_name,
            "test_samples": len(test_features),
            "features": len(feature_columns),
            "classes": model_classes,
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
            "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
            "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            "classification_report": classification_report(
                labels,
                predictions,
                labels=model_classes,
                target_names=model_classes,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=model_classes).tolist(),
        }
        results[model_name] = metrics
        (output_directory / f"{model_name}_classification_report.txt").write_text(
            metrics["classification_report"], encoding="utf-8"
        )
        pd.DataFrame(metrics["confusion_matrix"], index=model_classes, columns=model_classes).to_csv(
            output_directory / f"{model_name}_confusion_matrix.csv"
        )

    comparison = pd.DataFrame([
        {
            "Model": name.replace("_", " ").title(),
            "Accuracy": values["accuracy"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1"],
        }
        for name, values in results.items()
    ])
    comparison.to_csv(output_directory / "model_comparison.csv", index=False)
    report_path = output_directory / "final_test_report.txt"
    report_lines = [
        "=" * 62,
        "STEP 9: FINAL TEST RESULTS",
        "=" * 62,
        "Test source: output/splits/test",
        "Training and validation data were not used.",
        f"Test Samples: {len(test_features)}",
        f"Features: {len(feature_columns)}",
        f"Classes: {class_names}",
        "",
    ]
    for model_name, metrics in results.items():
        report_lines.extend([
            "-" * 62,
            model_name.replace("_", " ").upper(),
            "-" * 62,
            f"Accuracy: {metrics['accuracy']:.6f}",
            f"Precision: {metrics['precision']:.6f}",
            f"Recall: {metrics['recall']:.6f}",
            f"F1-score: {metrics['f1']:.6f}",
            "",
            "Classification Report:",
            metrics["classification_report"],
            "Confusion Matrix:",
            str(metrics["confusion_matrix"]),
            "",
        ])
    report_lines.extend([
        "-" * 62,
        "FINAL MODEL COMPARISON",
        "-" * 62,
        comparison.to_string(index=False),
        "",
        "=" * 62,
        "STEP 9 COMPLETED",
        "=" * 62,
    ])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "test_samples": len(test_features),
        "features": feature_columns,
        "classes": class_names,
        "results": results,
        "report_path": report_path,
        "output_directory": output_directory,
    }