"""Step 7: supervised training on the Step 6 train and validation sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.utils import ensure_directory


def _dataset_files(directory: Path) -> list[Path]:
    """Return split files in a deterministic order."""
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )


def _load_split(directory: Path, expected_features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load one split directory while preserving its shared feature contract."""
    files = _dataset_files(directory)
    if not files:
        raise FileNotFoundError(f"No split CSV datasets found under '{directory}'.")
    frames: list[pd.DataFrame] = []
    labels: list[pd.Series] = []
    feature_columns = expected_features
    for csv_path in files:
        dataframe = pd.read_csv(csv_path)
        label_matches = [column for column in dataframe.columns if str(column).strip().lower() == "label"]
        if not label_matches:
            raise ValueError(f"No Label column found in '{csv_path.name}'.")
        label_column = label_matches[0]
        current_features = [column for column in dataframe.columns if column != label_column]
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(f"Feature order mismatch in '{csv_path.name}'.")
        frames.append(dataframe[feature_columns].copy())
        labels.append(dataframe[label_column].copy())
    return pd.concat(frames, ignore_index=True), pd.concat(labels, ignore_index=True), feature_columns or []


def train_all_models(
    splits_directory: Path,
    output_directory: Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and save three models using only the Step 6 training data."""
    train_features, train_labels, feature_columns = _load_split(splits_directory / "train")
    if len(feature_columns) != 20:
        raise ValueError("Training data must contain exactly 20 selected features in the same order.")

    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels.astype(str))
    class_count = len(label_encoder.classes_)
    if class_count < 2:
        raise ValueError("Training data must contain at least two classes.")

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            objective="multiclass",
            num_class=class_count,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    models_directory = ensure_directory(output_directory)
    classes_path = models_directory / "model_classes.json"
    class_names = label_encoder.classes_.tolist()
    classes_path.write_text(json.dumps({name: class_names for name in models}, indent=2) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    print("=" * 62)
    print("STEP 7: MULTI-MODEL SUPERVISED TRAINING")
    print("=" * 62)
    print(f"\nTraining Samples: {len(train_features)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {class_names}")
    print("Training source: output/splits/train")
    print("Validation and test data are not loaded or used.")
    for model_name, model in models.items():
        display_name = model_name.replace("_", " ").upper()
        print("\n" + "-" * 62)
        print(f"MODEL {list(models).index(model_name) + 1}: {display_name}")
        print("-" * 62)
        print(f"Algorithm: {model.__class__.__name__}")
        if model_name == "random_forest":
            print(f"Number of Trees: {model.n_estimators}")
        print(f"Training Samples: {len(train_features)}")
        print(f"Features: {len(feature_columns)}")
        print(f"Classes Learned: {class_names}")
        model.fit(train_features, encoded_train_labels)
        model_path = models_directory / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        results.append({
            "model_name": model_name,
            "training_samples": len(train_features),
            "features": len(feature_columns),
            "classes": class_names,
            "model_path": model_path,
        })
        print("Status: TRAINING COMPLETED")

    report_lines = [
        "=" * 62,
        "MULTI-MODEL SUPERVISED TRAINING",
        "=" * 62,
        "Training source: output/splits/train",
        "Validation and test sources: not loaded or used",
        f"Training samples: {len(train_features)}",
        f"Number of features: {len(feature_columns)}",
        f"Classes: {class_names}",
        "",
    ]
    for result in results:
        report_lines.extend([
            f"Model name: {result['model_name']}",
            f"Training samples: {result['training_samples']}",
            f"Number of features: {result['features']}",
            f"Classes learned: {result['classes']}",
            f"Saved model: {result['model_path']}",
            "",
        ])
    report_path = models_directory / "training_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "results": results,
        "classes_path": classes_path,
        "report_path": report_path,
        "feature_columns": feature_columns,
    }