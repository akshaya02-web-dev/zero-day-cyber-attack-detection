"""Dataset-isolated preprocessing, training, and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_inspection import inspect_dataset
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.preprocessing import Preprocessor
from src.utils import ensure_directory


MODEL_NAMES = ("random_forest", "xgboost", "lightgbm")


def _csv_files(raw_directory: Path) -> list[Path]:
    return sorted(
        (path for path in raw_directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )


def _label_column(columns: pd.Index) -> str:
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column found.")
    return matches[0]


def _clean_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    original = dataframe.copy()
    raw_numeric = original.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(raw_numeric.to_numpy()).sum()) if not raw_numeric.empty else 0
    missing_count = int(original.isna().sum().sum())
    cleaned = Preprocessor().clean_data(original)
    label = _label_column(cleaned.columns)
    cleaned = cleaned.rename(columns={label: "Label"})
    duplicate_count = int(cleaned.duplicated().sum())
    summary = {
        "original_rows": len(original),
        "original_features": len(original.columns) - 1,
        "duplicates_removed": duplicate_count,
        "missing_values": missing_count,
        "infinite_values": infinite_count,
    }
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).drop_duplicates().reset_index(drop=True)
    return cleaned, summary


def _split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_first = dataframe["Label"] if dataframe["Label"].value_counts().min() >= 2 else None
    train, remainder = train_test_split(dataframe, train_size=0.70, random_state=42, stratify=stratify_first)
    stratify_second = remainder["Label"] if remainder["Label"].value_counts().min() >= 2 else None
    validation, test = train_test_split(remainder, test_size=0.50, random_state=42, stratify=stratify_second)
    if set(train.index) & set(validation.index) or set(train.index) & set(test.index) or set(validation.index) & set(test.index):
        raise ValueError("Dataset split overlap detected.")
    return train, validation, test


def _numeric(frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = values.median().fillna(0.0)
    return values.fillna(medians).fillna(0.0), medians


def _models(class_count: int) -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", num_class=class_count,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            objective="multiclass", num_class=class_count, random_state=42,
            n_jobs=-1, verbosity=-1,
        ),
    }


def _predict_labels(model: Any, features: pd.DataFrame, classes: list[str]) -> pd.Series:
    prediction = model.predict(features)
    if getattr(prediction, "ndim", 1) > 1:
        prediction = prediction.argmax(axis=1)
    return pd.Series(prediction).map(lambda value: classes[int(value)]).astype(str)


def _metrics(labels: pd.Series, predictions: pd.Series, classes: list[str]) -> dict[str, Any]:
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "F1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(labels, predictions, labels=classes, target_names=classes, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


def _preprocess_dataset(csv_path: Path, output_root: Path) -> dict[str, Any]:
    name = csv_path.stem
    processed = ensure_directory(output_root / "processed" / name)
    splits = output_root / "splits" / name
    models_directory = ensure_directory(output_root / "models" / name)
    results_directory = ensure_directory(output_root / "results" / name)
    raw = pd.read_csv(csv_path)
    cleaned, cleaning = _clean_frame(raw)
    cleaned.to_csv(processed / "cleaned.csv", index=False)
    inspection = inspect_dataset(csv_path)
    (processed / "inspection_report.txt").write_text(inspection + "\n", encoding="utf-8")

    labels = cleaned["Label"].copy()
    features = cleaned.drop(columns=["Label"])
    extracted_features = FeatureEngineer().transform(features)
    extracted = extracted_features.copy()
    extracted["Label"] = labels.to_numpy()
    extracted.to_csv(processed / "extracted_features.csv", index=False)
    feature_columns = [column for column in extracted_features.columns]
    prepared = extracted_features.copy()
    prepared["Label"] = labels.to_numpy()
    prepared.to_csv(processed / "prepared_features.csv", index=False)
    return {"name": name, "processed": processed, "splits": splits, "models": models_directory,
            "results": results_directory, "raw": prepared, "feature_columns": feature_columns,
            "cleaning": cleaning, "samples": {}}


def _split_dataset(record: dict[str, Any]) -> None:
    name = record["name"]
    processed = record["processed"]
    train_raw, validation_raw, test_raw = _split(record["raw"])
    feature_columns = record["feature_columns"]
    train_numeric, medians = _numeric(train_raw, feature_columns)
    validation_numeric, _ = _numeric(validation_raw, feature_columns, medians)
    test_numeric, _ = _numeric(test_raw, feature_columns, medians)
    labels_train = train_raw["Label"].astype(str)
    selector_fit_rows = min(100_000, len(train_numeric))
    if selector_fit_rows < len(train_numeric):
        selector_training, _, selector_labels, _ = train_test_split(
            train_numeric, labels_train, train_size=selector_fit_rows, random_state=42, stratify=labels_train
        )
    else:
        selector_training, selector_labels = train_numeric, labels_train
    selector = FeatureSelector(top_n_features=20, random_state=42).fit(selector_training, selector_labels)
    selected_features = selector.selected_features
    joblib.dump(selector, processed / "feature_selector.pkl")
    selector.importance_report.to_csv(processed / "feature_importance.csv", index=False)
    selected = {"train": selector.transform(train_numeric), "validation": selector.transform(validation_numeric),
                "test": selector.transform(test_numeric)}
    all_numeric, _ = _numeric(record["raw"].drop(columns="Label"), feature_columns, medians)
    selected_full = selector.transform(all_numeric)
    selected_full["Label"] = record["raw"]["Label"].to_numpy()
    selected_full.to_csv(processed / "selected_features.csv", index=False)
    scaler = StandardScaler().fit(selected["train"])
    joblib.dump(scaler, processed / "scaler.pkl")
    joblib.dump(selector, record["models"] / "feature_selector.pkl")
    joblib.dump(scaler, record["models"] / "scaler.pkl")
    normalized_full = pd.DataFrame(scaler.transform(selected_full[selected_features]), columns=selected_features)
    normalized_full["Label"] = selected_full["Label"].to_numpy()
    normalized_full.to_csv(processed / "normalized.csv", index=False)
    record["normalized_splits"] = {}
    for split_name, source in (("train", train_raw), ("validation", validation_raw), ("test", test_raw)):
        transformed = pd.DataFrame(scaler.transform(selected[split_name]), columns=selected_features)
        split_labels = source["Label"].reset_index(drop=True)
        record["normalized_splits"][split_name] = (transformed, split_labels)
    record["selected_features"] = selected_features
    record["samples"] = {key: len(value[1]) for key, value in record["normalized_splits"].items()}
    cleaning = record["cleaning"]
    lines = [f"Dataset: {name}", f"Original Samples: {cleaning['original_rows']}",
             f"Original Features: {cleaning['original_features']}", f"Duplicates Removed: {cleaning['duplicates_removed']}",
             f"Missing Values: {cleaning['missing_values']}", f"Infinite Values: {cleaning['infinite_values']}",
             f"Features Before Extraction: {cleaning['original_features']}",
             f"Features After Extraction: {len(feature_columns)}", f"Features Before Selection: {len(feature_columns)}",
             f"Selected Features: {len(selected_features)}", f"Selected Feature Names: {selected_features}",
             "Normalization: StandardScaler fitted on this dataset's training split", "Status: COMPLETED", f"Output: {processed}"]
    (processed / "preprocessing_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _train_dataset(record: dict[str, Any]) -> None:
    name = record["name"]
    models_directory = record["models"]
    results_directory = record["results"]
    normalized_splits = record["normalized_splits"]
    train_labels = normalized_splits["train"][1].astype(str)
    classes = sorted(train_labels.unique().tolist())
    training_limitations: list[str] = []
    if len(classes) < 2:
        training_limitations.append("Only one training class is available; supervised models were not fitted.")
    else:
        encoder = LabelEncoder().fit(train_labels)
        encoded_train = encoder.transform(train_labels)
        class_names = encoder.classes_.tolist()
        models = _models(len(class_names))
        joblib.dump(class_names, models_directory / "class_names.pkl")
        (models_directory / "class_names.json").write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")
        for model_name, model in models.items():
            model.fit(normalized_splits["train"][0], encoded_train)
            joblib.dump(model, models_directory / f"{model_name}.pkl")
    record["training_limitations"] = training_limitations


def _evaluate_dataset(record: dict[str, Any]) -> None:
    models_directory = record["models"]
    results_directory = record["results"]
    model_results: list[dict[str, Any]] = []
    if record.get("training_limitations"):
        record["model_results"] = model_results
        return
    class_names = joblib.load(models_directory / "class_names.pkl")
    for model_name in MODEL_NAMES:
        model = joblib.load(models_directory / f"{model_name}.pkl")
        for split_name in ("validation", "test"):
            split_features, split_labels = record["normalized_splits"][split_name]
            predictions = _predict_labels(model, split_features, class_names)
            metrics = _metrics(split_labels.astype(str), predictions, class_names)
            metrics["Model"] = model_name.replace("_", " ").title()
            metrics["Samples"] = len(split_labels)
            model_results.append({"model": model_name, "split": split_name, "metrics": metrics})
    for split_name in ("validation", "test"):
        rows = [item["metrics"] for item in model_results if item["split"] == split_name]
        if rows:
            pd.DataFrame([{key: value for key, value in row.items() if key not in {"classification_report", "confusion_matrix"}} for row in rows]).to_csv(results_directory / f"{split_name}_results.csv", index=False)
            mode = "w" if split_name == "validation" else "a"
            with (results_directory / "classification_report.txt").open(mode, encoding="utf-8") as report:
                report.write(f"{split_name.upper()}\n")
                for row in rows:
                    report.write(f"{row['Model']}\n{row['classification_report']}\n")
            confusion_rows = []
            for row in rows:
                matrix = row["confusion_matrix"]
                for true_index, values in enumerate(matrix):
                    for predicted_index, count in enumerate(values):
                        confusion_rows.append({"Model": row["Model"], "True": class_names[true_index], "Predicted": class_names[predicted_index], "Count": count})
            pd.DataFrame(confusion_rows).to_csv(results_directory / f"{split_name}_confusion_matrix.csv", index=False)
    record["model_results"] = model_results


def run_multi_dataset_pipeline(raw_directory: Path, output_root: Path) -> dict[str, Any]:
    """Run all four isolated stages for every raw dataset."""
    csv_files = _csv_files(raw_directory)
    if not csv_files:
        raise FileNotFoundError(f"No CSV datasets found under '{raw_directory}'.")
    print("=" * 70 + "\n             MULTI-DATASET PREPROCESSING AND TRAINING\n" + "=" * 70)
    print("\nDATASETS FOUND:")
    for index, path in enumerate(csv_files, 1):
        print(f"{index}. {path.stem}")
    records: list[dict[str, Any]] = []
    print("\n" + "=" * 70 + "\nSTAGE 1: PREPROCESSING\n" + "=" * 70)
    for path in csv_files:
        print(f"\n{'-' * 62}\nDATASET: {path.stem}\n{'-' * 62}")
        record = _preprocess_dataset(path, output_root)
        # Split first in memory so selector and scaler can be fitted on this
        # dataset's training partition only, while finishing every dataset's
        # preprocessing before any model training begins.
        _split_dataset(record)
        records.append(record)
        print(f"Original Samples: {record['cleaning']['original_rows']}")
        print(f"Original Features: {record['cleaning']['original_features']}")
        print(f"Duplicates Removed: {record['cleaning']['duplicates_removed']}")
        print(f"Missing Values: {record['cleaning']['missing_values']}")
        print(f"Infinite Values: {record['cleaning']['infinite_values']}")
        print(f"Features Before Extraction: {record['cleaning']['original_features']}")
        print(f"Features After Extraction: {len(record['feature_columns'])}")
        print(f"Features Before Selection: {len(record['feature_columns'])}")
        print(f"Selected Features: {len(record['selected_features'])}")
        print("Normalization:\nStatus: COMPLETED")
        print(f"Output: {record['processed']}")
    print("\n" + "=" * 70 + "\nSTAGE 1 COMPLETED\nALL DATASETS PREPROCESSED\n" + "=" * 70)
    print("\n" + "=" * 70 + "\nSTAGE 2: DATASET SPLITTING\n" + "=" * 70)
    for record in records:
        for split_name, (split_features, split_labels) in record["normalized_splits"].items():
            split_output = split_features.copy()
            split_output["Label"] = split_labels.to_numpy()
            split_directory = ensure_directory(record["splits"] / split_name)
            split_output.to_csv(split_directory / f"{record['name']}.csv", index=False)
        print(f"\nDataset: {record['name']}\nTraining Samples: {record['samples']['train']}\nValidation Samples: {record['samples']['validation']}\nTest Samples: {record['samples']['test']}")
    print("\n" + "=" * 70 + "\nSTAGE 2 COMPLETED\n" + "=" * 70)
    print("\n" + "=" * 70 + "\nSTAGE 3: SEPARATE MODEL TRAINING\n" + "=" * 70)
    for record in records:
        print(f"\n{'-' * 62}\nDATASET: {record['name']}\n{'-' * 62}")
        _train_dataset(record)
        if record["training_limitations"]:
            print(record["training_limitations"][0])
        else:
            for model_name in MODEL_NAMES:
                print(f"{model_name.replace('_', ' ').title()}:\nTraining Status: COMPLETED")
        print(f"Models saved to: {record['models']}")
    print("\n" + "=" * 70 + "\nSTAGE 3 COMPLETED\nALL DATASETS TRAINED SEPARATELY\n" + "=" * 70)
    print("\n" + "=" * 70 + "\nSTAGE 4: MODEL EVALUATION\n" + "=" * 70)
    for record in records:
        _evaluate_dataset(record)
        print(f"\nDataset: {record['name']}\nMODEL             ACCURACY    PRECISION    RECALL    F1")
        for item in record.get("model_results", []):
            if item["split"] == "test":
                metrics = item["metrics"]
                print(f"{metrics['Model']:<18}{metrics['Accuracy']:.4f}      {metrics['Precision']:.4f}      {metrics['Recall']:.4f}    {metrics['F1']:.4f}")
    print("\n" + "=" * 70 + "\nSTAGE 4 COMPLETED\n" + "=" * 70)
    limitations = [record["name"] for record in records if record.get("training_limitations")]
    print("\n" + "=" * 70 + "\nCOMPLETE PIPELINE FINISHED\n" + "=" * 70)
    print(f"Datasets Preprocessed: {len(records)}")
    print(f"Datasets Successfully Trained: {len(records) - len(limitations)}")
    print(f"Datasets With Training Limitations: {limitations or 'None'}")
    print("All outputs saved under: output/")
    return {"datasets": records, "limitations": limitations}
