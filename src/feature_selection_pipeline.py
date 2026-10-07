"""Step 4: train-only feature selection for extracted datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_selection import FeatureSelector
from src.utils import ensure_directory


def _find_label_column(columns: pd.Index) -> str:
    """Find the target column without modifying the input dataframe."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column was found in the extracted dataset.")
    return matches[0]


def _dataset_files(directory: Path) -> list[Path]:
    """Return extracted CSVs in a deterministic order."""
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )


def select_features_from_datasets(
    extracted_directory: Path,
    output_directory: Path,
    artifact_directory: Path,
    top_n_features: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit one selector on a temporary training pool and reuse it everywhere."""
    csv_files = _dataset_files(extracted_directory)
    if not csv_files:
        raise FileNotFoundError(f"No extracted CSV datasets found under '{extracted_directory}'.")

    training_frames: list[pd.DataFrame] = []
    dataset_frames: list[tuple[Path, pd.DataFrame, str, pd.Series]] = []
    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        label_column = _find_label_column(dataframe.columns)
        labels = dataframe[label_column].copy()
        features = dataframe.drop(columns=[label_column]).copy()
        numeric_features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        training_frames.append(numeric_features.assign(_dataset_identity=csv_path.name))
        dataset_frames.append((csv_path, numeric_features, label_column, labels))

    # This pool exists only for fitting; no merged dataset is written or passed downstream.
    training_pool = pd.concat(training_frames, ignore_index=True)
    training_features = training_pool.drop(columns=["_dataset_identity"])
    training_labels = pd.concat([frame[3] for frame in dataset_frames], ignore_index=True)
    selector = FeatureSelector(top_n_features=top_n_features, random_state=random_state)
    selector.fit(training_features, training_labels)
    selected_features = selector.get_selected_features()

    ensure_directory(artifact_directory)
    selector_path = artifact_directory / "feature_selector.joblib"
    selected_names_path = artifact_directory / "selected_feature_names.txt"
    joblib.dump(selector, selector_path)
    selected_names_path.write_text("\n".join(selected_features) + "\n", encoding="utf-8")

    ensure_directory(output_directory)
    dataset_summaries: list[dict[str, Any]] = []
    for csv_path, features, label_column, labels in dataset_frames:
        selected = selector.transform(features)
        selected["Label"] = labels.to_numpy()
        output_path = output_directory / csv_path.name
        selected.to_csv(output_path, index=False)
        dataset_summaries.append({
            "dataset": csv_path.name,
            "features_before": len(features.columns),
            "features_after": len(selected_features),
            "selected_features": selected_features,
            "output_path": output_path,
        })

    report_lines = [
        "=" * 62,
        "FEATURE SELECTION",
        "=" * 62,
        f"Training pool datasets: {len(dataset_frames)}",
        f"Training pool rows: {len(training_pool)}",
        f"Features before selection: {len(training_features.columns)}",
        f"Features after selection: {len(selected_features)}",
        "Selected feature names:",
        *[f"  - {feature}" for feature in selected_features],
        "",
    ]
    for summary in dataset_summaries:
        report_lines.extend([
            f"Dataset: {summary['dataset']}",
            f"  Features before selection: {summary['features_before']}",
            f"  Features after selection: {summary['features_after']}",
            f"  Output: {summary['output_path']}",
            "",
        ])
    report_lines.extend([
        f"Fitted selector: {selector_path}",
        f"Selected feature list: {selected_names_path}",
        "=" * 62,
    ])
    report_path = artifact_directory / "feature_selection_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "selector": selector,
        "selector_path": selector_path,
        "selected_names_path": selected_names_path,
        "report_path": report_path,
        "features_before": len(training_features.columns),
        "features_after": len(selected_features),
        "selected_features": selected_features,
        "datasets": dataset_summaries,
    }