"""Step 5: train-pool-only normalization for selected datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_directory


def _dataset_files(directory: Path) -> list[Path]:
    """Return selected datasets in a deterministic order."""
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )


def _find_label_column(columns: pd.Index) -> str:
    """Find the target column without modifying the input dataframe."""
    matches = [column for column in columns if str(column).strip().lower() == "label"]
    if not matches:
        raise ValueError("No Label column was found in the selected dataset.")
    return matches[0]


def normalize_selected_datasets(
    selected_directory: Path,
    output_directory: Path,
    models_directory: Path,
    report_directory: Path,
    selected_names_path: Path,
) -> dict[str, Any]:
    """Fit one scaler on the temporary training pool and reuse it everywhere."""
    csv_files = _dataset_files(selected_directory)
    if not csv_files:
        raise FileNotFoundError(f"No feature-selected CSV datasets found under '{selected_directory}'.")
    if not selected_names_path.is_file():
        raise FileNotFoundError(f"Selected feature list was not found: '{selected_names_path}'.")

    selected_features = [
        line.strip() for line in selected_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not selected_features:
        raise ValueError("The selected feature list is empty.")

    dataset_frames: list[tuple[Path, pd.DataFrame, pd.Series]] = []
    training_frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        label_column = _find_label_column(dataframe.columns)
        actual_features = [column for column in dataframe.columns if column != label_column]
        if actual_features != selected_features:
            raise ValueError(
                f"Feature order mismatch in '{csv_path.name}'. "
                f"Expected {selected_features}, found {actual_features}."
            )
        features = dataframe[selected_features].copy()
        labels = dataframe[label_column].copy()
        training_frames.append(features)
        dataset_frames.append((csv_path, features, labels))

    # This pool is temporary and is used only by scaler.fit; it is never saved.
    training_pool = pd.concat(training_frames, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(training_pool)

    ensure_directory(models_directory)
    scaler_path = models_directory / "standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    ensure_directory(output_directory)
    dataset_summaries: list[dict[str, Any]] = []
    for csv_path, features, labels in dataset_frames:
        normalized_features = pd.DataFrame(
            scaler.transform(features),
            columns=selected_features,
        )
        normalized_features["Label"] = labels.to_numpy()
        output_path = output_directory / csv_path.name
        normalized_features.to_csv(output_path, index=False)
        dataset_summaries.append({
            "dataset": csv_path.name,
            "features_before": len(selected_features),
            "features_after": len(selected_features),
            "output_path": output_path,
        })

    ensure_directory(report_directory)
    report_path = report_directory / "normalization_report.txt"
    report_lines = [
        "=" * 62,
        "NORMALIZATION",
        "=" * 62,
        f"Features before normalization: {len(selected_features)}",
        f"Features after normalization: {len(selected_features)}",
        "Scaler used: sklearn.preprocessing.StandardScaler",
        "Training data used for fitting:",
        "  Temporary pool of all feature-selected datasets listed below.",
        f"  Training pool rows: {len(training_pool)}",
        "  The pool was used only for scaler.fit and was not saved.",
        f"Scaler location: {scaler_path}",
        "",
        "Normalized datasets:",
    ]
    report_lines.extend(f"  - {summary['output_path']}" for summary in dataset_summaries)
    report_lines.extend(["", "Feature order:", *[f"  - {feature}" for feature in selected_features], "=" * 62])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "scaler": scaler,
        "scaler_path": scaler_path,
        "report_path": report_path,
        "features_before": len(selected_features),
        "features_after": len(selected_features),
        "selected_features": selected_features,
        "datasets": dataset_summaries,
        "training_pool_rows": len(training_pool),
    }