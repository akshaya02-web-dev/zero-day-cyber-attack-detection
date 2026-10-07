"""Data loading utilities for cybersecurity datasets.

This module is a placeholder for reading tabular network traffic datasets,
validating schema, and preparing raw data for preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
    """Responsible for loading raw dataset files into pandas DataFrames."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.resolved_file_path: str | None = None

    def _list_csv_files(self, directory: Path) -> list[Path]:
        """Return CSV files under a directory, excluding hidden/system files."""
        return sorted(
            [p for p in directory.rglob("*.csv") if p.is_file() and not p.name.startswith(".")],
            key=lambda p: p.name.lower(),
        )

    def list_csv_files(self, directory: str | Path | None = None) -> list[Path]:
        """Return every CSV dataset below the configured directory."""
        base_dir = Path(directory) if directory is not None else Path(self.file_path)
        return self._list_csv_files(base_dir)

    def _resolve_dataset_path(self, candidate: str | None = None) -> str:
        """Resolve a dataset path whether the user passes a file, a folder, or a missing path."""
        raw_path = Path(candidate or self.file_path)

        if raw_path.exists() and raw_path.is_file():
            self.resolved_file_path = str(raw_path)
            return self.resolved_file_path

        if raw_path.exists() and raw_path.is_dir():
            csv_files = self._list_csv_files(raw_path)
            if csv_files:
                self.resolved_file_path = str(csv_files[0])
                return self.resolved_file_path
            raise FileNotFoundError(f"No CSV files found in directory '{raw_path}'.")

        if raw_path.parent.exists() and raw_path.parent.is_dir():
            csv_files = self._list_csv_files(raw_path.parent)
            if csv_files:
                self.resolved_file_path = str(csv_files[0])
                return self.resolved_file_path

        raise FileNotFoundError(f"Dataset not found at '{raw_path}'. Please provide a CSV file in the configured path.")

    def choose_csv_file(self, directory: str | Path | None = None) -> str:
        """Prompt the user to choose a CSV file from a directory and return its path."""
        base_dir = Path(directory) if directory is not None else Path(self.file_path)
        if base_dir.exists() and base_dir.is_file():
            base_dir = base_dir.parent

        csv_files = self._list_csv_files(base_dir)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory '{base_dir}'.")

        print(f"Available CSV files in '{base_dir}':")
        for i, csv_file in enumerate(csv_files, start=1):
            print(f"  {i}. {csv_file.name}")

        while True:
            try:
                choice = input("Select a CSV file by number (press Enter for the first file): ").strip()
                if not choice:
                    selected = csv_files[0]
                    break
                index = int(choice) - 1
                if 0 <= index < len(csv_files):
                    selected = csv_files[index]
                    break
                print("Invalid selection. Please choose a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.resolved_file_path = str(selected)
        return self.resolved_file_path

    def load(self) -> pd.DataFrame:
        """Load a dataset from a CSV or similar supported tabular file."""
        resolved_path = self._resolve_dataset_path()
        df = pd.read_csv(resolved_path)
        return df

    def load_from_csv(self, file_path: str | None = None) -> pd.DataFrame:
        """Load a CSV dataset and return a DataFrame."""
        path = self._resolve_dataset_path(file_path)
        return pd.read_csv(path)

    def validate_columns(self, df: pd.DataFrame, required_columns: list[str]) -> bool:
        """Check whether the dataframe contains all required columns."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
