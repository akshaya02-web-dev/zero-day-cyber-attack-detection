"""Error analysis utilities.

This module will support confusion analysis, false-positive review, and
model-error analysis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class ErrorAnalyzer:
    """Gather information about model mistakes."""

    def __init__(self):
        pass

    def get_error_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe summary of prediction errors."""
        return df.copy()

    def identify_false_positives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows that were incorrectly predicted as attacks or anomalies."""
        return df.copy()

    def identify_false_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows that were missed by the model."""
        return df.copy()
