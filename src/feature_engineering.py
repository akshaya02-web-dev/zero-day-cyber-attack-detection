"""Feature extraction utilities for CIC-IDS2017 flow records."""

from __future__ import annotations

import re

import pandas as pd


class FeatureEngineer:
    """Create safe, flow-level features while preserving the label column."""

    def __init__(self):
        self.created_features: list[str] = []

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Divide safely and return zero for invalid or zero-denominator rows."""
        denominator = denominator.replace(0, pd.NA)
        result = numerator.div(denominator)
        return result.replace([float("inf"), float("-inf")], 0).fillna(0.0)

    @staticmethod
    def _column_lookup(columns: pd.Index) -> dict[str, str]:
        """Match raw CIC headers and normalized pipeline headers alike."""
        return {
            re.sub(r"[^a-z0-9]", "", str(column).lower()): column
            for column in columns
        }

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add CIC-IDS2017 packet, byte, rate, and ratio features."""
        engineered = df.copy()

        lookup = self._column_lookup(engineered.columns)
        required_columns = {
            "Total Fwd Packets": "totalfwdpackets",
            "Total Backward Packets": "totalbackwardpackets",
            "Total Length of Fwd Packets": "totallengthoffwdpackets",
            "Total Length of Bwd Packets": "totallengthofbwdpackets",
            "Flow Duration": "flowduration",
        }
        missing_columns = [
            display_name for display_name, normalized_name in required_columns.items()
            if normalized_name not in lookup
        ]
        if missing_columns:
            raise ValueError(f"Missing CIC flow columns for feature extraction: {missing_columns}")

        numeric = engineered.apply(pd.to_numeric, errors="coerce").replace(
            [float("inf"), float("-inf")], pd.NA
        )
        forward_packets = numeric[lookup[required_columns["Total Fwd Packets"]]].fillna(0.0)
        backward_packets = numeric[lookup[required_columns["Total Backward Packets"]]].fillna(0.0)
        forward_bytes = numeric[lookup[required_columns["Total Length of Fwd Packets"]]].fillna(0.0)
        backward_bytes = numeric[lookup[required_columns["Total Length of Bwd Packets"]]].fillna(0.0)
        total_packets = forward_packets + backward_packets
        total_bytes = forward_bytes + backward_bytes

        derived = {
            "Total Packets": total_packets,
            "Total Bytes": total_bytes,
            "Fwd/Bwd Packet Ratio": self._safe_divide(forward_packets, backward_packets),
            "Fwd/Bwd Byte Ratio": self._safe_divide(forward_bytes, backward_bytes),
            "Derived Packet Rate": self._safe_divide(total_packets * 1_000_000, numeric[lookup[required_columns["Flow Duration"]]]),
            "Derived Byte Rate": self._safe_divide(total_bytes * 1_000_000, numeric[lookup[required_columns["Flow Duration"]]]),
            "Average Packet Size": self._safe_divide(total_bytes, total_packets),
        }
        for name, values in derived.items():
            engineered[name] = values.replace([float("inf"), float("-inf")], 0).fillna(0.0)
            if name not in self.created_features:
                self.created_features.append(name)
        return engineered

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering operations to the dataset."""
        return self.add_basic_features(df)
