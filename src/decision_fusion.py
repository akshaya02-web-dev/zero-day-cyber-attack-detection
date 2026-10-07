"""Decision fusion logic for final cyber threat classification.

The final output should be one of:
- NORMAL
- KNOWN_ATTACK
- POTENTIAL_UNKNOWN_THREAT
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class DecisionFusion:
    """Simple rule-based fusion placeholder for supervised and anomaly outputs."""

    def __init__(
        self,
        known_attack_threshold: float = 0.6,
        anomaly_threshold: float = 0.0,
    ):
        self.known_attack_threshold = known_attack_threshold
        self.anomaly_threshold = anomaly_threshold

    def fuse(self, supervised_probs: pd.DataFrame, anomaly_scores: pd.Series) -> pd.Series:
        """Combine supervised probabilities and anomaly scores into final decisions.

        TODO:
            - implement rule-based or configurable fusion engine
            - use thresholds without leakage
        """
        decisions = []

        for idx, row in supervised_probs.iterrows():
            max_prob = row.max()
            anomaly_score = anomaly_scores.loc[idx]

            if anomaly_score < self.anomaly_threshold and max_prob >= self.known_attack_threshold:
                decisions.append("KNOWN_ATTACK")
            elif anomaly_score >= self.anomaly_threshold and max_prob < self.known_attack_threshold:
                decisions.append("POTENTIAL_UNKNOWN_THREAT")
            else:
                decisions.append("NORMAL")

        return pd.Series(decisions, index=supervised_probs.index)
