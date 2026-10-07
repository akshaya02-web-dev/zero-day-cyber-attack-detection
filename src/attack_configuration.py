"""Step 4C: configure known and unknown attack labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# These are configurable labels selected from the current master dataset.
# The remaining attack labels present in the dataset become the initial
# unknown configuration and can be edited for a specific experiment.
KNOWN_ATTACKS = ["DDoS", "PortScan", "DoS Hulk"]


def configure_attacks(master_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Read labels, validate configuration, and return display-ready lists."""
    dataframe = pd.read_csv(master_path)
    label_column = next(
        (column for column in dataframe.columns if str(column).strip().lower() == "label"),
        None,
    )
    if label_column is None:
        raise ValueError("ERROR: No Label column exists in the dataset.")

    actual_labels = {
        str(label).strip()
        for label in dataframe[label_column].dropna().unique()
    }
    if not actual_labels:
        raise ValueError("ERROR: The dataset contains no labels.")

    benign_labels = {label for label in actual_labels if label.lower() in {"benign", "normal"}}
    actual_attacks = actual_labels - benign_labels
    unknown_attacks = sorted(actual_attacks - set(KNOWN_ATTACKS))

    configured_known = list(KNOWN_ATTACKS)
    configured_unknown = unknown_attacks
    missing_known = [label for label in configured_known if label not in actual_labels]
    missing_unknown = [label for label in configured_unknown if label not in actual_labels]
    overlap = sorted(set(configured_known) & set(configured_unknown))
    invalid_known = [label for label in configured_known if label not in actual_attacks]
    invalid_unknown = [label for label in configured_unknown if label not in actual_attacks]

    for label in missing_known + missing_unknown:
        print(f"ERROR: Configured attack '{label}' does not exist in the dataset.")

    known_exists = not missing_known and not invalid_known
    unknown_exists = not missing_unknown and not invalid_unknown and "BENIGN" not in configured_unknown
    overlap_pass = not overlap
    if not known_exists or not unknown_exists or not overlap_pass:
        raise ValueError(
            "Attack configuration validation failed: "
            f"known_exists={known_exists}, unknown_exists={unknown_exists}, overlap={overlap}."
        )

    return configured_known, configured_unknown, sorted(actual_labels)


def format_configuration(known_attacks: list[str], unknown_attacks: list[str]) -> str:
    """Format the required final Step 4C block."""
    lines = [
        "=" * 61,
        "KNOWN / UNKNOWN ATTACK CONFIGURATION",
        "=" * 61,
        "Known Attacks:",
    ]
    lines.extend(f"- {label}" for label in known_attacks)
    lines.append("Unknown Attacks:")
    lines.extend(f"- {label}" for label in unknown_attacks)
    lines.extend([
        "Validation: Known labels exist: PASS",
        "Unknown labels exist: PASS",
        "Overlap check: PASS",
        "=" * 61,
    ])
    return "\n".join(lines)