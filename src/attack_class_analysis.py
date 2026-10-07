"""Step 4B: attack and traffic class analysis for the master dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_directory


RARE_PERCENTAGE_THRESHOLD = 1.0


def analyze_attack_classes(master_path: Path, output_path: Path) -> pd.DataFrame:
    """Calculate class counts and percentages without assigning unknown labels."""
    dataframe = pd.read_csv(master_path)
    label_column = next(
        (column for column in dataframe.columns if str(column).strip().lower() == "label"),
        None,
    )
    if label_column is None:
        raise ValueError("No Label column found in the master dataset.")

    total_samples = len(dataframe)
    counts = dataframe[label_column].value_counts(dropna=False)
    summary = pd.DataFrame({
        "Class": ["<MISSING>" if pd.isna(label) else str(label) for label in counts.index],
        "Samples": counts.astype(int).values,
    })
    summary["Percentage"] = (summary["Samples"] / total_samples * 100).round(2)
    summary["Traffic Type"] = summary["Class"].apply(
        lambda label: "BENIGN/normal traffic" if label.strip().lower() in {"benign", "normal"} else "Known attack class currently available"
    )
    summary["Rare (<1%)"] = summary["Percentage"] < RARE_PERCENTAGE_THRESHOLD

    ensure_directory(output_path.parent)
    summary.to_csv(output_path, index=False)
    return summary


def format_analysis(summary: pd.DataFrame) -> str:
    """Format the exact human-readable Step 4B analysis output."""
    lines = [
        "=" * 62,
        "MASTER DATASET – ATTACK DISTRIBUTION",
        "=" * 62,
        f"{'Class':<28}{'Samples':>12}{'Percentage':>15}",
        "-" * 55,
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"{row[0]:<28}{row[1]:>12}{row[2]:>14.2f}%")

    benign = summary[summary["Traffic Type"] == "BENIGN/normal traffic"]["Class"].tolist()
    attacks = summary[summary["Traffic Type"] != "BENIGN/normal traffic"]["Class"].tolist()
    rare = summary[summary["Rare (<1%)"]]["Class"].tolist()
    lines.extend([
        "=" * 62,
        "",
        "Available labels:",
        ", ".join(summary["Class"].tolist()),
        "",
        "BENIGN/normal traffic:",
        ", ".join(benign) if benign else "None",
        "",
        "Known attack classes currently available:",
        ", ".join(attacks) if attacks else "None",
        "",
        "Rare classes (<1% of total samples):",
        ", ".join(rare) if rare else "None",
        "",
        "STEP 4B STATUS: ATTACK CLASS ANALYSIS COMPLETED",
        "STOP HERE.",
    ])
    return "\n".join(lines)