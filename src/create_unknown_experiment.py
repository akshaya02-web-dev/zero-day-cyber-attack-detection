"""
src/create_unknown_experiment.py
=================================
Creates a held-out / open-set experimental split from the EXISTING
processed CIC-IDS2017 data (output/splits/).

Steps performed:
  1. Inspect label distributions for every dataset.
  2. Classify each dataset as SUITABLE or UNSUITABLE for the experiment.
  3. For suitable datasets, separate known vs. held-out attack classes.
  4. Write new files under data/unknown_experiment/<dataset>/.
  5. Perform an explicit no-leakage check.
  6. Print a concise final summary.

Rules:
  - Does NOT retrain or modify any existing models or results.
  - Does NOT re-run feature selection or normalization.
  - Does NOT merge datasets.
  - Does NOT fabricate samples.
  - random_state=42 wherever random splitting is required.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")


import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (all paths relative to the project root / cwd)
# ──────────────────────────────────────────────────────────────────────────────

SPLITS_ROOT    = "output/splits"
OUT_DATA_ROOT  = "data/unknown_experiment"
OUT_RESULT_ROOT = "output/unknown_experiment"

TARGET_COL = "Label"
RANDOM_STATE = 42

# ── Per-dataset held-out design ────────────────────────────────────────────────
# Derived from inspection of actual label distributions in output/splits/.
#
# KNOWN   = classes given to the supervised model (BENIGN + selected attacks)
# HELD_OUT = classes completely excluded from supervised training/validation
#
# Suitability reasoning:
#   Monday            → BENIGN only             → UNSUITABLE (no attack)
#   Friday DDos       → BENIGN + DDoS           → UNSUITABLE (only 1 attack)
#   Friday PortScan   → BENIGN + PortScan        → UNSUITABLE (only 1 attack)
#   Friday Morning    → BENIGN + Bot (1,367)     → UNSUITABLE (only 1 attack)
#   Thu Infiltration  → BENIGN + Infiltration(25)→ UNSUITABLE (25 samples)
#   Tuesday           → BENIGN + FTP-Patator + SSH-Patator → SUITABLE
#   Wednesday         → BENIGN + 4 DoS + Heartbleed(8)    → SUITABLE
#   Thu WebAttacks    → BENIGN + BruteForce + XSS + SqlInj(15) → SUITABLE

EXPERIMENT_CONFIG = {
    "Tuesday-WorkingHours.pcap_ISCX": {
        "known_attacks":   ["FTP-Patator"],
        "held_out_attacks": ["SSH-Patator"],
        "rationale": (
            "Two brute-force attack classes. FTP-Patator is treated as known; "
            "SSH-Patator is held out as an unknown attack for evaluation."
        ),
    },
    "Wednesday-workingHours.pcap_ISCX": {
        "known_attacks":   ["DoS Hulk", "DoS GoldenEye", "DoS Slowhttptest"],
        "held_out_attacks": ["DoS slowloris", "Heartbleed"],
        "rationale": (
            "Six classes total. Three DoS variants retained as known. "
            "DoS slowloris (3,769 train samples) and Heartbleed (8 train samples) "
            "held out as unknown. Heartbleed is intentionally included despite its "
            "tiny size — it is a real zero-day-style rare event."
        ),
    },
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX": {
        # Label contains special characters — match exactly as read from CSV
        "known_attacks":   [],   # resolved dynamically below
        "held_out_attacks": [], # resolved dynamically below
        "_dynamic": True,       # special flag: resolve at runtime from actual labels
        "rationale": (
            "Three web attack sub-types. Brute Force and XSS are kept as known. "
            "SQL Injection (15 train samples) held out as an extreme-rare unknown."
        ),
    },
}

# Datasets confirmed unsuitable — report only, create no files
UNSUITABLE = {
    "Monday-WorkingHours.pcap_ISCX":
        "Only BENIGN traffic — no attack class to split.",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX":
        "Only one attack class (DDoS) — cannot split known vs. held-out attacks.",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX":
        "Only one attack class (PortScan) — cannot split.",
    "Friday-WorkingHours-Morning.pcap_ISCX":
        "Only one attack class (Bot, 1,367 samples) — cannot split.",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX":
        "Only 25 Infiltration samples — too few for a meaningful held-out split.",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_split(ds_name, split_name):
    path = os.path.join(SPLITS_ROOT, ds_name, split_name, f"{ds_name}.csv")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return pd.read_csv(path)


def _safe_name(ds_name):
    """Strip .pcap_ISCX suffix for use in directory names."""
    return ds_name.replace(".pcap_ISCX", "").replace(" ", "_")


def _leakage_check(known_train, known_val, known_test, held_out_df, ds_name):
    """
    Confirm that held-out labels do not appear anywhere in known splits.
    Returns True if clean, False if leakage detected.
    """
    held_labels = set(held_out_df[TARGET_COL].unique())
    ktr_labels  = set(known_train[TARGET_COL].unique())
    kvl_labels  = set(known_val[TARGET_COL].unique())
    kte_labels  = set(known_test[TARGET_COL].unique())

    print(f"\n  LEAKAGE CHECK — {ds_name}")
    print(f"    KNOWN TRAIN LABELS      : {sorted(ktr_labels)}")
    print(f"    KNOWN VALIDATION LABELS : {sorted(kvl_labels)}")
    print(f"    KNOWN TEST LABELS       : {sorted(kte_labels)}")
    print(f"    HELD-OUT LABELS         : {sorted(held_labels)}")

    overlap_tr  = held_labels & ktr_labels
    overlap_vl  = held_labels & kvl_labels
    overlap_te  = held_labels & kte_labels
    all_overlap = overlap_tr | overlap_vl | overlap_te

    print(f"    OVERLAP (train)      : {sorted(overlap_tr)  or 'NONE'}")
    print(f"    OVERLAP (validation) : {sorted(overlap_vl) or 'NONE'}")
    print(f"    OVERLAP (test)       : {sorted(overlap_te)  or 'NONE'}")

    if all_overlap:
        print(f"\n  *** LEAKAGE DETECTED *** Overlapping classes: {sorted(all_overlap)}")
        print("  STOPPING for this dataset — do NOT use these files.")
        return False

    print("    DATA LEAKAGE CHECK: PASSED — overlap = 0")
    return True


def _save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _class_dist_report(df, split_label):
    vc = df[TARGET_COL].value_counts()
    rows = [{"Split": split_label, "Class": lbl, "Count": int(cnt)}
            for lbl, cnt in vc.items()]
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — INSPECTION REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_inspection_report():
    print("=" * 70)
    print("STEP 1 — DATASET INSPECTION REPORT")
    print("=" * 70)

    all_ds = sorted([
        d for d in os.listdir(SPLITS_ROOT)
        if os.path.isdir(os.path.join(SPLITS_ROOT, d))
        and d not in {"test", "train", "validation"}
    ])

    for ds in all_ds:
        df_tr = _load_split(ds, "train")
        df_vl = _load_split(ds, "validation")
        df_te = _load_split(ds, "test")
        if df_tr is None:
            continue

        total = len(df_tr) + (len(df_vl) if df_vl is not None else 0) + \
                (len(df_te) if df_te is not None else 0)

        all_df = pd.concat(
            [x for x in [df_tr, df_vl, df_te] if x is not None], ignore_index=True
        )
        vc = all_df[TARGET_COL].value_counts()

        suitable = ds in EXPERIMENT_CONFIG
        tag = "SUITABLE" if suitable else "UNSUITABLE"
        reason = "" if suitable else f"  → {UNSUITABLE.get(ds, 'See config')}"

        print(f"\n  Dataset : {ds}")
        print(f"  Status  : {tag}{reason}")
        print(f"  Features: {df_tr.shape[1] - 1}  |  Total samples: {total}")
        print(f"  {'Label':<40} {'Count':>8}")
        print(f"  {'-'*50}")
        for lbl, cnt in vc.items():
            print(f"  {str(lbl):<40} {cnt:>8}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# CORE — process one dataset
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset(ds_name, cfg):
    print("=" * 70)
    print(f"PROCESSING: {ds_name}")
    print("=" * 70)
    print(f"  Rationale: {cfg['rationale']}")

    # Load existing processed splits
    df_tr = _load_split(ds_name, "train")
    df_vl = _load_split(ds_name, "validation")
    df_te = _load_split(ds_name, "test")

    if df_tr is None or df_vl is None or df_te is None:
        print("  SKIPPING — missing split files.")
        return None

    # Resolve dynamic labels (WebAttacks special characters)
    if cfg.get("_dynamic"):
        all_labels = df_tr[TARGET_COL].unique().tolist()
        known_atk   = [l for l in all_labels if "Brute Force" in str(l) or "XSS" in str(l)]
        held_out_atk = [l for l in all_labels if "Sql Injection" in str(l)]
        cfg = dict(cfg)
        cfg["known_attacks"]    = known_atk
        cfg["held_out_attacks"] = held_out_atk

    known_attacks    = cfg["known_attacks"]
    held_out_attacks = cfg["held_out_attacks"]
    known_classes    = ["BENIGN"] + known_attacks

    print(f"  Known classes   : {known_classes}")
    print(f"  Held-out classes: {held_out_attacks}")

    # ── Verify classes exist in data ───────────────────────────────────────
    actual_labels = set(df_tr[TARGET_COL].unique())
    for lbl in known_attacks + held_out_attacks:
        if lbl not in actual_labels:
            print(f"  WARNING: '{lbl}' not found in training data labels: {actual_labels}")

    # ── Separate known vs. held-out from ALL splits ──────────────────────
    def filter_known(df):
        return df[df[TARGET_COL].isin(known_classes)].copy().reset_index(drop=True)

    def filter_held_out(df):
        return df[df[TARGET_COL].isin(held_out_attacks)].copy().reset_index(drop=True)

    known_train = filter_known(df_tr)
    known_val   = filter_known(df_vl)
    known_test  = filter_known(df_te)

    # Held-out comes from ALL splits combined (maximum evaluation data)
    held_out_all = pd.concat(
        [filter_held_out(df_tr), filter_held_out(df_vl), filter_held_out(df_te)],
        ignore_index=True
    )

    # ── Leakage check ─────────────────────────────────────────────────────
    if not _leakage_check(known_train, known_val, known_test, held_out_all, ds_name):
        return None

    # ── Save files ─────────────────────────────────────────────────────────
    safe = _safe_name(ds_name)
    base_data   = os.path.join(OUT_DATA_ROOT, safe)
    base_result = os.path.join(OUT_RESULT_ROOT, safe)

    _save_csv(known_train, os.path.join(base_data, "known", "train.csv"))
    _save_csv(known_val,   os.path.join(base_data, "known", "validation.csv"))
    _save_csv(known_test,  os.path.join(base_data, "known", "test.csv"))
    _save_csv(held_out_all, os.path.join(base_data, "held_out", "unknown_test.csv"))

    # ── Class distribution report ─────────────────────────────────────────
    dist_rows = []
    dist_rows += _class_dist_report(known_train, "known_train")
    dist_rows += _class_dist_report(known_val,   "known_validation")
    dist_rows += _class_dist_report(known_test,  "known_test")
    dist_rows += _class_dist_report(held_out_all,"held_out_unknown_test")
    dist_df = pd.DataFrame(dist_rows)

    os.makedirs(base_result, exist_ok=True)
    dist_df.to_csv(os.path.join(base_result, "class_distribution.csv"), index=False)

    # Known vs held-out split summary
    known_dist_df = dist_df[dist_df["Split"].str.startswith("known")].copy()
    held_dist_df  = dist_df[dist_df["Split"] == "held_out_unknown_test"].copy()
    known_dist_df.to_csv(os.path.join(base_result, "known_class_distribution.csv"), index=False)
    held_dist_df.to_csv(os.path.join(base_result, "held_out_class_distribution.csv"), index=False)

    # split_summary.csv
    summary = {
        "Dataset": ds_name,
        "Known Classes": str(known_classes),
        "Held-Out Classes": str(held_out_attacks),
        "Known Train Samples": len(known_train),
        "Known Validation Samples": len(known_val),
        "Known Test Samples": len(known_test),
        "Held-Out Unknown Samples": len(held_out_all),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(base_result, "split_summary.csv"), index=False)

    # leakage_check.txt
    leakage_text = (
        f"Dataset: {ds_name}\n"
        f"Known Classes: {known_classes}\n"
        f"Held-Out Classes: {held_out_attacks}\n"
        f"Known Train Labels: {sorted(known_train[TARGET_COL].unique().tolist())}\n"
        f"Known Validation Labels: {sorted(known_val[TARGET_COL].unique().tolist())}\n"
        f"Known Test Labels: {sorted(known_test[TARGET_COL].unique().tolist())}\n"
        f"Held-Out Labels: {sorted(held_out_all[TARGET_COL].unique().tolist())}\n"
        f"Overlap: NONE — DATA LEAKAGE CHECK: PASSED\n"
    )
    with open(os.path.join(base_result, "leakage_check.txt"), "w", encoding="utf-8") as f:
        f.write(leakage_text)

    # ── Class distribution terminal printout ───────────────────────────────
    print(f"\n  CLASS DISTRIBUTION — {ds_name}")
    print(f"\n  KNOWN CLASSES:")
    kn_vc = known_train[TARGET_COL].value_counts()
    for lbl, cnt in kn_vc.items():
        print(f"    {str(lbl):<40} {cnt:>8}  (train)")
    print(f"\n  HELD-OUT CLASSES:")
    ho_vc = held_out_all[TARGET_COL].value_counts()
    for lbl, cnt in ho_vc.items():
        print(f"    {str(lbl):<40} {cnt:>8}  (all splits combined)")

    print(f"\n  Known Training Samples   : {len(known_train)}")
    print(f"  Known Validation Samples : {len(known_val)}")
    print(f"  Known Test Samples       : {len(known_test)}")
    print(f"  Held-Out Unknown Samples : {len(held_out_all)}")

    print(f"\n  Files saved:")
    print(f"    {base_data}/known/train.csv")
    print(f"    {base_data}/known/validation.csv")
    print(f"    {base_data}/known/test.csv")
    print(f"    {base_data}/held_out/unknown_test.csv")
    print(f"    {base_result}/split_summary.csv")
    print(f"    {base_result}/class_distribution.csv")
    print(f"    {base_result}/known_class_distribution.csv")
    print(f"    {base_result}/held_out_class_distribution.csv")
    print(f"    {base_result}/leakage_check.txt")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Step 1 — Inspection
    print_inspection_report()

    # Step 2–7 — Process suitable datasets
    summaries = []
    for ds_name, cfg in EXPERIMENT_CONFIG.items():
        result = process_dataset(ds_name, cfg)
        if result:
            summaries.append(result)
        print()

    # Unsuitable datasets
    print("=" * 70)
    print("UNSUITABLE DATASETS (no experiment created)")
    print("=" * 70)
    for ds, reason in UNSUITABLE.items():
        print(f"  {ds}")
        print(f"    → {reason}")
    print()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  1. Existing baseline (Steps 1–6) preserved: YES")
    print(f"     → output/models/  and  output/results/  are UNTOUCHED")
    print()

    for i, s in enumerate(summaries, start=2):
        print(f"  Dataset : {s['Dataset']}")
        print(f"    Known classes    : {s['Known Classes']}")
        print(f"    Held-out classes : {s['Held-Out Classes']}")
        print(f"    Known train      : {s['Known Train Samples']}")
        print(f"    Known validation : {s['Known Validation Samples']}")
        print(f"    Known test       : {s['Known Test Samples']}")
        print(f"    Held-out unknown : {s['Held-Out Unknown Samples']}")
        print(f"    Leakage check    : PASSED")
        print()

    print("  New files location:")
    print(f"    data/unknown_experiment/")
    print(f"    output/unknown_experiment/")
    print()
    print("  Step 7 (Supervised Known-Attack Training): IMPLEMENTED (src/train_unknown_experiment.py)")
    print("  Step 8 (DBSCAN Unknown Attack Detection):  IMPLEMENTED (src/clustering_pipeline.py)")
    print()
    print("=" * 70)
    print("SCRIPT COMPLETED — experiment split created successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
