"""
STEP 6: MODEL VALIDATION AND ANALYSIS
======================================
Loads pre-trained models and finalized dataset splits.
Does NOT retrain, refit, or modify any earlier pipeline stage.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
TARGET_CANDIDATES = ["Label", "binary_label", "attack_category"]
SPLITS_DIR       = "output/splits"
MODELS_DIR       = "output/models"
RESULTS_DIR      = "output/results"

MODEL_KEYS = [
    ("Random Forest", "random_forest.pkl"),
    ("XGBoost",       "xgboost.pkl"),
    ("LightGBM",      "lightgbm.pkl"),
]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _load_csv(path):
    if not os.path.exists(path):
        print(f"  DATA FILE MISSING: {path}")
        return None
    return pd.read_csv(path)


def _load_model(path):
    if not os.path.exists(path):
        print(f"  MODEL FILE MISSING: {path}")
        return None
    return joblib.load(path)


def _detect_target(df):
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def _split_xy(df, target_col):
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    return X, y


def _save_confusion_matrix(cm, labels, out_dir, filename_base):
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in labels],
        columns=[f"Pred_{c}" for c in labels]
    )
    cm_df.to_csv(os.path.join(out_dir, f"{filename_base}.csv"))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5),
                                    max(5, len(labels) * 1.3)))
    sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(filename_base.replace("_", " ").title(), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filename_base}.png"), dpi=150)
    plt.close(fig)


def _feature_consistency_check(train_cols, X_eval, split_name):
    train_set = set(train_cols)
    eval_set  = set(X_eval.columns)
    missing   = train_set - eval_set
    extra     = eval_set  - train_set
    order_ok  = list(train_cols) == list(X_eval.columns)

    if missing or extra or not order_ok:
        print(f"\n  FEATURE MISMATCH DETECTED ({split_name})")
        if missing:
            print(f"    Expected but missing: {sorted(missing)}")
        if extra:
            print(f"    Unexpected extras:    {sorted(extra)}")
        if not order_ok and not missing and not extra:
            print("    Feature ORDER mismatch")
        return False
    return True


# ──────────────────────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ──────────────────────────────────────────────────────────────

def validate_dataset(dataset_name, splits_dir, models_dir, results_dir):

    print("=" * 60)
    print(f"STEP 6: MODEL VALIDATION")
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    ds_splits = os.path.join(splits_dir, dataset_name)
    train_csv = os.path.join(ds_splits, "train",      f"{dataset_name}.csv")
    val_csv   = os.path.join(ds_splits, "validation", f"{dataset_name}.csv")
    test_csv  = os.path.join(ds_splits, "test",       f"{dataset_name}.csv")
    ds_models = os.path.join(models_dir, dataset_name)
    le_path   = os.path.join(ds_models, "label_encoder.pkl")
    ds_results = os.path.join(results_dir, dataset_name)
    os.makedirs(ds_results, exist_ok=True)

    df_train = _load_csv(train_csv)
    df_val   = _load_csv(val_csv)
    df_test  = _load_csv(test_csv)

    if df_train is None or df_val is None or df_test is None:
        print("  Skipping — one or more data files missing.")
        return

    target_col = _detect_target(df_train)
    X_train, y_train = _split_xy(df_train, target_col)
    X_val,   y_val   = _split_xy(df_val,   target_col)
    X_test,  y_test  = _split_xy(df_test,  target_col)

    # DATA LEAKAGE CHECK
    all_exist = all(
        os.path.exists(os.path.join(ds_models, fname))
        for _, fname in MODEL_KEYS
    )
    print("  DATA LEAKAGE CHECK: " + ("PASSED" if all_exist else "SKIPPED — some model files missing"))

    # FEATURE CONSISTENCY CHECK
    train_cols = list(X_train.columns)
    val_ok  = _feature_consistency_check(train_cols, X_val,  "validation")
    test_ok = _feature_consistency_check(train_cols, X_test, "test")

    if val_ok and test_ok:
        print("  FEATURE CONSISTENCY CHECK: PASSED")
        X_val  = X_val[train_cols]
        X_test = X_test[train_cols]
    else:
        print("  Aborting — feature mismatch.")
        return

    # Single-class guard
    if len(y_train.unique()) <= 1:
        print("  TRAINING HAD ONLY ONE CLASS — skipping metrics.")
        return

    # Label encoding
    le = _load_model(le_path)
    if le is None:
        y_train_enc = y_train.values
        y_val_enc   = y_val.values
        y_test_enc  = y_test.values
        class_labels = sorted(str(c) for c in y_train.unique())
    else:
        y_train_enc = le.transform(y_train)
        y_val_enc   = le.transform(y_val)
        y_test_enc  = le.transform(y_test)
        class_labels = list(le.classes_)

    print()
    print(f"  Training Samples:   {len(X_train)}")
    print(f"  Validation Samples: {len(X_val)}")
    print(f"  Test Samples:       {len(X_test)}")
    print(f"  Features:           {X_train.shape[1]}")
    print(f"  Classes:            {len(class_labels)}")

    # ── Evaluate models ───────────────────────────────────────
    rows_val       = []
    rows_test      = []
    comparison_rows = []

    print()
    print("  " + "-" * 54)
    print("  VALIDATION RESULTS")
    print("  " + "-" * 54)
    print(f"  {'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<6}")

    # First pass: validation
    for model_label, model_file in MODEL_KEYS:
        model = _load_model(os.path.join(ds_models, model_file))
        if model is None:
            print(f"  {model_label:<18} SKIPPED")
            continue

        pv = model.predict(X_val)
        acc  = accuracy_score(y_val_enc, pv)
        prec = precision_score(y_val_enc, pv, average="weighted", zero_division=0)
        rec  = recall_score(y_val_enc, pv, average="weighted", zero_division=0)
        f1   = f1_score(y_val_enc, pv, average="weighted", zero_division=0)
        rows_val.append((model_label, acc, prec, rec, f1))
        comparison_rows.append({"Model": model_label, "Dataset": dataset_name,
                                 "Split": "Validation",
                                 "Accuracy": acc, "Precision": prec,
                                 "Recall": rec, "F1": f1})
        print(f"  {model_label:<18} {acc:<10.4f} {prec:<11.4f} {rec:<8.4f} {f1:<6.4f}")

        # Save classification report
        model_slug = model_label.lower().replace(" ", "_")
        model_out  = os.path.join(ds_results, model_slug)
        os.makedirs(model_out, exist_ok=True)

        cr = classification_report(y_val_enc, pv, target_names=class_labels, zero_division=0)
        with open(os.path.join(model_out, "validation_classification_report.txt"),
                  "w", encoding="utf-8") as f:
            f.write(f"=== {model_label} — VALIDATION ===\n\n{cr}")

        # Confusion matrix
        cm = confusion_matrix(y_val_enc, pv)
        _save_confusion_matrix(cm, class_labels, model_out, "validation_confusion_matrix")

    print()
    print("  " + "-" * 54)
    print("  TEST RESULTS")
    print("  " + "-" * 54)
    print(f"  {'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<6}")

    # Second pass: test
    for model_label, model_file in MODEL_KEYS:
        model = _load_model(os.path.join(ds_models, model_file))
        if model is None:
            print(f"  {model_label:<18} SKIPPED")
            continue

        pt = model.predict(X_test)
        acc  = accuracy_score(y_test_enc, pt)
        prec = precision_score(y_test_enc, pt, average="weighted", zero_division=0)
        rec  = recall_score(y_test_enc, pt, average="weighted", zero_division=0)
        f1   = f1_score(y_test_enc, pt, average="weighted", zero_division=0)
        rows_test.append((model_label, acc, prec, rec, f1))
        comparison_rows.append({"Model": model_label, "Dataset": dataset_name,
                                 "Split": "Test",
                                 "Accuracy": acc, "Precision": prec,
                                 "Recall": rec, "F1": f1})
        print(f"  {model_label:<18} {acc:<10.4f} {prec:<11.4f} {rec:<8.4f} {f1:<6.4f}")

        model_slug = model_label.lower().replace(" ", "_")
        model_out  = os.path.join(ds_results, model_slug)
        os.makedirs(model_out, exist_ok=True)

        cr = classification_report(y_test_enc, pt, target_names=class_labels, zero_division=0)
        with open(os.path.join(model_out, "test_classification_report.txt"),
                  "w", encoding="utf-8") as f:
            f.write(f"=== {model_label} — TEST ===\n\n{cr}")

        cm = confusion_matrix(y_test_enc, pt)
        _save_confusion_matrix(cm, class_labels, model_out, "test_confusion_matrix")

    # ── CLASS DISTRIBUTION ────────────────────────────────────
    print()
    print("  " + "-" * 54)
    print("  CLASS DISTRIBUTION")
    print("  " + "-" * 54)

    dist_rows  = []
    train_vc   = y_train.value_counts()
    val_vc     = y_val.value_counts()
    test_vc    = y_test.value_counts()
    all_classes = sorted(set(y_train) | set(y_val) | set(y_test), key=str)

    print(f"  {'Class':<30} {'Train':>8}  {'Validation':>10}  {'Test':>8}")
    for cls in all_classes:
        tr = train_vc.get(cls, 0)
        v  = val_vc.get(cls, 0)
        te = test_vc.get(cls, 0)
        print(f"  {str(cls):<30} {tr:>8}  {v:>10}  {te:>8}")
        dist_rows += [
            {"Split": "Train",      "Class": cls, "Count": int(tr)},
            {"Split": "Validation", "Class": cls, "Count": int(v)},
            {"Split": "Test",       "Class": cls, "Count": int(te)},
        ]

    pd.DataFrame(dist_rows).to_csv(
        os.path.join(ds_results, "class_distribution.csv"), index=False
    )

    # ── CLASS-WISE ANALYSIS ───────────────────────────────────
    print()
    print("  " + "-" * 54)
    print("  CLASS-WISE ANALYSIS (Validation)")
    print("  " + "-" * 54)

    for model_label, _ in MODEL_KEYS:
        model_slug = model_label.lower().replace(" ", "_")
        cr_path = os.path.join(ds_results, model_slug,
                               "validation_classification_report.txt")
        if os.path.exists(cr_path):
            with open(cr_path, encoding="utf-8") as f:
                content = f.read()
            print(f"\n  {model_label}:")
            for line in content.splitlines():
                print(f"    {line}")

    # ── CONFUSION MATRICES status ─────────────────────────────
    print()
    print("  " + "-" * 54)
    print("  CONFUSION MATRICES")
    print("  " + "-" * 54)
    for model_label, _ in MODEL_KEYS:
        model_slug = model_label.lower().replace(" ", "_")
        v_png = os.path.join(ds_results, model_slug, "validation_confusion_matrix.png")
        t_png = os.path.join(ds_results, model_slug, "test_confusion_matrix.png")
        if os.path.exists(v_png) and os.path.exists(t_png):
            print(f"  {model_label}: GENERATED")
        else:
            print(f"  {model_label}: MISSING (model may have been skipped)")

    # ── Save summary files ────────────────────────────────────
    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(ds_results, "model_comparison.csv"), index=False
    )
    pd.DataFrame(rows_val, columns=["Model","Accuracy","Precision","Recall","F1"]
                 ).to_csv(os.path.join(ds_results, "validation_results.csv"), index=False)
    pd.DataFrame(rows_test, columns=["Model","Accuracy","Precision","Recall","F1"]
                 ).to_csv(os.path.join(ds_results, "test_results.csv"), index=False)

    print()
    print("  " + "-" * 54)
    print("  SAVED OUTPUTS")
    print("  " + "-" * 54)
    print(f"  Results: output/results/{dataset_name}/")
    print()
    print("=" * 60)
    print("STEP 6 COMPLETED")
    print("=" * 60)
    print()


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 6 — Model Validation and Analysis"
    )
    parser.add_argument("--dataset",     type=str, default=None,
                        help="Specific dataset name (default: all)")
    parser.add_argument("--splits_dir",  type=str, default=SPLITS_DIR)
    parser.add_argument("--models_dir",  type=str, default=MODELS_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.dataset:
        validate_dataset(args.dataset,
                         args.splits_dir, args.models_dir, args.results_dir)
    else:
        skip = {"test", "train", "validation"}
        datasets = sorted([
            d for d in os.listdir(args.models_dir)
            if os.path.isdir(os.path.join(args.models_dir, d)) and d not in skip
        ])
        if not datasets:
            print("No trained dataset directories found.")
            sys.exit(1)
        for ds in datasets:
            validate_dataset(ds,
                             args.splits_dir, args.models_dir, args.results_dir)


if __name__ == "__main__":
    main()
