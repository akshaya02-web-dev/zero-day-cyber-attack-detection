"""
src/clustering_pipeline.py
===========================
STEP 8: DBSCAN DENSITY-BASED CLUSTERING & HELD-OUT UNKNOWN ATTACK DETECTION

Clean, modular, and professional implementation of DBSCAN clustering for
discovering unknown zero-day cyber threats.

Key Design:
  1. Identifies 5 primary dense clusters per dataset representing known traffic profiles.
  2. Samples falling outside all dense core clusters (distance > eps) are classified as
     DBSCAN NOISE (Cluster ID = -1), which indicates an UNKNOWN ZERO-DAY THREAT.
  3. Outputs are organized into cleanly numbered, human-readable files under output/unknown/.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASETS_CONFIG = {
    'Tuesday-WorkingHours': {
        'folder_name': 'Tuesday_SSH_Patator',
        'display_name': 'Tuesday-WorkingHours',
        'attack_name': 'SSH-Patator (Brute Force)',
        'held_out_classes': ['SSH-Patator'],
        'eps': 0.50,
        'min_samples': 10,
    },
    'Wednesday-workingHours': {
        'folder_name': 'Wednesday_Slowloris_Heartbleed',
        'display_name': 'Wednesday-workingHours',
        'attack_name': 'DoS slowloris + Heartbleed',
        'held_out_classes': ['DoS slowloris', 'Heartbleed'],
        'eps': 0.32,
        'min_samples': 15,
    },
    'Thursday-WorkingHours-Morning-WebAttacks': {
        'folder_name': 'Thursday_SQL_Injection',
        'display_name': 'Thursday-Morning-WebAttacks',
        'attack_name': 'Web Attack - SQL Injection',
        'held_out_classes': ['Web Attack \ufffd Sql Injection'],
        'clean_held_out_names': {'Web Attack \ufffd Sql Injection': 'Web Attack - SQL Injection'},
        'eps': 0.22,
        'min_samples': 10,
    }
}

DATA_ROOT = Path("data/unknown_experiment")
OUTPUT_UNKNOWN_ROOT = Path("output/unknown")
N_CLUSTERS = 5
RANDOM_STATE = 42


def clean_label_string(label: str) -> str:
    """Replace broken Unicode replacement characters with clean human-readable text."""
    if not isinstance(label, str):
        return str(label)
    return (label.replace("\ufffd", "-")
                 .replace("Web Attack - Sql Injection", "Web Attack - SQL Injection")
                 .replace("Web Attack - Brute Force", "Web Attack - Brute Force")
                 .replace("Web Attack - XSS", "Web Attack - XSS"))


# ==============================================================================
# DBSCAN DETECTOR CLASS
# ==============================================================================
class DBSCANUnknownAttackDetector:
    """DBSCAN Density-Based Detector using ALL discovered clusters for classification,
    while reporting the N_CLUSTERS largest as primary behavioral profiles."""

    def __init__(self, eps: float = 0.5, min_samples: int = 10, n_primary_clusters: int = 5, random_state: int = 42):
        self.eps = eps
        self.min_samples = min_samples
        self.n_primary_clusters = n_primary_clusters
        self.random_state = random_state

        self.dbscan: DBSCAN | None = None
        self.nn_model: NearestNeighbors | None = None
        self.core_points: np.ndarray | None = None
        self.core_labels: np.ndarray | None = None
        self.total_clusters_found: int = 0
        self.primary_cluster_ids: list[int] = []
        self.cluster_profiles: dict[int, dict] = {}
        self.cluster_centers: dict[int, np.ndarray] = {}
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit DBSCAN on representative training data.
        
        Uses ALL discovered clusters for classification (to avoid false alarms
        from discarding small but valid density regions), while reporting the
        N largest clusters as primary behavioral profiles.
        """
        self.feature_names = list(X_train.columns)
        X_mat = X_train.values

        # Subsample for core discovery if dataset is very large
        max_fit_samples = 30000
        if len(X_mat) > max_fit_samples:
            np.random.seed(self.random_state)
            fit_idx = np.random.choice(len(X_mat), max_fit_samples, replace=False)
            X_fit = X_mat[fit_idx]
            y_fit = y_train.iloc[fit_idx].values
        else:
            X_fit = X_mat
            y_fit = y_train.values

        # 1. Fit DBSCAN
        self.dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",
            n_jobs=-1
        )
        labels = self.dbscan.fit_predict(X_fit)

        # 2. Extract ALL Core Points (not just top-5)
        core_mask = np.zeros(len(X_fit), dtype=bool)
        core_mask[self.dbscan.core_sample_indices_] = True
        raw_core_points = X_fit[core_mask]
        raw_core_labels = labels[core_mask]
        core_y = y_fit[core_mask]

        # 3. Keep ALL valid clusters for classification
        valid_mask = raw_core_labels >= 0
        if valid_mask.sum() == 0:
            raise ValueError(f"DBSCAN found 0 clusters with eps={self.eps}. Adjust eps.")

        self.core_points = raw_core_points[valid_mask]
        self.core_labels = raw_core_labels[valid_mask]
        core_y_valid = core_y[valid_mask]

        unique_clusters, counts = np.unique(self.core_labels, return_counts=True)
        self.total_clusters_found = len(unique_clusters)

        # Identify the N largest as "primary" for reporting
        sorted_indices = np.argsort(-counts)
        n_primary = min(self.n_primary_clusters, len(unique_clusters))
        self.primary_cluster_ids = unique_clusters[sorted_indices[:n_primary]].tolist()

        # 4. Fit NearestNeighbors on ALL Core Points for Fast Inference
        self.nn_model = NearestNeighbors(
            n_neighbors=1,
            algorithm="kd_tree",
            metric="euclidean",
            n_jobs=-1
        ).fit(self.core_points)

        # 5. Compute Cluster Profiles for ALL clusters
        self.cluster_profiles = {}
        self.cluster_centers = {}

        for c_id in unique_clusters:
            c_mask = (self.core_labels == c_id)
            c_pts = self.core_points[c_mask]
            c_y = core_y_valid[c_mask]

            center = np.mean(c_pts, axis=0)
            self.cluster_centers[int(c_id)] = center
            y_counts = pd.Series(c_y).value_counts().to_dict()
            clean_y_counts = {clean_label_string(k): int(v) for k, v in y_counts.items()}
            dom_label = clean_label_string(max(y_counts, key=y_counts.get)) if y_counts else "BENIGN"
            dists_to_center = np.linalg.norm(c_pts - center, axis=1)

            self.cluster_profiles[int(c_id)] = {
                "cluster_id": int(c_id),
                "cluster_name": f"Cluster_{c_id}",
                "core_sample_count": int(np.sum(c_mask)),
                "dominant_traffic": dom_label,
                "label_distribution": clean_y_counts,
                "radius_eps": float(self.eps),
                "mean_dispersion": round(float(np.mean(dists_to_center)), 4),
                "is_primary": (int(c_id) in self.primary_cluster_ids),
            }

        return self

    def predict_samples(self, X: pd.DataFrame, sample_prefix: str = "sample") -> pd.DataFrame:
        """Assign samples to ANY discovered DBSCAN cluster or flag as NOISE / UNKNOWN (-1).
        
        Uses ALL clusters (not just the 5 primary ones) to ensure small but valid
        density regions are not incorrectly flagged as unknown attacks.
        """
        if self.nn_model is None or self.core_labels is None:
            raise ValueError("DBSCAN detector must be fitted before predict_samples.")

        X_mat = X.values
        dists, indices = self.nn_model.kneighbors(X_mat)
        dists = dists.flatten()
        indices = indices.flatten()

        predictions = []
        for i in range(len(X_mat)):
            d = float(dists[i])
            if d <= self.eps:
                assigned_cluster = int(self.core_labels[indices[i]])
                prof = self.cluster_profiles.get(assigned_cluster, {})
                dom_class = prof.get("dominant_traffic", "KNOWN")
                is_unknown = False
                pred_label = "KNOWN"
                score = round(min(0.49, (d / self.eps) * 0.49), 4)
            else:
                # Density unreachable -> Noise (-1) -> Unknown Attack
                assigned_cluster = -1
                dom_class = "NOISE_OUTLIER"
                is_unknown = True
                pred_label = "UNKNOWN_ATTACK"
                score = round(min(1.0, 0.50 + 0.50 * (d - self.eps) / (self.eps * 2.0)), 4)

            predictions.append({
                "sample_id": f"{sample_prefix}_{i}",
                "assigned_cluster": assigned_cluster,
                "cluster_dominant_class": dom_class,
                "distance_to_nearest_core": round(d, 5),
                "density_radius_eps": round(self.eps, 4),
                "anomaly_score": score,
                "predicted_status": pred_label,
                "is_unknown_attack": is_unknown,
            })

        return pd.DataFrame(predictions)


# ==============================================================================
# PIPELINE EXECUTION FOR ONE DATASET
# ==============================================================================
def run_dataset_pipeline(dataset_key: str, cfg: dict) -> dict:
    folder_name = cfg['folder_name']
    display_name = cfg['display_name']
    attack_name = cfg['attack_name']
    eps = cfg['eps']
    min_samples = cfg['min_samples']

    print("=" * 88)
    print(f"[{display_name}] -> TARGET ZERO-DAY ATTACK: {attack_name}")
    print("=" * 88)

    ds_dir = DATA_ROOT / dataset_key
    train_file = ds_dir / "known" / "train.csv"
    test_file = ds_dir / "known" / "test.csv"
    held_out_file = ds_dir / "held_out" / "unknown_test.csv"

    for f in [train_file, test_file, held_out_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required dataset split: {f}")

    # Create dataset-specific folder under output/unknown/
    ds_out_dir = OUTPUT_UNKNOWN_ROOT / folder_name
    ds_out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_UNKNOWN_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df_held = pd.read_csv(held_out_file)

    feature_cols = [c for c in df_train.columns if c != "Label"]
    X_train = df_train[feature_cols]
    y_train = df_train["Label"]

    X_test = df_test[feature_cols]
    y_test = df_test["Label"]

    X_held = df_held[feature_cols]
    y_held = df_held["Label"]

    print(f"  Configuration : eps = {eps}, min_samples = {min_samples}, primary_clusters = 5")
    print(f"  Training Data : {len(X_train):,} samples across {len(feature_cols)} features")
    print(f"  Known Test    : {len(X_test):,} samples")
    print(f"  Held-Out Test : {len(X_held):,} zero-day attack samples\n")

    # 2. Fit DBSCAN Detector
    print(f"  [1/4] Fitting DBSCAN on training data...")
    detector = DBSCANUnknownAttackDetector(
        eps=eps,
        min_samples=min_samples,
        n_primary_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE
    )
    detector.fit(X_train, y_train)
    print(f"        Total clusters discovered by DBSCAN: {detector.total_clusters_found}")
    print(f"        Primary clusters reported: {len(detector.primary_cluster_ids)}")
    print(f"        All {detector.total_clusters_found} clusters used for classification (no discarding)")

    # 3. Display Primary Clusters (report the 5 largest; all are used for prediction)
    print(f"  [2/4] The {len(detector.primary_cluster_ids)} Primary Baseline Clusters (of {detector.total_clusters_found} total):")
    cluster_rows = []
    for c_id in detector.primary_cluster_ids:
        prof = detector.cluster_profiles[c_id]
        print(f"        * Cluster {c_id}: {prof['core_sample_count']:>6,} core points  -->  "
              f"Dominant: {prof['dominant_traffic']:<18} (dispersion: {prof['mean_dispersion']:.4f})")
    # Build cluster overview for ALL clusters
    for c_id, prof in detector.cluster_profiles.items():
        cluster_rows.append({
            "Cluster_ID": c_id,
            "Cluster_Name": f"Cluster_{c_id}",
            "Core_Sample_Count": prof["core_sample_count"],
            "Dominant_Traffic": prof["dominant_traffic"],
            "Density_Radius_Eps": prof["radius_eps"],
            "Mean_Dispersion": prof["mean_dispersion"],
            "Is_Primary": prof.get("is_primary", False),
        })

    # Save 1_CLUSTERS_OVERVIEW.csv
    df_clusters_summary = pd.DataFrame(cluster_rows)
    df_clusters_summary.to_csv(ds_out_dir / "1_CLUSTERS_OVERVIEW.csv", index=False)

    # 4. Predict on Test Set (FPR) and Held-Out (Zero-Day Detection)
    print("\n  [3/4] Evaluating density boundary on known test set & zero-day attacks...")
    pred_test = detector.predict_samples(X_test, sample_prefix="test")
    pred_test["true_label"] = [clean_label_string(l) for l in y_test.values]

    pred_held = detector.predict_samples(X_held, sample_prefix="heldout")
    pred_held["true_label"] = [clean_label_string(l) for l in y_held.values]

    known_false_alarm_rate = float(np.mean(pred_test["is_unknown_attack"])) * 100.0
    known_inlier_rate = float(np.mean(~pred_test["is_unknown_attack"])) * 100.0

    # Separate benign vs attack FPR on known test set
    benign_mask = y_test.astype(str).str.upper() == "BENIGN"
    attack_mask = ~benign_mask
    benign_false_alarm = float(np.mean(pred_test.loc[benign_mask, "is_unknown_attack"])) * 100.0 if benign_mask.sum() > 0 else 0.0
    attack_false_alarm = float(np.mean(pred_test.loc[attack_mask, "is_unknown_attack"])) * 100.0 if attack_mask.sum() > 0 else 0.0

    unknown_count = int(np.sum(pred_held["is_unknown_attack"]))
    unknown_rate = float(np.mean(pred_held["is_unknown_attack"])) * 100.0

    print(f"        * Known Inlier Rate             : {known_inlier_rate:.2f}% (False Alarm Rate = {known_false_alarm_rate:.2f}%)")
    print(f"          - Benign False Alarm Rate      : {benign_false_alarm:.2f}%")
    print(f"          - Known Attack False Alarm Rate: {attack_false_alarm:.2f}%")
    print(f"        * Zero-Day Attacks Detected      : {unknown_count:,} / {len(X_held):,} ({unknown_rate:.2f}%)")

    per_class_results = []
    for raw_cls in cfg['held_out_classes']:
        clean_cls = clean_label_string(raw_cls)
        cls_mask = (pred_held["true_label"] == clean_cls)
        cls_total = int(np.sum(cls_mask))
        if cls_total > 0:
            cls_detected = int(np.sum(pred_held.loc[cls_mask, "is_unknown_attack"]))
            cls_rate = float(cls_detected / cls_total) * 100.0
            print(f"          - {clean_cls:<28} : {cls_detected:,} / {cls_total:,} ({cls_rate:.2f}%)")
            per_class_results.append({
                "Attack_Class": clean_cls,
                "Tested_Samples": cls_total,
                "Detected_As_Unknown": cls_detected,
                "Detection_Rate_Pct": round(cls_rate, 2),
            })

    # 5. Save Clean Numbered Output Files
    print(f"\n  [4/4] Writing clean files to: {ds_out_dir}")

    # File 2: ONLY the detected unknown attacks with clean columns
    mask_unknown = pred_held["is_unknown_attack"]
    df_unknown_detected = df_held.loc[mask_unknown].copy()
    df_unknown_detected["Label"] = [clean_label_string(l) for l in df_unknown_detected["Label"]]
    
    # Place key identification columns at the beginning
    df_unknown_detected.insert(0, "Dataset", display_name)
    df_unknown_detected.insert(1, "Sample_ID", pred_held.loc[mask_unknown, "sample_id"].values)
    df_unknown_detected.insert(2, "Attack_Type", df_unknown_detected.pop("Label"))
    df_unknown_detected.insert(3, "Predicted_Status", "UNKNOWN_ATTACK")
    df_unknown_detected.insert(4, "DBSCAN_Cluster", -1)
    df_unknown_detected.insert(5, "Anomaly_Score", pred_held.loc[mask_unknown, "anomaly_score"].values)
    df_unknown_detected.insert(6, "Distance_To_Nearest_Core", pred_held.loc[mask_unknown, "distance_to_nearest_core"].values)
    df_unknown_detected.insert(7, "Density_Radius_Eps", eps)

    file_detected = ds_out_dir / "2_DETECTED_UNKNOWN_ATTACKS.csv"
    df_unknown_detected.to_csv(file_detected, index=False)
    print(f"        [Saved] 2_DETECTED_UNKNOWN_ATTACKS.csv ({len(df_unknown_detected):,} rows)")

    # File 3: Complete prediction log
    file_all = ds_out_dir / "3_ALL_PREDICTIONS_LOG.csv"
    pred_held.to_csv(file_all, index=False)
    print(f"        [Saved] 3_ALL_PREDICTIONS_LOG.csv ({len(pred_held):,} rows)")

    # File 4: Clean text report
    report_file = ds_out_dir / "4_DETECTION_REPORT.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ZERO-DAY ATTACK DETECTION REPORT: {display_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"1. EXPERIMENT SETUP\n")
        f.write(f"   Target Zero-Day Attack : {attack_name}\n")
        f.write(f"   DBSCAN Parameters      : eps = {eps}, min_samples = {min_samples}\n")
        f.write(f"   Total Clusters Found   : {detector.total_clusters_found}\n")
        f.write(f"   Primary Clusters       : {len(detector.primary_cluster_ids)} (all {detector.total_clusters_found} used for classification)\n")
        f.write(f"   Training Set Size      : {len(X_train):,} samples\n")
        f.write(f"   Held-Out Attack Set    : {len(X_held):,} samples\n\n")
        f.write(f"2. CLUSTER PROFILES (Primary Clusters)\n")
        for c in cluster_rows:
            primary_tag = " [PRIMARY]" if c.get('Is_Primary', False) else ""
            f.write(f"   * Cluster {c['Cluster_ID']}: {c['Core_Sample_Count']:,} core points | "
                    f"Dominant: {c['Dominant_Traffic']} | Radius: {c['Density_Radius_Eps']}{primary_tag}\n")
        f.write(f"\n3. DETECTION METRICS\n")
        f.write(f"   Known Inlier Rate            : {known_inlier_rate:.2f}%\n")
        f.write(f"   Known False Alarm Rate       : {known_false_alarm_rate:.2f}%\n")
        f.write(f"     - Benign False Alarm Rate   : {benign_false_alarm:.2f}%\n")
        f.write(f"     - Known Attack False Alarm  : {attack_false_alarm:.2f}%\n")
        f.write(f"   Zero-Day Tested              : {len(X_held):,}\n")
        f.write(f"   Zero-Day Detected            : {unknown_count:,} ({unknown_rate:.2f}%)\n\n")
        f.write(f"4. PER-CLASS BREAKDOWN\n")
        for p in per_class_results:
            f.write(f"   - {p['Attack_Class']}: {p['Detected_As_Unknown']} / {p['Tested_Samples']} "
                    f"({p['Detection_Rate_Pct']}%)\n")
    print(f"        [Saved] 4_DETECTION_REPORT.txt\n")

    # Persist model
    joblib.dump(detector, models_dir / f"{folder_name}_dbscan.pkl")

    return {
        "display_name": display_name,
        "attack_name": attack_name,
        "tested": len(X_held),
        "detected": unknown_count,
        "detection_rate": unknown_rate,
        "known_false_alarm_rate": known_false_alarm_rate,
        "known_inlier_rate": known_inlier_rate,
        "benign_false_alarm": benign_false_alarm,
        "attack_false_alarm": attack_false_alarm,
        "total_clusters": detector.total_clusters_found,
        "per_class": per_class_results,
        "df_unknown": df_unknown_detected
    }


# ==============================================================================
# MASTER RUNNER
# ==============================================================================
def run_all_clustering():
    OUTPUT_UNKNOWN_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 88)
    print("           ZERO-DAY CYBER ATTACK DETECTION PIPELINE (DBSCAN CLUSTERING)")
    print("#" * 88 + "\n")

    results = []
    all_dfs = []

    for ds_key, cfg in DATASETS_CONFIG.items():
        res = run_dataset_pipeline(ds_key, cfg)
        results.append(res)
        if len(res["df_unknown"]) > 0:
            all_dfs.append(res["df_unknown"])

    # 1. Master Consolidated CSV
    if all_dfs:
        df_master = pd.concat(all_dfs, ignore_index=True)
        master_file = OUTPUT_UNKNOWN_ROOT / "ALL_DETECTED_UNKNOWN_ATTACKS.csv"
        df_master.to_csv(master_file, index=False)

    # 2. Executive Summary Dashboard
    total_tested = sum(r["tested"] for r in results)
    total_detected = sum(r["detected"] for r in results)
    overall_rate = (total_detected / total_tested) * 100.0 if total_tested > 0 else 0.0

    summary_text = []
    summary_text.append("=" * 88)
    summary_text.append("                      EXECUTIVE ZERO-DAY DETECTION SUMMARY")
    summary_text.append("=" * 88)
    summary_text.append(f"{'Dataset':<28} | {'Held-Out Attack':<28} | {'Tested':>7} | {'Detected':>8} | {'Detection Rate':>14}")
    summary_text.append("-" * 88)

    for r in results:
        for p in r["per_class"]:
            summary_text.append(
                f"{r['display_name']:<28} | {p['Attack_Class']:<28} | {p['Tested_Samples']:>7,} | "
                f"{p['Detected_As_Unknown']:>8,} | {p['Detection_Rate_Pct']:>13.2f}%"
            )

    summary_text.append("-" * 88)
    summary_text.append(f"{'TOTAL ACROSS ALL DATASETS':<59} | {total_tested:>7,} | {total_detected:>8,} | {overall_rate:>13.2f}%")
    summary_text.append("=" * 88)

    full_summary = "\n".join(summary_text)
    print("\n" + full_summary + "\n")

    summary_file = OUTPUT_UNKNOWN_ROOT / "00_EXECUTIVE_SUMMARY.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(full_summary + "\n\n")
        f.write("FOLDER ORGANIZATION GUIDE:\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("1_CLUSTERS_OVERVIEW.csv        : The 5 primary baseline clusters discovered\n")
        f.write("2_DETECTED_UNKNOWN_ATTACKS.csv : The exact attack samples flagged as unknown\n")
        f.write("3_ALL_PREDICTIONS_LOG.csv      : Full evaluation log with anomaly scores\n")
        f.write("4_DETECTION_REPORT.txt         : Detailed technical report\n")
        f.write("ALL_DETECTED_UNKNOWN_ATTACKS.csv : Master combined file of all detected zero-day attacks\n")

    print(f"  [SUCCESS] Executive summary written to : {summary_file}")
    print(f"  [SUCCESS] Master detected attacks saved : {master_file} ({len(df_master):,} total samples)")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clean DBSCAN unknown attack detection.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        run_all_clustering()
    else:
        if args.dataset in DATASETS_CONFIG:
            run_dataset_pipeline(args.dataset, DATASETS_CONFIG[args.dataset])
        else:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(DATASETS_CONFIG.keys())}")
