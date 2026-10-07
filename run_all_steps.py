"""
run_all_steps.py
=================
MASTER PIPELINE RUNNER: Executes and reports all 8 project steps end-to-end.

THE 8 STEPS:
  Step 1: Data Preprocessing
  Step 2: Feature Preparation & Extraction
  Step 3: Data Splitting & Feature Selection
  Step 4: Scaling & Normalization
  Step 5: Supervised Model Training (Random Forest, XGBoost, LightGBM)
  Step 6: Supervised Model Evaluation & Comparison
  Step 7: Known-Attack Detection & Zero-Day Blindspot Proof
  Step 8: Unknown-Attack Detection using DBSCAN Clustering
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path
import pandas as pd

from src.multi_dataset_pipeline import run_multi_dataset_pipeline
from src.create_unknown_experiment import main as create_unknown_experiment_main
from src.train_unknown_experiment import run_experiment_for_dataset, DATASETS_CONFIG as STEP7_CONFIG
from src.clustering_pipeline import run_all_clustering


def print_banner(step_num: int | str, title: str, description: str = "") -> None:
    print("\n" + "=" * 88)
    print(f"  STEP {step_num}: {title.upper()}")
    if description:
        print(f"  --> {description}")
    print("=" * 88 + "\n")


def run_steps_1_to_6(raw_dir: Path, output_dir: Path) -> None:
    print_banner("1 to 6", "Baseline Preprocessing, Splitting, Training & Evaluation",
                 "Processes raw datasets, selects top 20 features, scales, and trains RF + XGBoost + LightGBM")
    run_multi_dataset_pipeline(raw_dir, output_dir)


def run_step_7() -> None:
    print_banner(7, "Known-Attack Supervised Detection & Zero-Day Blindspot Proof",
                 "Separates held-out zero-day attacks and proves that supervised models falsely predict zero-days as Benign")
    print("[7A] Creating held-out experimental splits without data leakage...")
    create_unknown_experiment_main()

    print("\n[7B] Evaluating supervised models on known attacks vs held-out zero-days...")
    for ds in STEP7_CONFIG.keys():
        run_experiment_for_dataset(ds)


def run_step_8() -> None:
    print_banner(8, "Unknown-Attack Detection Using DBSCAN Clustering",
                 "Extracts 5 baseline clusters and flags zero-day packets as density Noise (-1) -> Unknown Attack")
    run_all_clustering()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 8 steps of the Zero-Day Cyber Attack Detection Framework.")
    parser.add_argument("--step", type=str, default="all",
                        help="Choose 'all', '1-6', '7', or '8' (default: 'all')")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip Steps 1-6 if baseline models are already trained, proceeding to Steps 7 and 8.")
    args = parser.parse_args()

    print("\n" + "#" * 88)
    print("      ADAPTIVE NETWORK BEHAVIOURAL ANALYSIS FRAMEWORK FOR ZERO-DAY ATTACK DETECTION")
    print("                      FULL 8-STEP PIPELINE EXECUTION")
    print("#" * 88)

    raw_dir = Path("data") / "raw"
    output_dir = Path("output")

    if args.step == "all":
        if not args.skip_baseline:
            run_steps_1_to_6(raw_dir, output_dir)
        else:
            print("\n[INFO] Skipping Steps 1-6 (Baseline already prepared).")
        run_step_7()
        run_step_8()

    elif args.step in ["1-6", "1", "2", "3", "4", "5", "6"]:
        run_steps_1_to_6(raw_dir, output_dir)

    elif args.step == "7":
        run_step_7()

    elif args.step == "8":
        run_step_8()

    else:
        print(f"Unknown step argument '{args.step}'. Options: 'all', '1-6', '7', '8'")
        return

    print("\n" + "#" * 88)
    print("                 ALL REQUESTED PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("#" * 88 + "\n")


if __name__ == "__main__":
    main()
