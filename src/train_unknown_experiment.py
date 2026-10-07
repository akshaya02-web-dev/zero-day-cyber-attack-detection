"""
src/train_unknown_experiment.py
================================
STEP 7: HELD-OUT UNKNOWN ATTACK EXPERIMENT (RF + XGBoost + LightGBM)

Trains NEW Random Forest, XGBoost, and LightGBM models on ONLY known classes
for each suitable dataset, evaluates on validation and test sets (known classes),
and evaluates behavior on separate held-out unknown attack samples.

Key constraints:
- Existing baseline models (output/models/) and results (output/results/) are UNTOUCHED.
- No held-out samples appear in training, validation, or test data.
- The held-out attack classes exist ONLY in held_out/unknown_test.csv.
- All three models (RF, XGBoost, LightGBM) use the identical splits.
- True labels of held-out samples are used ONLY for offline evaluation.
- Forced misclassification alone does NOT constitute unknown detection.
- DBSCAN will be used in subsequent steps for behavioural novelty detection.
- Reproducibility guaranteed (random_state=42).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DATASETS_CONFIG = {
    'Tuesday-WorkingHours': {
        'known': ['BENIGN', 'FTP-Patator'],
        'held_out': ['SSH-Patator'],
        'expected_held_out': 3219,
    },
    'Wednesday-workingHours': {
        'known': ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest'],
        'held_out': ['DoS slowloris', 'Heartbleed'],
        'expected_held_out': 5396,
    },
    'Thursday-WorkingHours-Morning-WebAttacks': {
        'known': ['BENIGN', 'Web Attack \ufffd Brute Force', 'Web Attack \ufffd XSS'],
        'held_out': ['Web Attack \ufffd Sql Injection'],
        'expected_held_out': 21,
    }
}

DATA_ROOT = 'data/unknown_experiment'
MODELS_ROOT = 'output/unknown_models'
RESULTS_ROOT = 'output/unknown_results'
RANDOM_STATE = 42


def run_experiment_for_dataset(dataset_name: str):
    print("=" * 80)
    print(f"STEP 7: HELD-OUT UNKNOWN ATTACK EXPERIMENT -> {dataset_name}")
    print("=" * 80)
    
    cfg = DATASETS_CONFIG.get(dataset_name)
    if not cfg:
        base_name = dataset_name.replace('.pcap_ISCX', '')
        cfg = DATASETS_CONFIG.get(base_name)
        if cfg:
            dataset_name = base_name
        else:
            raise ValueError(f"Unknown dataset configuration for: {dataset_name}")
            
    known_classes = cfg['known']
    held_out_classes = cfg['held_out']
    
    ds_data_dir = os.path.join(DATA_ROOT, dataset_name)
    train_path = os.path.join(ds_data_dir, 'known', 'train.csv')
    val_path = os.path.join(ds_data_dir, 'known', 'validation.csv')
    test_path = os.path.join(ds_data_dir, 'known', 'test.csv')
    held_out_path = os.path.join(ds_data_dir, 'held_out', 'unknown_test.csv')
    
    # 1. Verify existence of files
    for p in [train_path, val_path, test_path, held_out_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required experiment file: {p}")
            
    print("\n[1/5] Loading data splits...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_held = pd.read_csv(held_out_path)
    
    feature_cols = [c for c in df_train.columns if c != 'Label']
    print(f"  Features count:     {len(feature_cols)}")
    print(f"  Train samples:      {len(df_train):,}")
    print(f"  Validation samples: {len(df_val):,}")
    print(f"  Test samples:       {len(df_test):,}")
    print(f"  Held-out samples:   {len(df_held):,}")
    
    # 2. Strict Data Leakage and Class Verification
    print("\n[2/5] Running strict leakage check across all splits...")
    train_labels = set(df_train['Label'].unique())
    val_labels = set(df_val['Label'].unique())
    test_labels = set(df_test['Label'].unique())
    held_labels = set(df_held['Label'].unique())
    
    leakage_train = train_labels.intersection(held_labels)
    leakage_val = val_labels.intersection(held_labels)
    leakage_test = test_labels.intersection(held_labels)
    
    if leakage_train or leakage_val or leakage_test:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE DETECTED!\n"
            f"  Train leakage: {leakage_train}\n"
            f"  Val leakage:   {leakage_val}\n"
            f"  Test leakage:  {leakage_test}"
        )
    print("  LEAKAGE CHECK: PASSED (Zero overlap between known splits and held-out split)")
    print(f"  Known classes in train: {sorted(list(train_labels))}")
    print(f"  Held-out classes:       {sorted(list(held_labels))}")
    
    if len(df_held) != cfg['expected_held_out']:
        print(f"  [NOTE] Held-out count {len(df_held)} vs expected {cfg['expected_held_out']}")
    else:
        print(f"  Held-out sample count matches expected: {len(df_held):,}")
        
    # Prepare Arrays
    X_train = df_train[feature_cols].values
    y_train = df_train['Label'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['Label'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['Label'].values
    
    X_held = df_held[feature_cols].values
    y_held = df_held['Label'].values
    
    # Fit LabelEncoder on training known classes ONLY
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    # 3. Model Training (RF, XGBoost, LightGBM on Known Classes ONLY)
    print("\n[3/5] Training 3 NEW models on known classes only...")
    
    # 3a. Random Forest
    print("  -> Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_enc)
    print("     Random Forest completed.")
    
    # 3b. XGBoost
    print("  -> Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train_enc)
    print("     XGBoost completed.")
    
    # 3c. LightGBM
    print("  -> Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=300,
        max_depth=10,
        num_leaves=31,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    lgb.fit(X_train, y_train_enc)
    print("     LightGBM completed.")
    
    models = {
        'Random Forest': rf,
        'XGBoost': xgb,
        'LightGBM': lgb
    }
    
    # 4. Evaluation on Known Validation and Test Sets
    print("\n[4/5] Evaluating all 3 models on Known Validation and Test sets...")
    val_metrics = []
    test_metrics = []
    
    print(f"  {'Model':<15} | {'Val Acc':<8} {'Val Prec':<8} {'Val Rec':<8} {'Val F1':<8} | {'Test Acc':<8} {'Test Prec':<8} {'Test Rec':<8} {'Test F1':<8}")
    print("  " + "-" * 88)
    
    for name, model in models.items():
        # Validation
        v_preds = model.predict(X_val)
        v_acc = accuracy_score(y_val_enc, v_preds)
        v_prec = precision_score(y_val_enc, v_preds, average='weighted', zero_division=0)
        v_rec = recall_score(y_val_enc, v_preds, average='weighted', zero_division=0)
        v_f1 = f1_score(y_val_enc, v_preds, average='weighted', zero_division=0)
        val_metrics.append({
            'Model': name,
            'Accuracy': round(v_acc, 4),
            'Precision': round(v_prec, 4),
            'Recall': round(v_rec, 4),
            'F1': round(v_f1, 4)
        })
        
        # Test
        t_preds = model.predict(X_test)
        t_acc = accuracy_score(y_test_enc, t_preds)
        t_prec = precision_score(y_test_enc, t_preds, average='weighted', zero_division=0)
        t_rec = recall_score(y_test_enc, t_preds, average='weighted', zero_division=0)
        t_f1 = f1_score(y_test_enc, t_preds, average='weighted', zero_division=0)
        test_metrics.append({
            'Model': name,
            'Accuracy': round(t_acc, 4),
            'Precision': round(t_prec, 4),
            'Recall': round(t_rec, 4),
            'F1': round(t_f1, 4)
        })
        
        print(f"  {name:<15} | {v_acc:<8.4f} {v_prec:<8.4f} {v_rec:<8.4f} {v_f1:<8.4f} | {t_acc:<8.4f} {t_prec:<8.4f} {t_rec:<8.4f} {t_f1:<8.4f}")
        
    # 5. Held-Out Evaluation (Offline Analysis Only)
    print("\n[5/5] Evaluating all 3 models on Held-Out unknown samples...")
    # RF
    rf_probs = rf.predict_proba(X_held)
    rf_pred_idx = np.argmax(rf_probs, axis=1)
    rf_pred_class = le.inverse_transform(rf_pred_idx)
    rf_conf = np.max(rf_probs, axis=1)
    
    # XGBoost
    xgb_probs = xgb.predict_proba(X_held)
    xgb_pred_idx = np.argmax(xgb_probs, axis=1)
    xgb_pred_class = le.inverse_transform(xgb_pred_idx)
    xgb_conf = np.max(xgb_probs, axis=1)
    
    # LightGBM
    lgb_probs = lgb.predict_proba(X_held)
    lgb_pred_idx = np.argmax(lgb_probs, axis=1)
    lgb_pred_class = le.inverse_transform(lgb_pred_idx)
    lgb_conf = np.max(lgb_probs, axis=1)
    
    df_held_preds = pd.DataFrame({
        'sample_identifier': [f"{dataset_name}_heldout_{i}" for i in range(len(df_held))],
        'true_label': y_held,
        'rf_predicted_class': rf_pred_class,
        'rf_prediction_confidence': np.round(rf_conf, 4),
        'xgb_predicted_class': xgb_pred_class,
        'xgb_prediction_confidence': np.round(xgb_conf, 4),
        'lgb_predicted_class': lgb_pred_class,
        'lgb_prediction_confidence': np.round(lgb_conf, 4)
    })
    
    for m_name, preds_col, conf_arr in [
        ('Random Forest', 'rf_predicted_class', rf_conf),
        ('XGBoost', 'xgb_predicted_class', xgb_conf),
        ('LightGBM', 'lgb_predicted_class', lgb_conf)
    ]:
        print(f"\n  --- {m_name} on Held-Out Samples ---")
        pred_distribution = df_held_preds[preds_col].value_counts()
        for cls_name, cnt in pred_distribution.items():
            pct = (cnt / len(df_held)) * 100
            print(f"    Forced as '{cls_name}': {cnt:,} ({pct:.2f}%)")
        print(f"    Confidence: Mean={conf_arr.mean():.4f}, Median={np.median(conf_arr):.4f}, Min={conf_arr.min():.4f}, Max={conf_arr.max():.4f}")

    # --------------------------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------------------------
    out_models_dir = os.path.join(MODELS_ROOT, dataset_name)
    os.makedirs(out_models_dir, exist_ok=True)
    
    out_results_dir = os.path.join(RESULTS_ROOT, dataset_name)
    os.makedirs(out_results_dir, exist_ok=True)
    
    # Save Models & Label Encoder
    rf_path = os.path.join(out_models_dir, 'random_forest.pkl')
    xgb_path = os.path.join(out_models_dir, 'xgboost.pkl')
    lgb_path = os.path.join(out_models_dir, 'lightgbm.pkl')
    le_path = os.path.join(out_models_dir, 'label_encoder.pkl')
    
    joblib.dump(rf, rf_path, compress=3)
    joblib.dump(xgb, xgb_path, compress=3)
    joblib.dump(lgb, lgb_path, compress=3)
    joblib.dump(le, le_path)
    
    print(f"\n  Saved models to: {out_models_dir}/")
    print("    - random_forest.pkl")
    print("    - xgboost.pkl")
    print("    - lightgbm.pkl")
    print("    - label_encoder.pkl")
    
    # Save validation_results.csv
    val_results_df = pd.DataFrame(val_metrics)
    val_results_csv = os.path.join(out_results_dir, 'validation_results.csv')
    val_results_df.to_csv(val_results_csv, index=False)
    
    # Save test_results.csv
    test_results_df = pd.DataFrame(test_metrics)
    test_results_csv = os.path.join(out_results_dir, 'test_results.csv')
    test_results_df.to_csv(test_results_csv, index=False)
    
    # Save held_out_predictions.csv
    held_preds_csv = os.path.join(out_results_dir, 'held_out_predictions.csv')
    df_held_preds.to_csv(held_preds_csv, index=False)
    
    # Save class_distribution.csv
    all_classes = sorted(list(train_labels.union(held_labels)))
    dist_records = []
    train_counts = df_train['Label'].value_counts().to_dict()
    val_counts = df_val['Label'].value_counts().to_dict()
    test_counts = df_test['Label'].value_counts().to_dict()
    held_counts = df_held['Label'].value_counts().to_dict()
    
    for c in all_classes:
        tr_c = train_counts.get(c, 0)
        va_c = val_counts.get(c, 0)
        te_c = test_counts.get(c, 0)
        he_c = held_counts.get(c, 0)
        dist_records.append({
            'Class': c,
            'Train': tr_c,
            'Validation': va_c,
            'Test': te_c,
            'Held_Out': he_c,
            'Total': tr_c + va_c + te_c + he_c,
            'Status': 'Held-Out' if c in held_labels else 'Known'
        })
    class_dist_df = pd.DataFrame(dist_records)
    class_dist_csv = os.path.join(out_results_dir, 'class_distribution.csv')
    class_dist_df.to_csv(class_dist_csv, index=False)
    
    # Save experiment_summary.txt
    summary_txt_path = os.path.join(out_results_dir, 'experiment_summary.txt')
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 75 + "\n")
        f.write(f"STEP 7: HELD-OUT EXPERIMENT SUMMARY: {dataset_name}\n")
        f.write("=" * 75 + "\n\n")
        f.write("1. DATASET CONFIGURATION\n")
        f.write(f"   Known Classes:    {sorted(list(train_labels))}\n")
        f.write(f"   Held-Out Classes: {sorted(list(held_labels))}\n")
        f.write("   Data Leakage:     NONE (PASSED - zero overlap across train/val/test and held-out)\n\n")
        
        f.write("2. SAMPLE COUNTS\n")
        f.write(f"   Known Train Samples:      {len(df_train):,}\n")
        f.write(f"   Known Validation Samples: {len(df_val):,}\n")
        f.write(f"   Known Test Samples:       {len(df_test):,}\n")
        f.write(f"   Held-Out Unknown Samples: {len(df_held):,}\n\n")
        
        f.write("3. SUPERVISED PERFORMANCE (KNOWN CLASSES ONLY)\n")
        f.write("   Validation Results:\n")
        for m in val_metrics:
            f.write(f"     - {m['Model']:<15}: Acc={m['Accuracy']:.4f}, Prec={m['Precision']:.4f}, Rec={m['Recall']:.4f}, F1={m['F1']:.4f}\n")
        f.write("\n   Test Results:\n")
        for m in test_metrics:
            f.write(f"     - {m['Model']:<15}: Acc={m['Accuracy']:.4f}, Prec={m['Precision']:.4f}, Rec={m['Recall']:.4f}, F1={m['F1']:.4f}\n")
        f.write("\n")
        
        f.write("4. HELD-OUT UNKNOWN SAMPLE PREDICTIONS (OFFLINE ANALYSIS ONLY)\n")
        f.write(f"   Total Held-Out Evaluated: {len(df_held):,}\n")
        for m_name, preds_col, conf_arr in [
            ('Random Forest', 'rf_predicted_class', rf_conf),
            ('XGBoost', 'xgb_predicted_class', xgb_conf),
            ('LightGBM', 'lgb_predicted_class', lgb_conf)
        ]:
            f.write(f"\n   Model: {m_name}\n")
            p_dist = df_held_preds[preds_col].value_counts()
            for cls_name, cnt in p_dist.items():
                pct = (cnt / len(df_held)) * 100
                f.write(f"     - Forced into '{cls_name}': {cnt:,} ({pct:.2f}%)\n")
            f.write(f"     - Confidence: Mean={conf_arr.mean():.4f}, Median={np.median(conf_arr):.4f}, Min={conf_arr.min():.4f}, Max={conf_arr.max():.4f}\n")
        f.write("\n")
        
        f.write("5. RESEARCH & METHODOLOGICAL NOTES\n")
        f.write("   - True held-out labels were used ONLY for offline evaluation, never for prediction or training.\n")
        f.write("   - Closed-set models (RF, XGBoost, LightGBM) force all novel inputs into known classes.\n")
        f.write("   - Forced misclassification alone DOES NOT constitute unknown attack detection.\n")
        f.write("   - Behavioural feature extraction and DBSCAN will be used in subsequent steps for\n")
        f.write("     novelty detection.\n")
        f.write("   - DBSCAN noise will NOT automatically be treated as an attack.\n")
        f.write("   - Existing Step 5-6 baseline models in output/models/ remain 100% untouched.\n")
        
    print(f"  Saved results to:    {out_results_dir}/")
    print(f"  [COMPLETED] Step 7 experiment finished for {dataset_name}\n")


def main():
    parser = argparse.ArgumentParser(description="Step 7: Held-Out Unknown Attack Experiment (RF + XGBoost + LightGBM)")
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help="Optional dataset name. If not specified, runs for all 3 suitable datasets."
    )
    args = parser.parse_args()
    
    if args.dataset:
        run_experiment_for_dataset(args.dataset)
    else:
        print("\nStarting Step 7 experiment (RF, XGBoost, LightGBM) for all 3 configured datasets...\n")
        for ds in DATASETS_CONFIG.keys():
            run_experiment_for_dataset(ds)
        print("=" * 80)
        print("ALL 3 DATASET EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)


if __name__ == '__main__':
    main()
