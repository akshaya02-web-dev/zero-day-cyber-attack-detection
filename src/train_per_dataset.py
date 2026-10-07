import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_evaluate(dataset_name, data_dir, models_dir, results_dir):
    # Determine the dataset splits
    train_file = os.path.join(data_dir, dataset_name, "train", f"{dataset_name}.csv")
    val_file = os.path.join(data_dir, dataset_name, "validation", f"{dataset_name}.csv")
    test_file = os.path.join(data_dir, dataset_name, "test", f"{dataset_name}.csv")
    
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        # print(f"Skipping {dataset_name}: missing split files.")
        return
        
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)
    
    # Identify target column consistently
    target_col = None
    for col in ['Label', 'binary_label', 'attack_category']:
        if col in df_train.columns:
            target_col = col
            break
            
    if target_col is None:
        target_col = df_train.columns[-1]
        
    X_train = df_train.drop(columns=[target_col], errors='ignore')
    y_train = df_train[target_col]
    
    X_val = df_val.drop(columns=[target_col], errors='ignore')
    y_val = df_val[target_col]
    
    X_test = df_test.drop(columns=[target_col], errors='ignore')
    y_test = df_test[target_col]
    
    # Check classes
    classes = y_train.unique()
    if len(classes) <= 1:
        print("============================================================")
        print(f"DATASET: {dataset_name}")
        print("============================================================")
        print("TRAINING NOT POSSIBLE — ONLY ONE CLASS")
        return
        
    # Processed CSV files already contain final selected + normalized features.
    # We do NOT repeat scaler or feature selector.
    
    # Fit LabelEncoder on training labels only
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    # Explicit reproducible configurations
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20, 
        min_samples_leaf=2, 
        max_features="sqrt", 
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train_enc)
    
    xgb = XGBClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.08, 
        subsample=0.85, 
        colsample_bytree=0.85, 
        eval_metric='logloss', 
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train_enc)
    
    lgb = LGBMClassifier(
        n_estimators=300,
        max_depth=10,
        num_leaves=31,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42, 
        verbose=-1,
        n_jobs=-1
    )
    lgb.fit(X_train, y_train_enc)
    
    # Evaluate Validation
    res_val = []
    val_preds_dict = {}
    for model_name, model in [('Random Forest', rf), ('XGBoost', xgb), ('LightGBM', lgb)]:
        preds = model.predict(X_val)
        val_preds_dict[model_name] = preds
        acc = accuracy_score(y_val_enc, preds)
        prec = precision_score(y_val_enc, preds, average='weighted', zero_division=0)
        rec = recall_score(y_val_enc, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_val_enc, preds, average='weighted', zero_division=0)
        res_val.append((model_name, acc, prec, rec, f1))
        
    # Evaluate Test
    res_test = []
    test_preds_dict = {}
    for model_name, model in [('Random Forest', rf), ('XGBoost', xgb), ('LightGBM', lgb)]:
        preds = model.predict(X_test)
        test_preds_dict[model_name] = preds
        acc = accuracy_score(y_test_enc, preds)
        prec = precision_score(y_test_enc, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test_enc, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test_enc, preds, average='weighted', zero_division=0)
        res_test.append((model_name, acc, prec, rec, f1))
        
    # Output directories
    out_model_dir = os.path.join(models_dir, dataset_name)
    os.makedirs(out_model_dir, exist_ok=True)
    out_res_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(out_res_dir, exist_ok=True)
    
    # Save models and Label Encoder with compression to save disk space
    joblib.dump(rf, os.path.join(out_model_dir, 'random_forest.pkl'), compress=3)
    joblib.dump(xgb, os.path.join(out_model_dir, 'xgboost.pkl'), compress=3)
    joblib.dump(lgb, os.path.join(out_model_dir, 'lightgbm.pkl'), compress=3)
    joblib.dump(le, os.path.join(out_model_dir, 'label_encoder.pkl'))
        
    # Save results files
    val_df = pd.DataFrame(res_val, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    test_df = pd.DataFrame(res_test, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    val_df.to_csv(os.path.join(out_res_dir, 'validation_results.csv'), index=False)
    test_df.to_csv(os.path.join(out_res_dir, 'test_results.csv'), index=False)
    
    # Helper to save report and matrix
    def save_reports(y_true, y_true_orig, preds_dict, prefix):
        with open(os.path.join(out_res_dir, f'classification_report_{prefix}.txt'), 'w', encoding='utf-8') as f:
            for m_name, preds in preds_dict.items():
                f.write(f"=== {m_name} ===\n")
                f.write(classification_report(y_true, preds, target_names=le.classes_, zero_division=0))
                f.write("\n\n")
                
        # For confusion matrices, save for all models side by side or separately
        for m_name, preds in preds_dict.items():
            cm = confusion_matrix(y_true, preds)
            cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in le.classes_], columns=[f"Pred {c}" for c in le.classes_])
            cm_df.to_csv(os.path.join(out_res_dir, f'confusion_matrix_{prefix}_{m_name.replace(" ", "_")}.csv'))
            
    save_reports(y_val_enc, y_val, val_preds_dict, "validation")
    save_reports(y_test_enc, y_test, test_preds_dict, "test")

    # Terminal Output EXACTLY as requested
    print("============================================================")
    print(f"DATASET: {dataset_name}")
    print("============================================================")
    print()
    print("PREPROCESSING STATUS: ALREADY COMPLETED")
    print()
    print(f"Training Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(classes)}")
    print()
    print("------------------------------------------------------------")
    print("MODEL TRAINING")
    print("------------------------------------------------------------")
    print()
    print("Random Forest: TRAINED")
    print("XGBoost: TRAINED")
    print("LightGBM: TRAINED")
    print()
    print("---------------- VALIDATION RESULTS ----------------")
    print()
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<5}")
    for row in res_val:
        print(f"{row[0]:<18} {row[1]:<10.4f} {row[2]:<11.4f} {row[3]:<8.4f} {row[4]:<5.4f}")
    print()
    print("---------------- TEST RESULTS ----------------")
    print()
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<5}")
    for row in res_test:
        print(f"{row[0]:<18} {row[1]:<10.4f} {row[2]:<11.4f} {row[3]:<8.4f} {row[4]:<5.4f}")
    print()
    print("------------------------------------------------------------")
    print("SAVED OUTPUTS")
    print("------------------------------------------------------------")
    print()
    print("Models:")
    print(f"output/models/{dataset_name}/")
    print()
    print("Results:")
    print(f"output/results/{dataset_name}/")
    print()
    print("============================================================")
    print("DATASET TRAINING COMPLETED")
    print("============================================================")

def main():
    parser = argparse.ArgumentParser(description="Train models per dataset.")
    parser.add_argument('--dataset', type=str, help='Specific dataset name to train')
    parser.add_argument('--data_dir', type=str, default='output/splits', help='Directory with finalized splits')
    parser.add_argument('--models_dir', type=str, default='output/models', help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='output/results', help='Directory to save results')
    parser.add_argument('--config', type=str, help='Config file path (optional placeholder for compatibility)')
    args = parser.parse_args()
    
    if args.dataset:
        train_and_evaluate(args.dataset, args.data_dir, args.models_dir, args.results_dir)
    else:
        # Iterate over all datasets in data_dir
        if os.path.exists(args.data_dir):
            datasets = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
            for ds in sorted(datasets):
                # We skip 'test', 'train', 'validation' root dirs as they aren't datasets
                if ds in ['test', 'train', 'validation']:
                    continue
                train_and_evaluate(ds, args.data_dir, args.models_dir, args.results_dir)
        else:
            print(f"Data directory {args.data_dir} not found.")

if __name__ == "__main__":
    main()
