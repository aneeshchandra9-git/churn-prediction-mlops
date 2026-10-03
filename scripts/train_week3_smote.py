#!/usr/bin/env python
"""
WEEK 3: CLASS BALANCING WITH SMOTE
===================================
Apply SMOTE to balance training data, train all 4 models, log to MLflow.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import json
from pathlib import Path
from data_pipeline import DataPipeline

# CONFIG
DATA_FILE = "data/raw/churn_data.csv"
MODELS_DIR = Path("models/week3")
MLFLOW_DB = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "week3_class_balancing"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_DB)

# STEP 1: LOAD DATA
print("\n" + "="*70)
print("WEEK 3: CLASS BALANCING WITH SMOTE")
print("="*70)

print("\n[STEP 1] Loading and processing data...")
pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.run_pipeline(DATA_FILE)
print(f"\nData shapes: Train {X_train.shape}, Test {X_test.shape}")

# STEP 2: SHOW BEFORE SMOTE
print("\n[STEP 2] Class distribution BEFORE SMOTE:")
churn_before = (y_train == 1).sum()
non_churn_before = (y_train == 0).sum()
total_before = len(y_train)
print(f"  Non-churn: {non_churn_before:,} ({non_churn_before/total_before*100:.1f}%)")
print(f"  Churn:     {churn_before:,} ({churn_before/total_before*100:.1f}%)")
print(f"  Total:     {total_before:,}")

# STEP 3: APPLY SMOTE
print("\n[STEP 3] Applying SMOTE to training data...")
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"  ✓ SMOTE completed")

# STEP 4: SHOW AFTER SMOTE
print("\n[STEP 4] Class distribution AFTER SMOTE:")
churn_after = (y_train_smote == 1).sum()
non_churn_after = (y_train_smote == 0).sum()
total_after = len(y_train_smote)
print(f"  Non-churn: {non_churn_after:,} ({non_churn_after/total_after*100:.1f}%)")
print(f"  Churn:     {churn_after:,} ({churn_after/total_after*100:.1f}%)")
print(f"  Total:     {total_after:,}")

synthetic_created = churn_after - churn_before
print(f"\n  → Created {synthetic_created:,} synthetic churn samples")
print(f"  → Training set grew from {total_before:,} to {total_after:,}")

# STEP 5: TRAIN MODELS
print("\n[STEP 5] Training all four models on SMOTE-balanced data...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss", verbose=0),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

results = {}

# Create MLflow experiment
try:
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    print(f"Using existing MLflow experiment: {EXPERIMENT_NAME}\n")
except:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created new MLflow experiment: {EXPERIMENT_NAME}\n")

mlflow.set_experiment(EXPERIMENT_NAME)

# Train each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    with mlflow.start_run(run_name=f"{model_name}_week3_smote"):
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
        
        mlflow.log_param("smote_k_neighbors", 5)
        mlflow.log_param("training_data_size", total_after)
        mlflow.log_param("training_churn_ratio", churn_after / total_after)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
        
        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1-Score:  {f1:.4f}")
        if roc_auc is not None:
            print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")
        print()

# SUMMARY REPORT
print("\n" + "="*70)
print("WEEK 3 RESULTS (SMOTE-BALANCED TRAINING)")
print("="*70)

print("\nMetrics on test data (imbalanced - realistic):")
print("-" * 70)

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

# Save results
results_file = Path("week3_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {results_file}")

# KEY INSIGHTS
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"""
Training Data Transformation:
  Before: {non_churn_before:,} non-churn, {churn_before:,} churn ({churn_before/total_before*100:.1f}%)
  After:  {non_churn_after:,} non-churn, {churn_after:,} churn ({churn_after/total_after*100:.1f}%)
  Created: {synthetic_created:,} synthetic samples

Test Data (Unchanged & Realistic):
  Churn: {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.1f}%)
  Non-churn: {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.1f}%)

What Happened:
  ✓ Trained on SMOTE-balanced data (50/50 split)
  ✓ Evaluated on realistic imbalanced test data
  ✓ Models forced to learn actual churn patterns
  ✓ Metrics logged to MLflow for comparison

Next Step:
  View results in MLflow to compare Week 2 vs Week 3
  
  Run: mlflow ui
  Then visit: http://localhost:5000
""")

print("="*70)
print("✓ WEEK 3 COMPLETE!")
print("="*70)