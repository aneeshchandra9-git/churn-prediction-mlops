"""
Week 4c: Classification Threshold Tuning
=========================================

Objective:
- Load best model from Week 4a (tuned Logistic Regression, basic features)
- Generate probability predictions on test set
- Test different classification thresholds (0.1 to 0.9)
- Calculate Precision, Recall, F1 for each threshold
- Plot Precision-Recall curve and ROC curve
- Find optimal threshold based on F1 score
- Log results to MLflow

Why Threshold Tuning?
======================
Default threshold = 0.5 is arbitrary.
Different thresholds create different Precision-Recall trade-offs.

Lower threshold (0.3):
  → Predict more customers as "churn"
  → High Recall (catch 65% of churners) ✅
  → Low Precision (18% of predictions correct) ❌
  → Aggressive retention strategy

Higher threshold (0.7):
  → Predict fewer customers as "churn"
  → Low Recall (catch only 20% of churners) ❌
  → High Precision (40% of predictions correct) ✅
  → Conservative retention strategy

Optimal threshold depends on business cost of false alarm vs. missed churner.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow for experiment tracking
import mlflow

# ============================================================================
# PART 1: CONFIGURATION & SETUP
# ============================================================================

np.random.seed(42)

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
ARTIFACTS_DIR = Path('week4_artifacts')

ARTIFACTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri('sqlite:///mlflow.db')
EXPERIMENT_NAME = 'week4_threshold_tuning'

# ============================================================================
# PART 2: LOAD DATA & MODEL
# ============================================================================

def load_data_and_model():
    """
    Load test data and best model from Week 4a.
    """
    print("=" * 70)
    print("LOADING DATA AND MODEL")
    print("=" * 70)
    
    # Load test data
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    print(f"\n✅ Test Data Loaded:")
    print(f"   Shape: {X_test.shape}")
    print(f"   Churn rate: {y_test.mean():.2%}")
    print(f"   Churners: {(y_test==1).sum()}")
    print(f"   Non-churners: {(y_test==0).sum()}")
    
    # Load best model from Week 4a
    model_path = MODELS_DIR / 'logistic_regression_tuned.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\n✅ Model Loaded: {model_path}")
    
    return X_test, y_test, model


# ============================================================================
# PART 3: GENERATE PROBABILITY PREDICTIONS
# ============================================================================

def generate_predictions(model, X_test, y_test):
    """
    Generate probability predictions for all test samples.
    
    Model outputs:
    - y_pred_proba: Probabilities [0, 1] for each sample
      Example: 0.6 means model thinks customer has 60% chance to churn
    - y_pred_binary: Binary predictions using default threshold (0.5)
    """
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    
    # Get probability predictions (0 to 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # [:, 1] extracts churn probability (class 1) from [non-churn, churn] output
    
    print(f"\n✅ Generated {len(y_pred_proba)} probability predictions")
    print(f"\nProbability Distribution:")
    print(f"   Min: {y_pred_proba.min():.4f}")
    print(f"   Max: {y_pred_proba.max():.4f}")
    print(f"   Mean: {y_pred_proba.mean():.4f}")
    print(f"   Median: {np.median(y_pred_proba):.4f}")
    
    return y_pred_proba


# ============================================================================
# PART 4: TEST DIFFERENT THRESHOLDS
# ============================================================================

def test_thresholds(y_test, y_pred_proba):
    """
    Test different classification thresholds and calculate metrics.
    
    Process:
    1. For each threshold in [0.1, 0.2, ..., 0.9]:
       a. Convert probabilities to binary predictions
          If probability > threshold → predict 1 (churn)
          If probability ≤ threshold → predict 0 (no churn)
       b. Calculate Precision, Recall, F1 for this threshold
    2. Store results in DataFrame
    
    Why multiple thresholds?
    - We want to understand the Precision-Recall trade-off
    - Different thresholds serve different business objectives
    - We'll plot these to visualize the trade-off
    """
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT THRESHOLDS")
    print("=" * 70)
    
    thresholds = np.arange(0.1, 1.0, 0.05)  # 0.1, 0.15, 0.2, ..., 0.95
    
    results = []
    
    print(f"\nTesting {len(thresholds)} thresholds...\n")
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions using this threshold
        y_pred_binary = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        
        # Count predictions
        num_predicted_churn = y_pred_binary.sum()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_predicted_churn': num_predicted_churn,
        })
        
        print(f"Threshold {threshold:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Predicted={num_predicted_churn}")
    
    results_df = pd.DataFrame(results)
    
    return results_df


# ============================================================================
# PART 5: FIND OPTIMAL THRESHOLD
# ============================================================================

def find_optimal_threshold(results_df):
    """
    Find the threshold that maximizes F1 score.
    
    Why F1?
    - F1 is harmonic mean of Precision and Recall
    - Balances both metrics (doesn't favor one over the other)
    - Good for imbalanced classification where both matter
    
    Alternative: Could optimize for Recall if business prioritizes catching churners
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 70)
    
    # Find threshold with highest F1
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    best_precision = results_df.loc[best_idx, 'precision']
    best_recall = results_df.loc[best_idx, 'recall']
    
    print(f"\n✅ Optimal Threshold (maximizes F1):")
    print(f"   Threshold: {best_threshold:.2f}")
    print(f"   Precision: {best_precision:.4f}")
    print(f"   Recall: {best_recall:.4f}")
    print(f"   F1 Score: {best_f1:.4f}")
    print(f"   Predicted Churners: {results_df.loc[best_idx, 'num_predicted_churn']:.0f}")
    
    return best_threshold, results_df


# ============================================================================
# PART 6: PLOT PRECISION-RECALL CURVE
# ============================================================================

def plot_precision_recall_curve(results_df, best_threshold):
    """
    Plot Precision-Recall curve to visualize trade-off.
    
    Interpretation:
    - As threshold decreases (left side): Recall increases, Precision decreases
    - As threshold increases (right side): Recall decreases, Precision increases
    - Optimal threshold is marked with a point
    """
    print("\n" + "=" * 70)
    print("PLOTTING PRECISION-RECALL CURVE")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curve
    ax.plot(results_df['recall'], results_df['precision'], 
            marker='o', linewidth=2, markersize=6, label='Precision-Recall Curve')
    
    # Mark optimal threshold
    best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
    ax.plot(best_row['recall'], best_row['precision'], 
            marker='*', markersize=15, color='red', label=f'Optimal (threshold={best_threshold:.2f})')
    
    # Labels and formatting
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.set_title('Precision-Recall Curve: Threshold Tuning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add threshold labels on points
    for idx, row in results_df.iterrows():
        ax.annotate(f"{row['threshold']:.2f}", 
                   (row['recall'], row['precision']),
                   textcoords="offset points", xytext=(0,5), 
                   ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {ARTIFACTS_DIR / 'precision_recall_curve.png'}")
    plt.close()


# ============================================================================
# PART 7: PLOT ROC CURVE
# ============================================================================

def plot_roc_curve(y_test, y_pred_proba, best_threshold):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    ROC Curve:
    - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
      "Of actual non-churners, how many did we falsely predict as churners?"
    - Y-axis: True Positive Rate (Recall) = TP / (TP + FN)
      "Of actual churners, how many did we catch?"
    
    AUC (Area Under Curve):
    - 0.5 = Random guessing (diagonal line)
    - 1.0 = Perfect classification (top-left corner)
    - Your model: Likely ~0.45-0.50 (imbalanced data effect)
    
    Interpretation:
    - Higher AUC = better model
    - Marked point shows performance at chosen threshold
    """
    print("\n" + "=" * 70)
    print("PLOTTING ROC CURVE")
    print("=" * 70)
    
    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Find performance at optimal threshold
    idx_best = np.argmin(np.abs(thresholds_roc - best_threshold))
    fpr_best = fpr[idx_best]
    tpr_best = tpr[idx_best]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ROC curve
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Random classifier (diagonal)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
    
    # Mark optimal threshold point
    ax.plot(fpr_best, tpr_best, marker='*', markersize=15, color='red', 
           label=f'Optimal Threshold ({best_threshold:.2f})')
    
    # Labels and formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve: Threshold Tuning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {ARTIFACTS_DIR / 'roc_curve.png'}")
    plt.close()
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Performance at optimal threshold:")
    print(f"   False Positive Rate: {fpr_best:.4f}")
    print(f"   True Positive Rate (Recall): {tpr_best:.4f}")


# ============================================================================
# PART 8: CREATE THRESHOLD DECISION TABLE
# ============================================================================

def create_decision_table(results_df, y_test):
    """
    Create a human-readable decision table showing:
    "If I use threshold X, then I will:"
    - Catch Y% of actual churners (Recall)
    - Have Z% precision in my predictions
    - Reach out to N customers
    """
    print("\n" + "=" * 70)
    print("THRESHOLD DECISION TABLE")
    print("=" * 70)
    
    total_churners = (y_test == 1).sum()
    
    decision_table = results_df[['threshold', 'precision', 'recall', 'f1', 'num_predicted_churn']].copy()
    decision_table.columns = ['Threshold', 'Precision', 'Recall', 'F1', 'Predicted Churners']
    
    # Calculate number of actual churners caught at each threshold
    decision_table['Actual Churners Caught'] = (decision_table['Recall'] * total_churners).astype(int)
    
    # Format for display
    decision_table['Precision'] = decision_table['Precision'].apply(lambda x: f"{x:.2%}")
    decision_table['Recall'] = decision_table['Recall'].apply(lambda x: f"{x:.2%}")
    decision_table['F1'] = decision_table['F1'].apply(lambda x: f"{x:.4f}")
    decision_table['Threshold'] = decision_table['Threshold'].apply(lambda x: f"{x:.2f}")
    
    print("\n" + decision_table.to_string(index=False))
    
    return decision_table


# ============================================================================
# PART 9: LOG TO MLFLOW
# ============================================================================

def log_to_mlflow(results_df, best_threshold, y_test, y_pred_proba):
    """
    Log threshold tuning results to MLflow.
    """
    print("\n" + "=" * 70)
    print("LOGGING TO MLFLOW")
    print("=" * 70)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name='week4_threshold_tuning_run'):
        # Log best threshold
        best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
        
        mlflow.log_param('optimal_threshold', best_threshold)
        mlflow.log_metric('optimal_precision', best_row['precision'])
        mlflow.log_metric('optimal_recall', best_row['recall'])
        mlflow.log_metric('optimal_f1', best_row['f1'])
        
        # Log ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric('roc_auc', roc_auc)
        
        # Log threshold results table
        results_df.to_csv('threshold_results.csv', index=False)
        mlflow.log_artifact('threshold_results.csv')
        os.remove('threshold_results.csv')
        
        # Log plots
        mlflow.log_artifact(str(ARTIFACTS_DIR / 'precision_recall_curve.png'))
        mlflow.log_artifact(str(ARTIFACTS_DIR / 'roc_curve.png'))
        
        print(f"\n✅ Logged to MLflow experiment: {EXPERIMENT_NAME}")


# ============================================================================
# PART 10: SAVE RESULTS
# ============================================================================

def save_results(results_df, best_threshold, y_test, y_pred_proba):
    """
    Save threshold tuning results as JSON summary.
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
    
    summary = {
        'week': 4,
        'task': 'threshold_tuning',
        'optimal_threshold': float(best_threshold),
        'metrics_at_optimal_threshold': {
            'precision': float(best_row['precision']),
            'recall': float(best_row['recall']),
            'f1': float(best_row['f1']),
        },
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'interpretation': f"Use threshold {best_threshold:.2f}: catches {best_row['recall']:.1%} of churners with {best_row['precision']:.1%} precision",
    }
    
    summary_path = ARTIFACTS_DIR / 'week4_threshold_tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved summary to {summary_path}")


# ============================================================================
# PART 11: MAIN ORCHESTRATION
# ============================================================================

def main():
    """
    Orchestrate the entire Week 4c threshold tuning pipeline.
    """
    print("\n")
    print("🚀 " * 20)
    print("WEEK 4C: CLASSIFICATION THRESHOLD TUNING")
    print("🚀 " * 20)
    
    try:
        # Step 1: Load data and model
        X_test, y_test, model = load_data_and_model()
        
        # Step 2: Generate predictions
        y_pred_proba = generate_predictions(model, X_test, y_test)
        
        # Step 3: Test different thresholds
        results_df = test_thresholds(y_test, y_pred_proba)
        
        # Step 4: Find optimal threshold
        best_threshold, results_df = find_optimal_threshold(results_df)
        
        # Step 5: Plot Precision-Recall curve
        plot_precision_recall_curve(results_df, best_threshold)
        
        # Step 6: Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, best_threshold)
        
        # Step 7: Create decision table
        decision_table = create_decision_table(results_df, y_test)
        
        # Step 8: Log to MLflow
        log_to_mlflow(results_df, best_threshold, y_test, y_pred_proba)
        
        # Step 9: Save results
        save_results(results_df, best_threshold, y_test, y_pred_proba)
        
        print("\n" + "=" * 70)
        print("✅ WEEK 4C COMPLETE")
        print("=" * 70)
        print("\nKey Findings:")
        print(f"  Optimal Threshold: {best_threshold:.2f}")
        print(f"  Recall: {results_df[results_df['threshold']==best_threshold].iloc[0]['recall']:.2%} (catch this % of churners)")
        print(f"  Precision: {results_df[results_df['threshold']==best_threshold].iloc[0]['precision']:.2%} (accuracy of predictions)")
        print("\nArtifacts Created:")
        print(f"  - {ARTIFACTS_DIR / 'precision_recall_curve.png'}")
        print(f"  - {ARTIFACTS_DIR / 'roc_curve.png'}")
        print(f"  - {ARTIFACTS_DIR / 'week4_threshold_tuning_summary.json'}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()