"""
Week 4b: Feature Engineering
=============================

Objective:
- Load Week 4a balanced training and imbalanced test data
- Create engineered features (interactions, polynomials, domain-informed)
- Retrain Logistic Regression with best hyperparameters from Week 4a
- Compare performance: Week 4a (basic features) vs Week 4b (engineered features)
- Log results to MLflow

Features to Create:
===================

INTERACTION FEATURES (combine two features):
- ChargePerMonth: TotalCharges / (Tenure + 1)
  Why: Captures cost efficiency. Expensive early customers = high churn risk
  
- TenureCharge: MonthlyCharges * Tenure
  Why: Captures total commitment. High value for long-term customers
  
- ChargePerTenureYear: MonthlyCharges / max(Tenure, 1)
  Why: Monthly cost normalized by tenure. Shows cost per year of relationship

POLYNOMIAL FEATURES (squared values, non-linear):
- Tenure_squared: Tenure ** 2
  Why: Churn is not linear with tenure. Peaks at year 2, then drops (non-linear curve)
  
- MonthlyCharges_squared: MonthlyCharges ** 2
  Why: Churn might accelerate with high prices (non-linear price effect)

DOMAIN FEATURES (business logic):
- IsHighValue: (MonthlyCharges > median) & (Tenure > 12)
  Why: VIP customers (high revenue + loyal). May have different churn behavior
  
- IsEarlyCustomer: Tenure <= 12
  Why: First year is high-risk period. Flag these explicitly
  
- IsMonthToMonth: (ContractType == 'Month-to-month')
  Why: Month-to-month contracts are high-risk (no lock-in). Binary flag is strong signal

RESULT: 12 original features + 8 engineered = 20 total features
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow for experiment tracking
import mlflow
from mlflow.sklearn import log_model

# ============================================================================
# PART 1: CONFIGURATION & SETUP
# ============================================================================

np.random.seed(42)

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
ARTIFACTS_DIR = Path('week4_artifacts')

ARTIFACTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri('sqlite:///mlflow.db')
EXPERIMENT_NAME = 'week4_feature_engineering'

# ============================================================================
# PART 2: LOAD DATA
# ============================================================================

def load_data():
    """
    Load the same data as Week 4a (balanced training, imbalanced test).
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    X_train = pd.read_csv(DATA_DIR / 'X_train_balanced.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train_balanced.csv').squeeze()
    
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    print(f"\n✅ Training Data (Balanced via SMOTE):")
    print(f"   Shape: {X_train.shape}")
    print(f"   Churn rate: {y_train.mean():.2%}")
    
    print(f"\n✅ Test Data (Imbalanced - Realistic):")
    print(f"   Shape: {X_test.shape}")
    print(f"   Churn rate: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# PART 3: CREATE ENGINEERED FEATURES
# ============================================================================

def engineer_features(X):
    """
    Create new features from existing ones.
    
    Input: X (features dataframe)
    Output: X_engineered (original + new features)
    
    Strategy:
    1. Copy original features
    2. Create interaction features (combine two features)
    3. Create polynomial features (squared values)
    4. Create domain features (business logic)
    5. Return combined dataframe
    """
    X_eng = X.copy()
    
    print("\n" + "=" * 70)
    print("CREATING ENGINEERED FEATURES")
    print("=" * 70)
    
    # ========================================================================
    # INTERACTION FEATURES
    # ========================================================================
    # ========================================================================
    # INTERACTION FEATURES
    # ========================================================================
    print("\n[1] Interaction Features (combining two features):")
    
    # Feature 1: ChargePerMonth
    print("\n    Creating: ChargePerMonth = TotalCharges / (tenure + 1)")
    X_eng['ChargePerMonth'] = X_eng['TotalCharges'] / (X_eng['tenure'] + 1)
    print(f"       Mean: {X_eng['ChargePerMonth'].mean():.2f}")
    print(f"       Std:  {X_eng['ChargePerMonth'].std():.2f}")
    
    # Feature 2: TenureCharge
    print("\n    Creating: TenureCharge = MonthlyCharges * tenure")
    X_eng['TenureCharge'] = X_eng['MonthlyCharges'] * X_eng['tenure']
    print(f"       Mean: {X_eng['TenureCharge'].mean():.2f}")
    print(f"       Std:  {X_eng['TenureCharge'].std():.2f}")
    
    # Feature 3: ChargePerTenureYear
    print("\n    Creating: ChargePerTenureYear = MonthlyCharges / max(tenure, 1)")
    X_eng['ChargePerTenureYear'] = X_eng['MonthlyCharges'] / np.maximum(X_eng['tenure'], 1)
    print(f"       Mean: {X_eng['ChargePerTenureYear'].mean():.2f}")
    print(f"       Std:  {X_eng['ChargePerTenureYear'].std():.2f}")
    
    # ========================================================================
    # POLYNOMIAL FEATURES
    # ========================================================================
    print("\n[2] Polynomial Features (non-linear effects):")
    
    # Feature 4: tenure_squared
    print("\n    Creating: tenure_squared = tenure ** 2")
    X_eng['tenure_squared'] = X_eng['tenure'] ** 2
    print(f"       Mean: {X_eng['tenure_squared'].mean():.2f}")
    print(f"       Std:  {X_eng['tenure_squared'].std():.2f}")
    
    # Feature 5: MonthlyCharges_squared
    print("\n    Creating: MonthlyCharges_squared = MonthlyCharges ** 2")
    X_eng['MonthlyCharges_squared'] = X_eng['MonthlyCharges'] ** 2
    print(f"       Mean: {X_eng['MonthlyCharges_squared'].mean():.2f}")
    print(f"       Std:  {X_eng['MonthlyCharges_squared'].std():.2f}")
    
    # ========================================================================
    # DOMAIN FEATURES
    # ========================================================================
    print("\n[3] Domain Features (business logic):")
    
    # Feature 6: IsHighValue
    print("\n    Creating: IsHighValue = (MonthlyCharges > median) & (tenure > 12)")
    monthly_charges_median = X_eng['MonthlyCharges'].median()
    X_eng['IsHighValue'] = (
        (X_eng['MonthlyCharges'] > monthly_charges_median) & 
        (X_eng['tenure'] > 12)
    ).astype(int)
    print(f"       Count: {X_eng['IsHighValue'].sum()} high-value customers")
    print(f"       Proportion: {X_eng['IsHighValue'].mean():.2%}")
    
    # Feature 7: IsEarlyCustomer
    print("\n    Creating: IsEarlyCustomer = tenure <= 12")
    X_eng['IsEarlyCustomer'] = (X_eng['tenure'] <= 12).astype(int)
    print(f"       Count: {X_eng['IsEarlyCustomer'].sum()} early customers")
    print(f"       Proportion: {X_eng['IsEarlyCustomer'].mean():.2%}")
    
    # Feature 8: IsMonthToMonth
    print("\n    Creating: IsMonthToMonth = (Contract == 'Month-to-month')")
    X_eng['IsMonthToMonth'] = (X_eng['Contract'] == 'Month-to-month').astype(int)
    print(f"       Count: {X_eng['IsMonthToMonth'].sum()} month-to-month customers")
    print(f"       Proportion: {X_eng['IsMonthToMonth'].mean():.2%}")
    
    # Feature 8: IsMonthToMonth
    # Why: Contract type is strong churn signal
    # Logic: Binary flag for month-to-month contracts
    # Interpretation: Month-to-month = no lock-in, customer can leave anytime
    # Why binary? Original might be categorical. Binary is stronger signal.
    print("\n    Creating: IsMonthToMonth = (ContractType == 'Month-to-month')")
    X_eng['IsMonthToMonth'] = (X_eng['Contract'] == 'Month-to-month').astype(int)
    print(f"       Count: {X_eng['IsMonthToMonth'].sum()} month-to-month customers")
    print(f"       Proportion: {X_eng['IsMonthToMonth'].mean():.2%}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"\nOriginal features: {X.shape[1]}")
    print(f"New engineered features: {X_eng.shape[1] - X.shape[1]}")
    print(f"Total features: {X_eng.shape[1]}")
    
    print(f"\nFeature list:")
    for i, col in enumerate(X_eng.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return X_eng


# ============================================================================
# PART 4: STANDARDIZE FEATURES
# ============================================================================

def standardize_features(X_train, X_test):
    """
    Standardize (normalize) all features to mean=0, std=1.
    
    Why? Different features have different scales:
    - Tenure: 0-72 (months)
    - MonthlyCharges: 18-119 (dollars)
    - ChargePerMonth: 0.25-119 (computed)
    
    StandardScaler makes all features comparable to the model.
    Fit on training data, apply to test data (prevent data leakage).
    """
    print("\n" + "=" * 70)
    print("STANDARDIZING FEATURES")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame (for readability)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"\n✅ Features standardized")
    print(f"   Train: mean~0, std~1")
    print(f"   Test:  applied same scaler (no data leakage)")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# PART 5: TRAIN MODEL WITH ENGINEERED FEATURES
# ============================================================================

def train_model_with_engineered_features(X_train, y_train):
    """
    Train Logistic Regression using best hyperparameters from Week 4a.
    
    Best hyperparameters found in Week 4a:
    - C: 0.001
    - solver: liblinear
    - max_iter: 200
    """
    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION WITH ENGINEERED FEATURES")
    print("=" * 70)
    print("\nUsing best hyperparameters from Week 4a:")
    print("  C: 0.001")
    print("  solver: liblinear")
    print("  max_iter: 200")
    
    model = LogisticRegression(
        C=0.001,
        solver='liblinear',
        max_iter=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    print("✅ Training complete")
    
    return model


# ============================================================================
# PART 6: EVALUATE MODEL
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate on test set and return metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
    }
    
    print(f"\n{model_name} Test Set Performance:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc_roc:.4f}")
    
    return metrics


# ============================================================================
# PART 7: COMPARE WITH WEEK 4A
# ============================================================================

def create_comparison_table(week4a_metrics, week4b_metrics):
    """
    Compare Week 4a (basic features) vs Week 4b (engineered features).
    """
    print("\n" + "=" * 70)
    print("WEEK 4A vs WEEK 4B COMPARISON")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
        'Week 4a (Basic Features)': [
            week4a_metrics['precision'],
            week4a_metrics['recall'],
            week4a_metrics['f1'],
            week4a_metrics['auc_roc'],
        ],
        'Week 4b (Engineered Features)': [
            week4b_metrics['precision'],
            week4b_metrics['recall'],
            week4b_metrics['f1'],
            week4b_metrics['auc_roc'],
        ],
    })
    
    # Calculate improvement
    comparison_df['Improvement'] = (
        (comparison_df['Week 4b (Engineered Features)'] - comparison_df['Week 4a (Basic Features)']) / 
        comparison_df['Week 4a (Basic Features)'] * 100
    ).round(2).astype(str) + '%'
    
    print("\n" + comparison_df.to_string(index=False))
    
    return comparison_df


# ============================================================================
# PART 8: LOAD WEEK 4A BASELINE
# ============================================================================

def load_week4a_metrics():
    """
    Load Week 4a results from the summary JSON file.
    """
    print("\n" + "=" * 70)
    print("LOADING WEEK 4A RESULTS FOR COMPARISON")
    print("=" * 70)
    
    summary_path = ARTIFACTS_DIR / 'week4_hyperparameter_tuning_summary.json'
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            week4a_summary = json.load(f)
        
        week4a_metrics = week4a_summary['test_metrics']
        print(f"\n✅ Loaded Week 4a results")
        print(f"   F1: {week4a_metrics['f1']:.4f}")
        
        return week4a_metrics
    else:
        print(f"\n❌ Week 4a summary not found at {summary_path}")
        return None


# ============================================================================
# PART 9: LOG TO MLFLOW
# ============================================================================

def log_to_mlflow(model, week4b_metrics, comparison_df, X_train, y_train, X_test, y_test):
    """
    Log Week 4b results to MLflow.
    """
    print("\n" + "=" * 70)
    print("LOGGING TO MLFLOW")
    print("=" * 70)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name='week4_feature_engineering_run'):
        # Log parameters
        mlflow.log_param('num_features', X_train.shape[1])
        mlflow.log_param('num_original_features', 12)
        mlflow.log_param('num_engineered_features', X_train.shape[1] - 12)
        mlflow.log_param('feature_scaling', 'StandardScaler')
        
        # Log metrics
        mlflow.log_metric('test_precision', week4b_metrics['precision'])
        mlflow.log_metric('test_recall', week4b_metrics['recall'])
        mlflow.log_metric('test_f1', week4b_metrics['f1'])
        mlflow.log_metric('test_auc_roc', week4b_metrics['auc_roc'])
        
        # Log model
        log_model(model, 'model')
        
        # Log comparison table
        if comparison_df is not None:
            comparison_df.to_csv('week4_comparison_detailed.csv', index=False)
            mlflow.log_artifact('week4_comparison_detailed.csv')
            os.remove('week4_comparison_detailed.csv')
        
        print(f"\n✅ Logged to MLflow experiment: {EXPERIMENT_NAME}")


# ============================================================================
# PART 10: SAVE RESULTS
# ============================================================================

def save_results(model, week4b_metrics, comparison_df, X_train):
    """
    Save the engineered model and summary.
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save model
    model_path = MODELS_DIR / 'logistic_regression_engineered.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Saved engineered model to {model_path}")
    
    # Save summary
    summary = {
        'week': 4,
        'task': 'feature_engineering',
        'num_features': X_train.shape[1],
        'num_engineered_features': X_train.shape[1] - 12,
        'test_metrics': {
            'precision': float(week4b_metrics['precision']),
            'recall': float(week4b_metrics['recall']),
            'f1': float(week4b_metrics['f1']),
            'auc_roc': float(week4b_metrics['auc_roc']),
        },
    }
    
    summary_path = ARTIFACTS_DIR / 'week4_feature_engineering_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to {summary_path}")


# ============================================================================
# PART 11: MAIN ORCHESTRATION
# ============================================================================

def main():
    """
    Orchestrate the entire Week 4b feature engineering pipeline.
    """
    print("\n")
    print("🚀 " * 20)
    print("WEEK 4B: FEATURE ENGINEERING")
    print("🚀 " * 20)
    
    try:
        # Step 1: Load data
        X_train, y_train, X_test, y_test = load_data()
        
        # Step 2: Engineer features
        X_train_eng = engineer_features(X_train)
        X_test_eng = engineer_features(X_test)
        
        # Step 3: Standardize features
        X_train_scaled, X_test_scaled, scaler = standardize_features(X_train_eng, X_test_eng)
        
        # Step 4: Train model with engineered features
        model = train_model_with_engineered_features(X_train_scaled, y_train)
        
        # Step 5: Evaluate on test set
        print("\n" + "=" * 70)
        print("EVALUATING ENGINEERED MODEL ON TEST SET")
        print("=" * 70)
        week4b_metrics = evaluate_model(model, X_test_scaled, y_test, "Week 4b Engineered")
        
        # Step 6: Load Week 4a results and compare
        week4a_metrics = load_week4a_metrics()
        if week4a_metrics:
            comparison_df = create_comparison_table(week4a_metrics, week4b_metrics)
        else:
            comparison_df = None
        
        # Step 7: Log to MLflow
        log_to_mlflow(model, week4b_metrics, comparison_df, X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Step 8: Save results
        save_results(model, week4b_metrics, comparison_df, X_train_scaled)
        
        print("\n" + "=" * 70)
        print("✅ WEEK 4B COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review MLflow dashboard at http://localhost:5000")
        print("2. Examine week4_feature_engineering_summary.json for details")
        print("3. Proceed to Week 4c: Threshold Tuning (optional)")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()