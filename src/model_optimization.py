"""
Week 4: Model Optimization - Hyperparameter Tuning
====================================================

Objective:
- Load Week 3 balanced training data and imbalanced test data
- Use GridSearchCV to tune Logistic Regression hyperparameters
- Compare tuned model vs Week 3 baseline
- Log all results to MLflow

Key Hyperparameters to Tune:
- C: Inverse regularization strength (0.001 to 100)
- solver: Optimization algorithm (lbfgs, liblinear)
- max_iter: Maximum iterations to converge (200 to 1000)
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow for experiment tracking
import mlflow
from mlflow.sklearn import log_model

# ============================================================================
# PART 1: CONFIGURATION & SETUP
# ============================================================================

# Set random seed for reproducibility
np.random.seed(42)

# Define file paths
DATA_DIR = Path('data')
MODELS_DIR = Path('models')
ARTIFACTS_DIR = Path('week4_artifacts')

# Create artifacts directory if it doesn't exist
ARTIFACTS_DIR.mkdir(exist_ok=True)

# MLflow configuration
mlflow.set_tracking_uri('sqlite:///mlflow.db')
EXPERIMENT_NAME = 'week4_hyperparameter_tuning'

# ============================================================================
# PART 2: LOAD DATA
# ============================================================================

def load_data():
    """
    Load training and test data from Week 3.
    
    Returns:
    - X_train, y_train: Balanced training data (8,206 rows after SMOTE)
    - X_test, y_test: Imbalanced test data (1,409 rows, 8.6% churn)
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load training data
    X_train = pd.read_csv(DATA_DIR / 'X_train_balanced.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train_balanced.csv').squeeze()
    
    # Load test data
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    print(f"\n✅ Training Data (Balanced via SMOTE):")
    print(f"   Shape: {X_train.shape}")
    print(f"   Churn distribution:\n{y_train.value_counts()}")
    print(f"   Churn rate: {y_train.mean():.2%}")
    
    print(f"\n✅ Test Data (Imbalanced - Realistic):")
    print(f"   Shape: {X_test.shape}")
    print(f"   Churn distribution:\n{y_test.value_counts()}")
    print(f"   Churn rate: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# PART 3: DEFINE HYPERPARAMETER GRID
# ============================================================================

def get_param_grid():
    """
    Define the hyperparameter search space for Logistic Regression.
    
    Hyperparameters:
    ----------------
    C (Regularization Strength):
        - Controls how hard the model tries to fit the training data
        - Smaller C = more regularization (model is conservative, avoids overfitting)
        - Larger C = less regularization (model fits training data harder)
        - Range: [0.001, 0.01, 0.1, 1, 10, 100]
        - Why this range? 0.001-0.1 for heavy regularization, 1-100 for lighter
    
    solver (Optimization Algorithm):
        - Algorithm used to find the best coefficients
        - 'lbfgs': Robust, stable, good for small datasets, slower
        - 'liblinear': Fast, good for binary classification (like our problem)
        - Why both? Different convergence behaviors may yield different optima
    
    max_iter (Convergence Iterations):
        - Maximum number of iterations the solver can take
        - Higher = more time to converge, slower but may find better solution
        - Lower = faster but may not fully converge
        - Range: [200, 500, 1000]
        - Why this range? 200-500 usually sufficient, 1000 for hard problems
    
    Total combinations: 6 C values × 2 solvers × 3 max_iter = 36 combinations
    GridSearchCV will test all 36 with 5-fold CV, so 36 × 5 = 180 model trains
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [200, 500, 1000],
    }
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH SPACE")
    print("=" * 70)
    print(f"\nC (Regularization): {param_grid['C']}")
    print(f"solver (Algorithm): {param_grid['solver']}")
    print(f"max_iter (Iterations): {param_grid['max_iter']}")
    print(f"\nTotal combinations to test: {len(param_grid['C']) * len(param_grid['solver']) * len(param_grid['max_iter'])}")
    print(f"With 5-fold CV: {len(param_grid['C']) * len(param_grid['solver']) * len(param_grid['max_iter']) * 5} model training runs\n")
    
    return param_grid


# ============================================================================
# PART 4: GRIDSEARCHCV - THE CORE OPTIMIZATION
# ============================================================================

def perform_grid_search(X_train, y_train, param_grid):
    """
    Perform GridSearchCV to find the best hyperparameters.
    
    What GridSearchCV does:
    -----------------------
    1. Creates a base model: LogisticRegression()
    2. For each hyperparameter combination in param_grid:
       a. Uses StratifiedKFold to split training data into 5 folds
       b. Trains the model 5 times (once on each fold's training subset)
       c. Evaluates on each fold's validation subset
       d. Averages the 5 evaluation scores (CV score)
    3. Remembers which combination had the highest CV score
    4. Retrains on full training data using best parameters
    
    Why StratifiedKFold?
    -------------------
    - Maintains class distribution in each fold
    - Important for imbalanced data (even though train is balanced, good practice)
    - Prevents one fold from being 60% churn, another 40% churn
    
    Why F1 scoring?
    ---------------
    - F1 = harmonic mean of Precision and Recall
    - Balances both metrics (doesn't favor Precision or Recall alone)
    - Good for imbalanced classification where both matter
    """
    print("=" * 70)
    print("GRIDSEARCHCV: SEARCHING HYPERPARAMETER SPACE")
    print("=" * 70)
    print("\nStarting GridSearchCV with 5-fold Stratified CV...")
    print("This will train ~180 models. Please wait (2-5 minutes)...\n")
    
    # Define base model
    base_model = LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1',                              # Optimize for F1 score
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # 5-fold CV
        n_jobs=-1,                                 # Use all CPU cores
        verbose=1,                                 # Print progress
    )
    
    # Fit GridSearchCV (trains all models)
    grid_search.fit(X_train, y_train)
    
    print("\n✅ GridSearchCV Complete!")
    print(f"\nBest Hyperparameters Found:")
    print(f"   C: {grid_search.best_params_['C']}")
    print(f"   solver: {grid_search.best_params_['solver']}")
    print(f"   max_iter: {grid_search.best_params_['max_iter']}")
    print(f"\nBest CV F1 Score: {grid_search.best_score_:.4f}")
    
    return grid_search


# ============================================================================
# PART 5: EVALUATE ON TEST SET
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model on test set and return all metrics.
    
    Metrics:
    --------
    - Precision: Of predicted churners, how many actually churned? (TP / (TP + FP))
    - Recall: Of actual churners, how many did we catch? (TP / (TP + FN))
    - F1: Harmonic mean of Precision and Recall
    - AUC-ROC: Area under Receiver Operating Characteristic curve
             Measures how well model separates churn vs non-churn across all thresholds
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn (class 1)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }
    
    print(f"\n{model_name} Test Set Performance:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc_roc:.4f}")
    
    return metrics


# ============================================================================
# PART 6: LOAD WEEK 3 BASELINE & COMPARE
# ============================================================================

def load_week3_baseline(X_test, y_test):
    """
    Load the Week 3 best model (Logistic Regression) for comparison.
    Compare Week 3 baseline vs Week 4 tuned model.
    """
    print("\n" + "=" * 70)
    print("LOADING WEEK 3 BASELINE FOR COMPARISON")
    print("=" * 70)
    
    # Load Week 3 model
    week3_model_path = MODELS_DIR / 'logistic_regression_model.pkl'
    
    if week3_model_path.exists():
        with open(week3_model_path, 'rb') as f:
            week3_model = pickle.load(f)
        
        print(f"\n✅ Loaded Week 3 baseline model from {week3_model_path}")
        
        # Evaluate Week 3 model
        week3_metrics = evaluate_model(week3_model, X_test, y_test, "Week 3 Baseline")
        
        return week3_model, week3_metrics
    else:
        print(f"❌ Week 3 model not found at {week3_model_path}")
        print("   Proceeding without baseline comparison.")
        return None, None


# ============================================================================
# PART 7: SAVE RESULTS & CREATE COMPARISON TABLE
# ============================================================================

def create_comparison_table(week3_metrics, week4_metrics):
    """
    Create a comparison table: Week 3 baseline vs Week 4 tuned.
    """
    if week3_metrics is None:
        print("\nSkipping comparison (Week 3 baseline not available)")
        return None
    
    print("\n" + "=" * 70)
    print("WEEK 3 vs WEEK 4 COMPARISON")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
        'Week 3 (Baseline)': [
            week3_metrics['precision'],
            week3_metrics['recall'],
            week3_metrics['f1'],
            week3_metrics['auc_roc'],
        ],
        'Week 4 (Tuned)': [
            week4_metrics['precision'],
            week4_metrics['recall'],
            week4_metrics['f1'],
            week4_metrics['auc_roc'],
        ],
    })
    
    # Calculate improvement
    comparison_df['Improvement'] = (
        (comparison_df['Week 4 (Tuned)'] - comparison_df['Week 3 (Baseline)']) / 
        comparison_df['Week 3 (Baseline)'] * 100
    ).round(2).astype(str) + '%'
    
    print("\n" + comparison_df.to_string(index=False))
    
    return comparison_df


# ============================================================================
# PART 8: LOG TO MLFLOW
# ============================================================================

def log_to_mlflow(grid_search, week4_metrics, comparison_df, X_train, y_train, X_test, y_test):
    """
    Log all Week 4 results to MLflow for experiment tracking.
    
    What we log:
    - Parameters: Best hyperparameters found
    - Metrics: Precision, Recall, F1, AUC-ROC on test set
    - Model: The best model as artifact
    - Artifacts: Comparison table, grid search results
    """
    print("\n" + "=" * 70)
    print("LOGGING TO MLFLOW")
    print("=" * 70)
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name='week4_gridsearch_run'):
        # Log hyperparameters
        mlflow.log_param('C', grid_search.best_params_['C'])
        mlflow.log_param('solver', grid_search.best_params_['solver'])
        mlflow.log_param('max_iter', grid_search.best_params_['max_iter'])
        mlflow.log_param('cv_folds', 5)
        mlflow.log_param('cv_scoring', 'f1')
        
        # Log metrics
        mlflow.log_metric('best_cv_f1_score', grid_search.best_score_)
        mlflow.log_metric('test_precision', week4_metrics['precision'])
        mlflow.log_metric('test_recall', week4_metrics['recall'])
        mlflow.log_metric('test_f1', week4_metrics['f1'])
        mlflow.log_metric('test_auc_roc', week4_metrics['auc_roc'])
        mlflow.log_metric('train_size', len(X_train))
        mlflow.log_metric('test_size', len(X_test))
        mlflow.log_metric('train_churn_rate', y_train.mean())
        mlflow.log_metric('test_churn_rate', y_test.mean())
        
        # Log model
        log_model(grid_search.best_estimator_, 'model')
        
        # Log comparison table as CSV artifact
        if comparison_df is not None:
            comparison_df.to_csv('week4_comparison.csv', index=False)
            mlflow.log_artifact('week4_comparison.csv')
            os.remove('week4_comparison.csv')
        
        # Log grid search results
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_df.to_csv('grid_search_results.csv', index=False)
        mlflow.log_artifact('grid_search_results.csv')
        os.remove('grid_search_results.csv')
        
        print(f"\n✅ Logged to MLflow experiment: {EXPERIMENT_NAME}")
        print(f"   Best hyperparameters")
        print(f"   Test metrics")
        print(f"   Model artifact")
        print(f"   Comparison table")


# ============================================================================
# PART 9: SAVE BEST MODEL & SUMMARY
# ============================================================================

def save_results(grid_search, week4_metrics, comparison_df):
    """
    Save the best model and create a summary JSON file.
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save best model
    best_model_path = MODELS_DIR / 'logistic_regression_tuned.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    print(f"\n✅ Saved best tuned model to {best_model_path}")
    
    # Create summary JSON
    summary = {
        'week': 4,
        'task': 'hyperparameter_tuning',
        'model_type': 'LogisticRegression',
        'best_hyperparameters': {
            'C': float(grid_search.best_params_['C']),
            'solver': str(grid_search.best_params_['solver']),
            'max_iter': int(grid_search.best_params_['max_iter']),
        },
        'best_cv_f1_score': float(grid_search.best_score_),
        'test_metrics': {
            'precision': float(week4_metrics['precision']),
            'recall': float(week4_metrics['recall']),
            'f1': float(week4_metrics['f1']),
            'auc_roc': float(week4_metrics['auc_roc']),
        },
    }
    
    summary_path = ARTIFACTS_DIR / 'week4_hyperparameter_tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to {summary_path}")


# ============================================================================
# PART 10: MAIN ORCHESTRATION
# ============================================================================

def main():
    """
    Orchestrate the entire Week 4a hyperparameter tuning pipeline.
    """
    print("\n")
    print("🚀 " * 20)
    print("WEEK 4A: HYPERPARAMETER TUNING")
    print("🚀 " * 20)
    
    try:
        # Step 1: Load data
        X_train, y_train, X_test, y_test = load_data()
        
        # Step 2: Define hyperparameter grid
        param_grid = get_param_grid()
        
        # Step 3: Perform GridSearchCV
        grid_search = perform_grid_search(X_train, y_train, param_grid)
        
        # Step 4: Evaluate best model on test set
        print("\n" + "=" * 70)
        print("EVALUATING BEST MODEL ON TEST SET")
        print("=" * 70)
        week4_metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, "Week 4 Tuned")
        
        # Step 5: Load Week 3 baseline and compare
        week3_model, week3_metrics = load_week3_baseline(X_test, y_test)
        comparison_df = create_comparison_table(week3_metrics, week4_metrics)
        
        # Step 6: Log to MLflow
        log_to_mlflow(grid_search, week4_metrics, comparison_df, X_train, y_train, X_test, y_test)
        
        # Step 7: Save results
        save_results(grid_search, week4_metrics, comparison_df)
        
        print("\n" + "=" * 70)
        print("✅ WEEK 4A COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review MLflow dashboard at http://localhost:5000")
        print("2. Examine week4_hyperparameter_tuning_summary.json for details")
        print("3. Proceed to Week 4b: Feature Engineering")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()