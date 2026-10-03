"""
Save Week 3 Logistic Regression model as pickle for Week 4 comparison.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from data_pipeline import DataPipeline
from imblearn.over_sampling import SMOTE

print("\n" + "="*70)
print("SAVING WEEK 3 LOGISTIC REGRESSION MODEL")
print("="*70)

# Load data
print("\n[1] Loading and processing data...")
pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.run_pipeline("data/raw/churn_data.csv")

# Apply SMOTE
print("\n[2] Applying SMOTE...")
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train Logistic Regression (same as Week 3)
print("\n[3] Training Logistic Regression on balanced data...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Save as pickle
print("\n[4] Saving model...")
model_path = Path('models') / 'logistic_regression_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"    ✓ Saved to {model_path}")

print("\n" + "="*70)
print("✓ WEEK 3 MODEL SAVED")
print("="*70 + "\n")