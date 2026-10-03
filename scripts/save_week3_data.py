"""
WEEK 3 DATA SAVER
=================
Load data via pipeline, apply SMOTE, save as CSVs for Week 4.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from data_pipeline import DataPipeline
from imblearn.over_sampling import SMOTE

print("\n" + "="*70)
print("SAVING WEEK 3 DATA FOR WEEK 4")
print("="*70)

# Load data
print("\n[1] Loading and processing data...")
pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.run_pipeline("data/raw/churn_data.csv")

print(f"    Train: {X_train.shape}")
print(f"    Test:  {X_test.shape}")

# Apply SMOTE
print("\n[2] Applying SMOTE to training data...")
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"    Balanced train: {X_train_balanced.shape}")
print(f"    Class distribution: {y_train_balanced.value_counts().to_dict()}")

# Save as CSVs
print("\n[3] Saving to CSV files...")
data_dir = Path('data')

X_train_balanced.to_csv(data_dir / 'X_train_balanced.csv', index=False)
pd.Series(y_train_balanced, name='Churn').to_csv(data_dir / 'y_train_balanced.csv', index=False)
X_test.to_csv(data_dir / 'X_test.csv', index=False)
pd.Series(y_test, name='Churn').to_csv(data_dir / 'y_test.csv', index=False)

print(f"    ✓ Saved X_train_balanced.csv ({X_train_balanced.shape[0]} rows)")
print(f"    ✓ Saved y_train_balanced.csv ({len(y_train_balanced)} labels)")
print(f"    ✓ Saved X_test.csv ({X_test.shape[0]} rows)")
print(f"    ✓ Saved y_test.csv ({len(y_test)} labels)")

print("\n" + "="*70)
print("✓ WEEK 3 DATA SAVED SUCCESSFULLY")
print("="*70)
print("\nReady for Week 4!\n")