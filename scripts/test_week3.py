#!/usr/bin/env python
"""Simple test to see if Week 3 script works"""

print("=" * 70)
print("TEST SCRIPT STARTING")
print("=" * 70)

import sys
print(f"\n✓ Python is working, version: {sys.version}")

print("\n[1] Adding src to path...")
sys.path.insert(0, 'src')

print("\n[2] Importing DataPipeline...")
try:
    from data_pipeline import DataPipeline
    print("✓ DataPipeline imported")
except Exception as e:
    print(f"✗ Failed to import DataPipeline: {e}")
    sys.exit(1)

print("\n[3] Creating pipeline and loading data...")
try:
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.run_pipeline('data/raw/churn_data.csv')
    print(f"✓ Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4] Importing SMOTE...")
try:
    from imblearn.over_sampling import SMOTE
    print("✓ SMOTE imported")
except Exception as e:
    print(f"✗ Failed to import SMOTE: {e}")
    sys.exit(1)

print("\n[5] Applying SMOTE...")
try:
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"✓ SMOTE applied: {X_train.shape} → {X_train_smote.shape}")
except Exception as e:
    print(f"✗ Failed to apply SMOTE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - Week 3 script should work!")
print("=" * 70)