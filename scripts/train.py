import sys
sys.path.insert(0, 'src')

from data_pipeline import DataPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json

print("="*60)
print("BASELINE MODEL TRAINING")
print("="*60)

# Step 1: Prepare data
print("\n[1/3] Preparing data...")
pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.run_pipeline('data/raw/churn_data.csv')

# Step 2: Train model
print("\n[2/3] Training baseline model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✓ Model trained!")

# Step 3: Evaluate
print("\n[3/3] Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
}

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
for metric, value in metrics.items():
    print(f"{metric:12s}: {value:.4f}")

# Save model
print("\n[SAVE] Saving model...")
with open('models/baseline_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to: models/baseline_model.pkl")

# Save metrics
with open('models/baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✓ Metrics saved to: models/baseline_metrics.json")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nYour baseline model is ready!")