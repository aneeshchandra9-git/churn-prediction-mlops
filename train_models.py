import sys
sys.path.insert(0, 'src')

from src.data_pipeline import DataPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from models import MODELS

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("week2_multi_model_comparison")

# Load data using DataPipeline (same as train.py)
print("Loading data using DataPipeline...")
pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.run_pipeline('data/raw/churn_data.csv')

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train each model and log to MLflow
print("\n" + "="*60)
print("Training Models and Logging to MLflow")
print("="*60 + "\n")

for model_name, model in MODELS.items():
    print(f"Training {model_name}...")
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fi = f1_score(y_test, y_pred)
        
        # Handle ROC-AUC: only calculate if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None  # SVM doesn't support predict_proba by default
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", fi)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model (different methods for different model types)
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Print results
        print(f"✅ {model_name} Complete")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {fi:.4f}")
        if roc_auc is not None:
            print(f"   ROC-AUC:   {roc_auc:.4f}")
        else:
            print(f"   ROC-AUC:   N/A (model doesn't support predict_proba)")
        print()

print("="*60)
print("All models trained and logged to MLflow!")
print("="*60)