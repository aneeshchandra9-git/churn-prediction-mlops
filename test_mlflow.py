import mlflow

# Set tracking URI to SQLite (more reliable than file store)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Start an experiment
mlflow.set_experiment("test_experiment")

# Log a test metric
with mlflow.start_run():
    mlflow.log_metric("test_metric", 0.95)
    print("MLflow test successful!")
