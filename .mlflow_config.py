import os
import mlflow

# Set the tracking URI to SQLite (more reliable than file store)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set the artifacts location (where models are saved)
mlflow.set_artifact_uri("./mlflow_artifacts")

print("MLflow configured successfully!")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Artifact URI: {mlflow.get_artifact_uri()}")
