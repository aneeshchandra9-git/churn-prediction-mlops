"""
Churn model retraining pipeline.

prepare_data -> train_model -> evaluate_and_log -> promote_model

Reuses DataPipeline from src/, trains the tuned Logistic Regression
(Week 4a settings) on SMOTE-balanced data, logs metrics to MLflow,
and promotes the new model only if it passes a minimum F1 quality gate.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, task

# ---------- Paths (inside the container) ----------
PROJECT_DIR = Path("/opt/airflow/project")
RAW_DATA = PROJECT_DIR / "data" / "raw" / "churn_data.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "airflow"
MODELS_DIR = PROJECT_DIR / "models"
CANDIDATE_DIR = MODELS_DIR / "candidate"
PREVIOUS_DIR = MODELS_DIR / "previous"
MLFLOW_URI = f"sqlite:///{PROJECT_DIR / 'mlflow.db'}"

# ---------- Settings ----------
MIN_F1 = 0.30  # quality gate: candidate must reach this F1 to be promoted

default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="churn_model_retraining",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "churn"],
)
def churn_model_retraining():

    @task
    def prepare_data():
        """Run DataPipeline; save splits, scaler and encoders to disk."""
        sys.path.insert(0, str(PROJECT_DIR))
        import joblib
        from src.data_pipeline import DataPipeline

        pipeline = DataPipeline(random_state=42)
        X_train, X_test, y_train, y_test = pipeline.run_pipeline(str(RAW_DATA))

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
        X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
        y_train.to_frame("Churn").to_csv(PROCESSED_DIR / "y_train.csv", index=False)
        y_test.to_frame("Churn").to_csv(PROCESSED_DIR / "y_test.csv", index=False)

        joblib.dump(pipeline.scaler, CANDIDATE_DIR / "scaler.pkl")
        joblib.dump(pipeline.label_encoders, CANDIDATE_DIR / "label_encoders.pkl")

        return {"train_rows": len(X_train), "test_rows": len(X_test)}

    @task
    def train_model(data_info: dict):
        """Balance training data with SMOTE and train the tuned model."""
        import joblib
        import pandas as pd
        from imblearn.over_sampling import SMOTE
        from sklearn.linear_model import LogisticRegression

        X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Churn"]

        X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = LogisticRegression(
            C=0.001, solver="liblinear", max_iter=200, random_state=42
        )
        model.fit(X_bal, y_bal)

        joblib.dump(model, CANDIDATE_DIR / "model.pkl")

        return {
            "train_rows": data_info["train_rows"],
            "train_rows_after_smote": len(X_bal),
        }

    @task
    def evaluate_and_log(train_info: dict):
        """Evaluate on the original imbalanced test set; log to MLflow."""
        import joblib
        import mlflow
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, roc_auc_score,
        )

        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Churn"]
        model = joblib.load(CANDIDATE_DIR / "model.pkl")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("week7_airflow_retraining")
        with mlflow.start_run(run_name="airflow_retrain"):
            mlflow.log_params({
                "model": "LogisticRegression",
                "C": 0.001,
                "solver": "liblinear",
                "max_iter": 200,
                "smote": True,
                "train_rows_after_smote": train_info["train_rows_after_smote"],
            })
            mlflow.log_metrics(metrics)

        return metrics

    @task(retries=0)
    def promote_model(metrics: dict):
        """Quality gate: replace the served model only if F1 >= MIN_F1."""
        import shutil

        if metrics["f1"] < MIN_F1:
            raise ValueError(
                f"Candidate F1 {metrics['f1']:.4f} is below the minimum "
                f"{MIN_F1}. Model NOT promoted."
            )

        # Back up the current production files before replacing them.
        # copyfile copies contents only: Linux permission flags can't be
        # set on Windows-mounted folders, so shutil.copy would fail here.
        PREVIOUS_DIR.mkdir(parents=True, exist_ok=True)
        for name in ["logistic_regression_tuned.pkl", "scaler.pkl"]:
            current = MODELS_DIR / name
            if current.exists():
                shutil.copyfile(current, PREVIOUS_DIR / name)

        shutil.copyfile(CANDIDATE_DIR / "model.pkl", MODELS_DIR / "logistic_regression_tuned.pkl")
        shutil.copyfile(CANDIDATE_DIR / "scaler.pkl", MODELS_DIR / "scaler.pkl")
        shutil.copyfile(CANDIDATE_DIR / "label_encoders.pkl", MODELS_DIR / "label_encoders.pkl")

        return f"Promoted model with F1={metrics['f1']:.4f}"
    # ---------- Wiring: each task's output feeds the next ----------
    data_info = prepare_data()
    train_info = train_model(data_info)
    metrics = evaluate_and_log(train_info)
    promote_model(metrics)


churn_model_retraining()