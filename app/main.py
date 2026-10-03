import time
import joblib

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


app = FastAPI()
model = joblib.load("models/logistic_regression_tuned.pkl")
scaler = joblib.load("models/scaler.pkl")

# Column order the scaler and model were trained on. Must not change.
FEATURE_ORDER = [
    "age",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# --- Prometheus metrics (created once, at module level) ---
REQUEST_COUNT = Counter(
    'predict_requests_total', 'Total prediction requests', ['status']
)
REQUEST_LATENCY = Histogram(
    'predict_request_latency_seconds', 'Prediction request latency in seconds'
)
CHURN_PROBABILITY = Gauge(
    'churn_prediction_probability', 'Most recent predicted churn probability'
)


class CustomerData(BaseModel):
    # Numeric inputs: cannot be negative
    age: float = Field(ge=0)
    tenure: float = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    # Category codes, from models/label_encoders.pkl
    Contract: int = Field(ge=0, le=2)          # 0 Month-to-month, 1 One year, 2 Two year
    InternetService: int = Field(ge=0, le=2)   # 0 DSL, 1 Fiber optic, 2 No
    OnlineSecurity: int = Field(ge=0, le=1)    # 0 No, 1 Yes
    OnlineBackup: int = Field(ge=0, le=1)      # 0 No, 1 Yes
    DeviceProtection: int = Field(ge=0, le=1)  # 0 No, 1 Yes
    TechSupport: int = Field(ge=0, le=1)       # 0 No, 1 Yes
    StreamingTV: int = Field(ge=0, le=1)       # 0 No, 1 Yes
    StreamingMovies: int = Field(ge=0, le=1)   # 0 No, 1 Yes
    
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(customer: CustomerData):
    start_time = time.time()
    try:
        raw_values = [getattr(customer, name) for name in FEATURE_ORDER]
        features = scaler.transform([raw_values])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Each feature's push on the score: coefficient x scaled value.
        # Positive = towards churn, negative = towards staying.
        coefficients = model.coef_[0]
        intercept = float(model.intercept_[0])
        contributions = [
            {
                "feature": name,
                "value": float(raw),
                "contribution": float(coef * scaled),
            }
            for name, raw, coef, scaled in zip(
                FEATURE_ORDER, raw_values, coefficients, features[0]
            )
        ]
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        CHURN_PROBABILITY.set(probability)
        REQUEST_COUNT.labels(status="success").inc()

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "intercept": intercept,
            "contributions": contributions,
        }

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)