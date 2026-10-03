# Week 5 Summary — Model Serving with FastAPI

## Goal
Turn the trained Week 4 model into a live, callable prediction service using FastAPI — the first production-facing piece of the MLOps pipeline.

## What was built

**File:** `app/main.py`

- **FastAPI app initialized** (`app = FastAPI()`), the core service object all endpoints attach to.
- **Model loaded once at startup** via `joblib.load("models/logistic_regression_tuned.pkl")` — loaded into memory a single time, not per-request, to avoid disk-load latency on every call.
- **Request schema defined** (`CustomerData`, a Pydantic `BaseModel`) — enforces the exact 12 trained features (`age`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`), all typed as `float`. Invalid or missing fields are rejected automatically by FastAPI before the endpoint logic runs.
- **`GET /health`** — liveness check; confirms the service is up and the model loaded successfully (`model is not None`).
- **`POST /predict`** — accepts a `CustomerData` payload, assembles it into a feature array in trained-feature order, and returns:
  - `churn_prediction` (0 or 1)
  - `churn_probability` (model's confidence for the churn class)

## Model selection decision

Chose `logistic_regression_tuned.pkl` over three other saved candidates (`baseline_model.pkl`, `logistic_regression_model.pkl`, `logistic_regression_engineered.pkl`), confirmed against `week4_hyperparameter_tuning_summary.json`:
- `best_hyperparameters`: `C=0.001`, `solver=liblinear`, `max_iter=200`
- `test_metrics`: F1 = 0.321, AUC-ROC = 0.455

## Verification

- Server run locally via `uvicorn app.main:app --reload`
- `/health` confirmed: `{"status": "ok", "model_loaded": true}`
- `/predict` tested via FastAPI's auto-generated `/docs` (Swagger UI) with sample standardized inputs; returned a valid response:
```json
  {"churn_prediction": 0, "churn_probability": 0.4967156900251938}
```

## Known limitations / deliberate simplifications

- **No preprocessing inside the API**: the model was trained on standardized features, so `/predict` currently assumes the caller sends already-standardized values. Real preprocessing (fitting/applying the same scaler used in training) is deferred to a later week.
- **Weak model discrimination carries over**: the near-0.5 probability output reflects the AUC ≈ 0.455 finding from Week 4 — the API plumbing is correct, but the underlying model has limited predictive power. This is a modeling issue, not a serving issue.
- **No error handling yet**: a malformed model call or unexpected input currently surfaces as a raw, unfriendly error rather than a clean API error response.

## Next steps (not yet done)

- Add basic error handling for `/predict`
- Decide and implement where preprocessing/standardization happens (inside the API vs. expected of the caller)
- Containerize with Docker (per the planned stack)