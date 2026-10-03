# Week 6 Summary — Containerizing the API with Docker

## Goal
Package the Week 5 FastAPI service into a Docker container — a self-contained, portable unit that runs identically on any machine, independent of the local Python/virtual-environment setup.

## What was built

**File:** `requirements-api.txt`

fastapi==0.103.0
uvicorn==0.23.2
pydantic==2.3.0
scikit-learn==1.9.0
joblib==1.5.3
numpy==2.5.2


**File:** `Dockerfile`
```dockerfile
FROM python:3.12-slim

WORKDIR /code

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY models/ ./models/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File:** `.dockerignore` — excludes `venv/`, `.git/`, `notebooks/`, `logs/`, `mlruns/`, `mlflow_artifacts/`, `mlflow.db`, `data/`, `tests/`, and docs from the build context.

## Environment setup (one-time, this machine)

- Installed Docker Desktop; required enabling **Virtual Machine Platform** and **Windows Subsystem for Linux** via "Turn Windows features on or off," a restart, `wsl --set-default-version 2`, `wsl --update`.
- Root cause of initial "Virtualization support not detected" was Windows features being off, not hardware — confirmed via Task Manager showing hardware virtualization `Enabled`.

## Debugging chain (build/runtime version mismatches)

1. `ModuleNotFoundError: No module named 'numpy._core'` — container's numpy/scikit-learn versions were older than what actually saved the model. Verified real installed versions via `pip show` → `numpy==2.5.2`, `scikit-learn==1.9.0`, `joblib==1.5.3`.
2. `scikit-learn==1.9.0` needs Python ≥3.11 → bumped base image to `python:3.11-slim`.
3. `numpy==2.5.2` needs Python ≥3.12 → bumped base image to `python:3.12-slim`, resolved both.

**Lesson**: a `.pkl` file is tied to the exact library versions that created it. `requirements.txt` can silently drift from what's actually installed — verify with `pip show <package>` rather than trusting the file.

## Leftover items — completed this session

**1. Error handling on `/predict`**
Wrapped the prediction logic in `try/except`, raising `HTTPException(status_code=500, ...)` on failure instead of exposing a raw crash traceback. Verified with a valid request (succeeds normally) and a deliberately broken one (`NaN` input, which `LogisticRegression` rejects) — confirmed a clean `500` JSON error response instead of a server crash.

**2. Real preprocessing (scaler) inside the API**
- Discovered `DataPipeline` (`src/data_pipeline.py`) fits a `StandardScaler` and `LabelEncoder`s **in memory during training** but never persists them to a file — only the trained model was saved, not the transformers that produced its input.
- Wrote `save_scaler.py`: re-runs `DataPipeline`'s `load_data` → `clean_data` → `transform_data(fit=True)` on the original training data (`data/raw/churn_data.csv`) to refit the identical scaler, then saves it via `joblib.dump(pipeline.scaler, 'models/scaler.pkl')`.
- Confirmed the scaler's expected column order matches the existing `CustomerData` field order exactly — no reordering needed.
- Updated `main.py` to load `scaler.pkl` at startup alongside the model, and apply `scaler.transform(raw_features)` inside `/predict` before calling `model.predict(...)`.
- API now accepts genuine raw, human-scale values (e.g. `tenure: 24`, `MonthlyCharges: 70`) instead of requiring pre-standardized input.

**Known simplification, deliberately deferred**: categorical fields (`Contract`, `InternetService`, etc.) are still expected as their already-label-encoded numeric form, not raw text (e.g. `"Month-to-month"`). Saving and loading the fitted `LabelEncoder`s to accept raw category strings is a future step.

**3. `.dockerignore`** — added to keep the build context lean (excludes `venv/`, data, logs, notebooks, etc. from being sent to Docker).

## Verification

- Local (`uvicorn app.main:app --reload`): realistic raw input (`age: 45, tenure: 24, MonthlyCharges: 70, ...`) → `{"churn_prediction": 1, "churn_probability": 0.5215...}`
- Rebuilt Docker image (`docker build -t churn-api .`) and re-ran (`docker run -p 8000:8000 churn-api`) with the same input → **identical result**, confirming full parity between local and containerized versions.

## Next steps (not yet done)

- Persist and load `LabelEncoder`s so `/predict` can accept raw category text, not just pre-encoded numbers
- Decide on container registry / deployment target beyond local Docker
- Move into Prometheus (metrics) as the next stack item