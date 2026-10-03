# Customer Churn Prediction — End-to-End MLOps System

A production-style machine learning system that predicts which telecom customers are likely to leave, built over 8 weeks as a portfolio project.

It covers the full lifecycle: data pipeline, experiment tracking, a containerised prediction API, monitoring, automated retraining, and a web app where a **local LLM explains each prediction** using the model's exact per-feature contributions.

> **Focus:** the engineering system around the model. The model itself is deliberately simple, and its weakness (ROC-AUC ≈ 0.455) is documented openly rather than hidden.

---

## Architecture

```
                         docker compose (shared network)
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  Browser ─► Streamlit :8501 ──POST /predict──► FastAPI :8000         │
 │                 │            ◄─ probability + contributions ─        │
 │                 │                                    │               │
 │                 └─ factors ─► Ollama :11434          └─/metrics─►    │
 │                              (llama3.2:3b)          Prometheus :9090 │
 │                                                           │          │
 │                                                     Grafana :3000    │
 │                                                                      │
 │  Airflow :8080 ─(weekly)─► retrains ─► models/ + MLflow              │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Data pipeline** — reusable `DataPipeline` class: load, clean, label-encode, scale, stratified split
- **Experiment tracking** — Logistic Regression, Random Forest, XGBoost and SVM compared in MLflow
- **Class imbalance handling** — SMOTE on the training set only; evaluation on the original imbalanced test set
- **Prediction API** — FastAPI with Pydantic input validation (invalid inputs rejected with 422)
- **Explainability** — `/predict` returns each feature's exact contribution (coefficient × scaled value), verified to reproduce the predicted probability
- **Containerised** — six services in Docker Compose, versions pinned
- **Monitoring** — Prometheus metrics (request count, latency histogram, prediction gauge) and a Grafana dashboard **provisioned as code**
- **Automated retraining** — Airflow DAG: prepare → train → evaluate & log to MLflow → promote behind an F1 quality gate, with backup of the previous model
- **Web app** — Streamlit form with readable inputs, a contribution chart, a code-generated key-factors list, and an LLM summary
- **Local LLM** — Ollama running `llama3.2:3b`; no API keys, no cost

---

## Tech stack

| Area | Tools |
|---|---|
| ML | Python 3.12, pandas, scikit-learn, XGBoost, imbalanced-learn |
| Experiment tracking | MLflow |
| Serving | FastAPI, Pydantic, uvicorn |
| Containers | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Orchestration | Apache Airflow 3 |
| Front end | Streamlit, Altair |
| LLM | Ollama, Llama 3.2 3B |

---

## Results

| Metric | Value |
|---|---|
| F1 (churn class) | 0.3211 |
| Recall | 0.4569 |
| ROC-AUC | ≈ 0.455 |

The final model is a tuned Logistic Regression (`C=0.001`, `liblinear`) trained on SMOTE-balanced data. The Airflow pipeline reproduces these metrics exactly.

**Honest assessment:** predicted probabilities cluster between about 0.45 and 0.55, so the model barely separates churners from non-churners. The explanation layer makes this visible rather than hiding it — and it surfaced counter-intuitive learned patterns (e.g. short tenure *lowering* risk).

---

## Quick start

**Requirements:** Docker Desktop with about 6 GB of memory available to Docker.

```bash
git clone https://github.com/aneeshchandra9-git/churn-prediction-mlops.git
cd churn-prediction-mlops

# Start the app stack (Airflow excluded to save memory)
docker compose up -d api prometheus grafana ollama streamlit

# Download the LLM once (~2 GB)
docker compose exec ollama ollama pull llama3.2:3b
```

| Service | URL |
|---|---|
| Web app | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| Grafana (admin / admin) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

The first explanation after startup takes 1–3 minutes while the LLM loads into memory; after that, about 5 seconds.

**Retraining with Airflow** (stop Ollama first — both together exceed 6 GB):

```bash
docker compose stop ollama
docker compose up -d airflow      # UI at http://localhost:8080
```

---

## How the explanations stay grounded

A small LLM asked "why might this customer churn?" will invent plausible reasons. Instead:

1. The API computes each feature's **exact contribution** to the logistic regression score.
2. **Code** writes the prediction and probability, and a **key-factors list** — always accurate.
3. The LLM only rewords the top factors, which are pre-grouped into "raises risk" / "lowers risk", with a worked example and temperature 0. It never sees numbers.

Early versions invented reasons and even flipped a factor's direction; each fix made the input harder to misread rather than asking the model to try harder.

---

## Project structure

```
mlops-project/
├── app/main.py                     # FastAPI service
├── streamlit_app/app.py            # Web app
├── src/data_pipeline.py            # Reusable data pipeline
├── scripts/                        # Training and experiment scripts
├── dags/churn_retraining_dag.py    # Airflow retraining pipeline
├── monitoring/                     # Prometheus config, Grafana provisioning + dashboard
├── models/                         # Served model, scaler, label encoders
├── Dockerfile                      # API image
├── Dockerfile.airflow              # Airflow image
├── Dockerfile.streamlit            # Web app image
├── docker-compose.yml
├── requirements-*.txt              # Per-service dependencies
  ├── WEEK4/6/7/8_SUMMARY.md          # Detailed weekly write-ups (Week 5: app/WEEK5_SUMMARY.md)
  └── docs/early-planning/            # Original Week 1 planning documents
```

---

## Engineering challenges solved

- **Zero recall** across all models → traced to a 73/27 class imbalance; fixed with SMOTE on training data only
- **Model wouldn't load in Docker** → traced a chain of numpy / scikit-learn / Python version mismatches; pinned all three
- **Airflow `PermissionError`** → `shutil.copy` sets Linux permissions, impossible on Windows-mounted folders; switched to `shutil.copyfile`
- **API accepted impossible category codes** → found through the explanation layer; added Pydantic range validation
- **LLM hallucination** → four iterations of input design (see above)
- **Corrupted Grafana storage** → recovered in minutes because the dashboard and data source are provisioned from files

---

## Known limitations

- Weak model (ROC-AUC ≈ 0.455); explanations are faithful to the model, not evidence of real causes
- Scaler is fitted before the train/test split (minor data leakage)
- Model files are baked into the API image, so retraining requires an image rebuild
- Airflow runs in standalone mode with SQLite and no login — development only
- No authentication on any service — local use only

---

## Weekly write-ups

  Detailed summaries covering concepts, decisions, bugs and fixes: `app/WEEK5_SUMMARY.md`, `WEEK4_SUMMARY.md`, `WEEK6_SUMMARY.md`, `WEEK7_SUMMARY.md` and `WEEK8_SUMMARY.md`. The original Week 1 planning documents are in `docs/early-planning/`.