# Week 7 Summary — Monitoring & Orchestration

## Goal

Weeks 5–6 put the churn model behind a FastAPI service inside Docker. Week 7 answers two production questions:

1. **Is the service healthy, and what is the model predicting?** → Monitoring with **Prometheus + Grafana**
2. **How does the model get retrained without manual steps?** → Orchestration with **Airflow**

---

## Architecture

```
                      docker compose  (shared network: mlops-project_default)
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                                                                              │
 │  churn-api :8000  ──/metrics──►  prometheus :9090  ──PromQL──►  grafana :3000│
 │       ▲                          (scrapes every 15s)           (4 panels)    │
 │       │ model files baked in at build time                                   │
 │       │                                                                      │
 │  airflow :8080  ──writes──►  models/  +  mlflow.db   (whole project mounted) │
 │  (weekly retraining DAG)                                                     │
 └──────────────────────────────────────────────────────────────────────────────┘
```

Containers reach each other by **service name** (`api:8000`, `prometheus:9090`), never `localhost` — inside a container, `localhost` means that container itself.

---

## Files created or changed this week

| File | Purpose |
|---|---|
| `app/main.py` | Added `/metrics` endpoint + 3 Prometheus metrics |
| `requirements-api.txt` | Added `prometheus-client==0.26.0` |
| `docker-compose.yml` | Runs api, prometheus, grafana, airflow together |
| `monitoring/prometheus.yml` | Scrape config: target `api:8000`, every 15s |
| `monitoring/grafana/provisioning/datasources/datasource.yml` | Prometheus data source as code |
| `monitoring/grafana/provisioning/dashboards/dashboards.yml` | Tells Grafana which folder holds dashboards |
| `monitoring/grafana/dashboards/churn-api-monitoring.json` | The exported dashboard |
| `requirements-airflow.txt` | ML libraries for the Airflow image (pinned to venv versions) |
| `Dockerfile.airflow` | `apache/airflow:3.3.1-python3.12` + ML libraries |
| `dags/churn_retraining_dag.py` | The retraining pipeline |

---

## Part 1 — Instrumenting the API

### Concepts
- **Pull-based monitoring:** the API doesn't send metrics anywhere. It exposes a plain-text `/metrics` page, and Prometheus **scrapes** (polls) it on a schedule.
- **Metric types:**
  - **Counter** — only goes up (total requests). Use `rate()` to turn it into "per second".
  - **Histogram** — counts observations into buckets (latency under 0.1s, under 0.25s, …). Buckets are **cumulative**. Lets you estimate percentiles.
  - **Gauge** — goes up or down; holds only the **latest** value.

### Our three metrics

| Python variable | Prometheus name | Type | Meaning |
|---|---|---|---|
| `REQUEST_COUNT` | `predict_requests_total{status=...}` | Counter | Predictions served, labelled success/error |
| `REQUEST_LATENCY` | `predict_request_latency_seconds_bucket{le=...}` | Histogram | How long `/predict` took |
| `CHURN_PROBABILITY` | `churn_prediction_probability` | Gauge | Most recent predicted probability |

A histogram also produces `_count` (number of requests) and `_sum` (total time). `_sum / _count` = average latency.

**Labelled counters don't exist until first used.** `predict_requests_total{status="success"}` only appears after the first successful request.

---

## Part 2 — Prometheus

### `docker-compose.yml` ideas
- `build: .` builds from the local `Dockerfile`; `image:` uses a pre-built image.
- `ports: "host:container"` maps a port on Windows to one inside the container.
- **Bind mount** (`./file:/path`) — a file/folder from my machine appears inside the container. Used for config I edit.
- **Named volume** (`grafana-data:/var/lib/grafana`) — storage Docker manages; survives container restarts. Deleted by `docker compose down -v`.
- `depends_on` controls **start order only**, not readiness.
- Images are **pinned** (`prom/prometheus:v3.5.0`, `grafana/grafana:12.1.0`) for reproducibility.

### `prometheus.yml`
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: "churn-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```
Prometheus automatically adds `job="churn-api"` and `instance="api:8000"` labels to every metric.

### Verification
- `localhost:9090/targets` → `churn-api` **UP**
- Query `predict_requests_total` → `{status="success"} 3` after 3 requests

---

## Part 3 — Grafana

Grafana stores no data. It sends PromQL to Prometheus and draws the answers.

### Dashboard: Churn API Monitoring

| Panel | Layer | Query | Visualization |
|---|---|---|---|
| Total Predictions | Service | `sum(predict_requests_total)` | Stat |
| Request Rate (req/s) | Service | `sum(rate(predict_requests_total[1m]))` | Time series |
| p95 Latency | Service | `histogram_quantile(0.95, sum by (le) (rate(predict_request_latency_seconds_bucket[5m])))` | Time series (unit: seconds) |
| Latest Churn Probability | **Model** | `churn_prediction_probability` | Gauge, 0–1, red at ≥ 0.5 |

### PromQL explained
- `[1m]` — a **range**: all samples from the last minute (~4 at a 15s scrape interval).
- `rate(counter[1m])` — average increase per second over that minute. Rises with traffic, falls to 0 when idle.
- `sum(...)` — combines series (e.g. success + error) into one.
- `sum by (le)` — combines series but **keeps** the bucket label, which `histogram_quantile` needs.
- `histogram_quantile(0.95, ...)` — estimates the time 95% of requests finish within.

### Why p95, not average
Observed buckets: 4 requests < 10 ms, 2 requests 100–250 ms. Average = 0.356 s / 6 ≈ 59 ms — which describes **no actual request**. p95 exposes the slow tail.

### Behaviours observed (not bugs)
- **Cold start:** the first request into a fresh container took ~1 s; later ones < 10 ms.
- **`rate()` needs two samples:** a counter created at 1 shows no rate for its first request.
- **p95 shows "No data"** when there were no requests in the last 5 minutes.
- **Gauge clustering:** customers A/B/C gave 52.5% / 49.4% / 50.2% — the Week 4 weak-discrimination finding, now visible live.

### Provisioning (configuration as code)
Without provisioning, the dashboard lived only in the `grafana-data` volume — a fresh clone would get an empty Grafana.
- Exported dashboard JSON → `monitoring/grafana/dashboards/`
- `datasource.yml` creates the Prometheus data source with **the same `uid`** (`fg03avbmdww74b`) the panels reference
- `dashboards.yml` tells Grafana to load JSON files from `/etc/grafana/dashboards`
- **Tested** with `docker compose down -v` → `up -d`: dashboard and data source reappeared automatically
- `access: proxy` — the Grafana server fetches data (the browser can't resolve `prometheus`)
- `allowUiUpdates: true` — edits in the browser are **not** written back to the JSON; re-export after changes

---

## Part 4 — Airflow

### Concepts
- **Orchestration:** running multi-step workflows automatically, in order, on a schedule.
- **Task:** one unit of work. **DAG:** tasks + dependencies (directed, no loops).
- **Scheduler** starts due tasks; **web UI** shows runs (green = success, red = failed) and logs.
- **Schedule** `@weekly` = cron `0 0 * * 0` (Sunday 00:00 UTC).
- **Retries**, **clearing** a task (re-run it using upstream outputs), **paused** by default.

### The DAG: `churn_model_retraining`

```
prepare_data ──► train_model ──► evaluate_and_log ──► promote_model
```

| Task | Does |
|---|---|
| `prepare_data` | Reuses `DataPipeline.run_pipeline()`; saves train/test CSVs to `data/processed/airflow/`; saves scaler + label encoders from the **same run** to `models/candidate/` |
| `train_model` | SMOTE (training set only) → `LogisticRegression(C=0.001, solver="liblinear", max_iter=200, random_state=42)` → `candidate/model.pkl` |
| `evaluate_and_log` | Metrics on the original **imbalanced** test set → MLflow experiment `week7_airflow_retraining` |
| `promote_model` | **Quality gate:** fails if F1 < 0.30; backs up current files to `models/previous/`; copies candidate files into `models/` |

### Design decisions
- **Airflow 3 TaskFlow style:** `from airflow.sdk import dag, task`. Passing one task's return value into the next creates the dependency.
- **Heavy imports inside tasks:** Airflow re-parses DAG files every few seconds; parse took 0.26 s.
- **Tasks share files, not memory.** XCom passes only small values (row counts, metrics dict). Values are converted to plain `float` so they serialise to JSON.
- **Absolute paths** from `PROJECT_DIR = /opt/airflow/project`.
- **`catchup=False`:** no back-filling of missed weekly runs.
- **Only params/metrics logged to MLflow,** not model artifacts — artifact paths would point inside the container.
- **Quality gate fails loudly** (red task) instead of silently skipping. `retries=0` on that task: retrying can't change the F1.

### Container setup
- `Dockerfile.airflow`: `FROM apache/airflow:3.3.1-python3.12` — same Airflow as the venv, Python 3.12 for the `.pkl` files.
- `pip install "apache-airflow==${AIRFLOW_VERSION}" -r requirements-airflow.txt` — pins Airflow so ML libraries can't change it.
- **Built alone first** (`docker build -f Dockerfile.airflow -t churn-airflow .`) to catch dependency conflicts — none.
- **Layer caching:** first build 311 s, rebuild 3 s.
- `command: standalone` — all Airflow components in one container (dev only).
- `AIRFLOW__CORE__LOAD_EXAMPLES: "False"`, `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS: "True"` (no login — **local only, never on a reachable server**).
- Mounts: `./dags → /opt/airflow/dags`, `. → /opt/airflow/project`.

### Results

| Metric | Week 4a (manual) | Week 7 (Airflow) |
|---|---|---|
| F1 | 0.3211 | 0.3211 |
| Recall | 0.4569 | 0.4569 |
| ROC-AUC | ≈ 0.455 | 0.4550 |

The automated pipeline **reproduces the manual model exactly** — the point is reproducibility, not a better model.

The promoted model file is 1387 bytes vs the original 1095: trained on a DataFrame, so it stores `feature_names_in_`. Same coefficients, extra metadata. Verified inside the API container with `docker compose exec api ls -l models`.

---

## Problems hit and how they were fixed

| Problem | Cause | Fix |
|---|---|---|
| API container would have crashed on startup | `prometheus-client` installed in venv but missing from `requirements-api.txt` | Added `prometheus-client==0.26.0` |
| `failed to connect to the docker API` | Docker Desktop not running | Start Docker Desktop, wait for "Engine running" |
| Grafana "Please enter a valid URL" | Field was empty; grey `http://localhost:9090` was placeholder text | Typed `http://prometheus:9090` |
| Dashboard JSON file missing | File never saved (VS Code subfolder issue) | Created with `New-Item`, opened with `code <path>` |
| `Rename-Item` refused | Second argument must be a new **name**, not a path | `Rename-Item models\previous previous_original` |
| `PermissionError` in `promote_model` | `shutil.copy` = copy contents **+ permission bits**; Linux can't set permissions on Windows-mounted files | `shutil.copyfile` (contents only), with a comment explaining why |
| MLflow experiment looked empty | MLflow 3 UI was in **GenAI** mode | Switched to **Model training** |

---

## Known limitations (and how I'd improve them)

1. **Scaler data leakage:** `DataPipeline` fits the scaler on the full dataset before splitting. Fix: split first, fit scaler on training data only.
2. **Weak model:** ROC-AUC ≈ 0.455; probabilities cluster around 0.5. Needs better features or data.
3. **Models baked into the API image:** every retrain needs `docker compose up -d --build api`. Alternative: mount `models/` and restart the API — faster updates, less self-contained images.
4. **Airflow standalone + SQLite:** development only. Production uses separate scheduler/API containers and PostgreSQL.
5. **No auth on Airflow** (`ALL_ADMINS`) — local only.
6. **Airflow run history is not persisted** — lost on `docker compose down`. Models and MLflow metrics are kept.
7. **Gauge shows only the latest prediction.** A histogram of probabilities would show the distribution and support drift detection.
8. **Categorical inputs to the API are still numeric codes.** `label_encoders.pkl` is now saved — the next step is decoding human-readable values in the API.

---

## Useful commands

```powershell
docker compose up -d                      # start / apply compose changes
docker compose up -d --build api          # rebuild + restart one service
docker compose ps                         # what's running
docker compose logs airflow --tail 40     # recent logs of one service
docker compose exec api ls -l models      # run a command inside a container
docker compose down                       # stop and remove containers
docker compose down -v                    # ...and delete named volumes
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |

---

## Interview talking points

- *"I instrumented the FastAPI service with Prometheus counters, a latency histogram and a prediction gauge, and built a Grafana dashboard that monitors both service health and model output."*
- *"I track p95 latency rather than the average, because averages hide slow outliers — in my own data the average was 59 ms while two cold-start requests took ~200 ms."*
- *"The Grafana data source and dashboard are provisioned from files in the repo, so `docker compose up` on a fresh machine gives the full dashboard."*
- *"Airflow orchestrates weekly retraining: data prep, SMOTE + training, evaluation logged to MLflow, and promotion behind an F1 quality gate with automatic backup of the previous model."*
- *"The automated pipeline reproduced my manual model's metrics exactly, which validated it."*
- *"I know the limitations: scaler leakage, the weak model, standalone Airflow being dev-only."*

---

## Week 8 preview

Web application + LLM integration: a user-facing app that calls the `/predict` API, with an LLM explaining predictions in plain language.
