# Week 8 Summary — Web Application + LLM Explanations

## Goal

Until Week 7, the only way to use the model was Swagger: raw JSON with coded values (`"Contract": 0`), returning a bare probability. Week 8 adds:

1. **A web app** (Streamlit) — readable dropdowns, a clear result, a chart of what drove it.
2. **Plain-English explanations** from a **local LLM** (Ollama + `llama3.2:3b`) — grounded in the model's real calculation, not invented.

---

## Final architecture

```
                         docker compose (shared network)
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  Browser ─► streamlit :8501 ──POST /predict──► churn-api :8000       │
 │                 │            ◄─ probability + contributions ─        │
 │                 │                                    │               │
 │                 └─ factors ─► ollama :11434          └─/metrics─►    │
 │                              (llama3.2:3b)          prometheus :9090 │
 │                                                           │          │
 │                                                     grafana :3000    │
 │                                                                      │
 │  airflow :8080 ─(weekly)─► retrains ─► models/ + mlflow.db           │
 └──────────────────────────────────────────────────────────────────────┘
```

Six services. Containers reach each other by **service name**; the Streamlit app is configured through **environment variables** so the same code runs locally and in Docker.

---

## Files created or changed

| File | Change |
|---|---|
| `app/main.py` | Returns per-feature **contributions** + intercept; `FEATURE_ORDER`; Pydantic **input validation** |
| `streamlit_app/app.py` | The web app |
| `requirements-streamlit.txt` | `streamlit==1.65.0`, `requests==2.34.2` |
| `Dockerfile.streamlit` | `python:3.12-slim` + Streamlit, headless |
| `docker-compose.yml` | Added `ollama` and `streamlit`; Grafana moved to new volume `grafana-storage`, plugin preinstall disabled |

---

## Part 1 — The local LLM (Ollama)

- **Why local:** a Claude Pro subscription does not include API access (the API is billed separately). Ollama runs open-source models locally — free, no keys.
- **Model choice:** `llama3.2:3b` (~2 GB download, ~3 GB in memory). Docker's memory limit is 5.6 GB, so **Airflow stays stopped** while Ollama runs.
- **Quantization:** 3 billion parameters stored at ~4 bits each, which is why it fits in ~2 GB.
- **Measured speed (CPU):** first load ~90–144 s (reading 2 GB from a Docker volume); generation ~3–7 s once loaded.
- **`OLLAMA_KEEP_ALIVE: "24h"`** keeps the model in memory instead of unloading after 5 minutes idle.
- **Image pinned** to `ollama/ollama:0.35.1` (checked with `ollama --version`); model stored in the `ollama-data` volume.

---

## Part 2 — Making the model explain itself

### Contributions (the model's "itemised receipt")
Logistic regression computes `score = intercept + Σ (coefficient × scaled feature)`, then `probability = sigmoid(score)`.
Each `coefficient × scaled value` is that feature's **contribution**: positive pushes towards churn, negative towards staying.

`/predict` now also returns `intercept` and `contributions` (sorted by absolute size). The existing fields are unchanged — a **backward-compatible** change.

**Verified exactly:** intercept + Σ contributions → sigmoid = **0.5250**, identical to the API's probability. The contributions are the model's complete, real calculation.

### Faithful ≠ true
The contributions are **faithful** (exactly what the model computed), but the model learned weak, sometimes counter-intuitive patterns — e.g. short tenure *lowers* churn risk, high monthly charges barely matter. Explainability exposed these. The app carries a disclaimer.

---

## Part 3 — A bug found on the way: invalid category codes

Printing `models/label_encoders.pkl` showed the six service features are **`0 = No, 1 = Yes` only**. Earlier test customers used `2` for "Yes" — an impossible value the model had never seen. The API accepted it silently because every field was `float`.

**Impact:** `StreamingMovies = 2` tripled its apparent influence (+0.104 vs +0.035 with the correct `1`).

**Fixes:**
1. **API validation** with Pydantic `Field(ge=..., le=...)`: categories are `int` with exact ranges; numeric fields `ge=0`. Invalid input → **422** with the exact field and reason (`"loc":["body","StreamingMovies"]`, `"Input should be less than or equal to 1"`).
2. **Dropdowns** in the app built from the real encoder lists — invalid codes can't be sent.

Note: 422s are rejected before `predict()` runs, so they don't appear in the Prometheus error counter.

### Real category codes

| Column | Codes |
|---|---|
| Contract | 0 Month-to-month, 1 One year, 2 Two year |
| InternetService | 0 DSL, 1 Fiber optic, 2 No |
| OnlineSecurity … StreamingMovies | 0 No, 1 Yes |

---

## Part 4 — The Streamlit app

- **Streamlit re-runs the whole script on each interaction**; inputs sit inside `st.form`, so it runs only on "Predict".
- **Page:** prediction (green/red), probability bar, **Altair** chart of contributions (sorted by size, red = raises, green = lowers), **Key factors** list, **AI summary**, disclaimer.
- **Graceful degradation:** if Ollama fails, the prediction still shows with a warning.
- **Config:** `API_URL` / `OLLAMA_URL` from environment variables, defaulting to `localhost`.

---

## Part 5 — Getting reliable explanations from a small LLM

| Version | Problem observed | Fix |
|---|---|---|
| 1 | Invented reasons ("financially strained"), general telecom knowledge, **flipped tenure's direction** | Factors pre-grouped into "raises" / "lowers"; numbered rules; **few-shot example**; temperature 0 |
| 2 | Called the churn probability a "stay" probability | **Code writes the first sentence** (prediction + probability); the LLM never sees numbers |
| 3 | Dropped the "No" in "Streaming movies = No" | Yes/No features phrased **negation-first**: "No streaming movies", "Has tech support" |
| 4 | Occasional "$" or small rewording | Accepted; the code-generated **Key factors** list is the source of truth |

**Lessons:**
- Instructions reduce hallucination; they don't eliminate it — small models follow rules weakly.
- Make the input **impossible to misread** rather than asking the model to try harder.
- **Don't ask an LLM to do what code can do exactly** (numbers, labels).
- Label AI text clearly and keep a deterministic source of truth beside it.

---

## Part 6 — Containerising the app

- `Dockerfile.streamlit`: same `python:3.12-slim` base as the API (layers reused); requirements installed before code is copied (layer caching).
- CMD flags: `--server.address=0.0.0.0` (reachable from outside the container), `--server.headless=true` (no browser / email prompt), `--browser.gatherUsageStats=false`.
- Compose sets `API_URL=http://api:8000` and `OLLAMA_URL=http://ollama:11434`; `app.py` is unchanged between local and Docker.
- The front end needs no ML libraries — the model stays behind the API.

---

## Part 7 — Incident: Grafana storage corruption

**Symptoms:** Grafana showed a blank error page; logs said `database disk image is malformed`. Prometheus had also stopped.

**Diagnosis:** after deleting the volume failed with `readdirent ... bad message`, it was clear a folder inside Docker's virtual disk was damaged at the filesystem level — most likely from the laptop sleeping or shutting down while containers were writing. Every restart reused the same damaged volume and crashed while scanning `/var/lib/grafana/plugins`.

**Recovery:**
1. `docker compose down` (no `-v`, keeping the Ollama model), `wsl --shutdown`, restart Docker Desktop → Prometheus came back.
2. Pointed Grafana at a **new volume** (`grafana-storage`) instead of the damaged one.
3. Set `GF_PLUGINS_PREINSTALL_DISABLED: "true"` — our panels are built in, and it avoids downloading plugins on each fresh start.
4. Grafana logged `finished to provision dashboards`: **dashboard and data source rebuilt from files**.

**Takeaway:** provisioning-as-code turned a corrupted database into a 2-minute recovery. Prevention: run `docker compose stop` before closing the laptop.

The old damaged volume `mlops-project_grafana-data` is unused and can be removed after a Docker Desktop disk cleanup.

---

## Results

- App predictions match the API exactly; contributions reproduce the probability exactly.
- AI summaries matched every factor and direction across the final test runs.
- Grafana showed the app's traffic: Total Predictions 4, Request Rate bumps, Latest Churn Probability 49.6% = last app prediction.
- p95 ≈ 460–480 ms with only 4 requests after a restart — dominated by the cold-start request.

---

## Known limitations

1. **Weak model** (ROC-AUC ≈ 0.455): explanations are faithful to the model, not proof of real causes.
2. **Small local LLM** occasionally adds minor details (e.g. "$"); the key-factors list is authoritative.
3. **CPU-only LLM**: slow first load after a restart (~1.5–2.5 min).
4. **Memory**: Ollama + Airflow together exceed Docker's 5.6 GB; run one at a time.
5. **Rejected requests (422) aren't counted** in Prometheus metrics.
6. **No authentication** on the app, Airflow or Ollama — local use only.
7. Inherited from earlier weeks: scaler leakage, models baked into the API image, standalone Airflow.

---

## Useful commands

```powershell
docker compose up -d api prometheus grafana ollama streamlit   # everything except Airflow
docker compose up -d --build streamlit                         # after editing app.py
docker compose exec ollama ollama pull llama3.2:3b             # download a model
docker compose config --services                               # validate compose file
docker compose stop                                            # before closing the laptop!
```

| Service | URL |
|---|---|
| Web app | http://localhost:8501 |
| API (Swagger) | http://localhost:8000/docs |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Ollama | http://localhost:11434 |
| Airflow (when running) | http://localhost:8080 |

---

## Interview talking points

- *"I added a Streamlit front end and a local LLM that explains each prediction. The explanations are grounded in the model's exact per-feature contributions, which I verified reproduce the probability exactly."*
- *"My first LLM explanations invented reasons and even contradicted the model. I fixed it by restructuring the input — grouping factors by direction, a few-shot example, and having code write all numbers — rather than just asking the model to behave."*
- *"Explainability exposed both a data bug — my API silently accepted impossible category codes — and counter-intuitive model behaviour. I added Pydantic validation so invalid input now gets a clear 422."*
- *"When Grafana's storage was corrupted, I recovered in minutes because the dashboard and data source were provisioned as code."*

---

## Project status

**All 8 weeks complete.** Data pipeline → experiments → SMOTE & tuning → FastAPI → Docker → monitoring → orchestration → web app with grounded LLM explanations.