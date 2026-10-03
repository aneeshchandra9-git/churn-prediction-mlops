"""
Tests for the churn prediction API.

Run from the project root:  pytest tests/ -v
"""
import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Corrected Customer A from Week 8 (valid codes: Yes = 1)
VALID_CUSTOMER = {
    "age": 28,
    "tenure": 2,
    "MonthlyCharges": 95.5,
    "TotalCharges": 191.0,
    "Contract": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert len(body["contributions"]) == 12


def test_contributions_reproduce_probability():
    """The Week 8 proof: intercept + contributions -> sigmoid = the API's probability."""
    body = client.post("/predict", json=VALID_CUSTOMER).json()
    score = body["intercept"] + sum(c["contribution"] for c in body["contributions"])
    recomputed = 1 / (1 + math.exp(-score))
    assert math.isclose(recomputed, body["churn_probability"], abs_tol=1e-9)


def test_invalid_category_code_is_rejected():
    bad = {**VALID_CUSTOMER, "StreamingMovies": 2}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_negative_tenure_is_rejected():
    bad = {**VALID_CUSTOMER, "tenure": -1}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_counts_predictions():
    client.post("/predict", json=VALID_CUSTOMER)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predict_requests_total" in response.text