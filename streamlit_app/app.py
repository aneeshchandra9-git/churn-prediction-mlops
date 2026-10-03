"""
Churn prediction web app.

Form -> FastAPI /predict (prediction + contributions) -> Ollama (plain-English summary)
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ---------- Configuration (overridden by environment variables in Docker) ----------
API_URL = os.getenv("API_URL", "http://localhost:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Options in label-encoder order: a choice's position in the list = the code the API expects.
# Source: models/label_encoders.pkl
CATEGORY_OPTIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
    "StreamingTV": ["No", "Yes"],
    "StreamingMovies": ["No", "Yes"],
}

FEATURE_LABELS = {
    "age": "Age",
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Contract": "Contract type",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}

TOP_N = 5  # how many factors to list and send to the LLM

SYSTEM_PROMPT = """You describe which factors a churn model weighed, for a retention manager.

Rules:
1. Mention only the factors listed. Never add reasons, causes, feelings or industry knowledge.
2. Keep every factor in its group ("raises churn risk" or "lowers churn risk"). Never move a factor to the other group, even if it seems surprising.
3. Do not mention probabilities, percentages or the overall prediction.
4. Do not use the words "because", "suggests", "often", "typically" or "usually".
5. Write exactly 2 short sentences: one for the factors that raise the risk, one for the factors that lower it. If a group is "none", say that none of the listed factors do that.

Example input:
Raises churn risk: Contract type = Month-to-month; No tech support
Lowers churn risk: Age = 45

Example output:
In the model's calculation, a month-to-month contract and having no tech support raise the churn risk. Being 45 years old lowers it."""


# ---------- Helper functions ----------
def readable_value(feature, value):
    """Turn a value from the API back into something a person understands."""
    if feature in CATEGORY_OPTIONS:
        return CATEGORY_OPTIONS[feature][int(value)]
    return f"{value:g}"


def describe(c):
    """Readable text for one contribution, e.g. 'No streaming movies' or 'Contract type = One year'."""
    feature, value = c["feature"], c["value"]
    if CATEGORY_OPTIONS.get(feature) == ["No", "Yes"]:
        prefix = "Has" if int(value) == 1 else "No"
        return f"{prefix} {FEATURE_LABELS[feature].lower()}"
    return f"{FEATURE_LABELS[feature]} = {readable_value(feature, value)}"


def get_prediction(payload):
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def build_prompt(contributions):
    """Factors only. The LLM never sees the probability, so it can't misstate it."""
    top = contributions[:TOP_N]
    raises = [describe(c) for c in top if c["contribution"] > 0]
    lowers = [describe(c) for c in top if c["contribution"] <= 0]
    return (
        f"Raises churn risk: {'; '.join(raises) or 'none'}\n"
        f"Lowers churn risk: {'; '.join(lowers) or 'none'}"
    )


def get_explanation(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ---------- Page ----------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's details to predict churn risk and see why.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=5.0)
        total = st.number_input("Total charges", min_value=0.0, value=840.0, step=50.0)
    with col2:
        choices = {
            feature: st.selectbox(FEATURE_LABELS[feature], options)
            for feature, options in CATEGORY_OPTIONS.items()
        }
    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    payload = {
        "age": age,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    for feature, label in choices.items():
        payload[feature] = CATEGORY_OPTIONS[feature].index(label)

    try:
        result = get_prediction(payload)
    except requests.RequestException as e:
        st.error(f"Could not reach the prediction API at {API_URL}: {e}")
        st.stop()

    prediction = result["churn_prediction"]
    probability = result["churn_probability"]
    contributions = result["contributions"]

    # --- Prediction ---
    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"Likely to churn: {probability:.1%} probability")
    else:
        st.success(f"Likely to stay: {probability:.1%} churn probability")
    st.progress(probability)

    # --- Chart: biggest influence first, coloured by direction ---
    st.subheader("What drove this prediction")
    chart_data = pd.DataFrame(
        {
            "Factor": [FEATURE_LABELS[c["feature"]] for c in contributions],
            "Push towards churn": [c["contribution"] for c in contributions],
        }
    )
    chart_data["Direction"] = [
        "Raises risk" if v > 0 else "Lowers risk" for v in chart_data["Push towards churn"]
    ]
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Push towards churn:Q", title="Push towards churn"),
            y=alt.Y("Factor:N", sort=None, title=None),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Raises risk", "Lowers risk"],
                    range=["#e4572e", "#4caf50"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["Factor", alt.Tooltip("Push towards churn:Q", format=".4f")],
        )
    )
    st.altair_chart(chart)

    # --- Key factors: generated by code from the contributions, always accurate ---
    st.subheader("Key factors")
    for c in contributions[:TOP_N]:
        arrow = "🔺 raises" if c["contribution"] > 0 else "🔻 lowers"
        st.markdown(f"- **{describe(c)}**: {arrow} churn risk")

    # --- AI summary: code writes the numbers, the LLM only rewords the factors ---
    st.subheader("AI summary")
    with st.spinner("Writing a summary with the local LLM..."):
        try:
            outcome = "likely to churn" if prediction == 1 else "likely to stay"
            first_sentence = (
                f"The model predicts this customer is {outcome}, "
                f"with a {probability:.1%} churn probability."
            )
            summary = get_explanation(build_prompt(contributions))
            st.write(f"{first_sentence} {summary}")
            st.caption(f"Generated by {OLLAMA_MODEL} running locally. Check it against the key factors above.")
        except requests.RequestException as e:
            st.warning(f"The prediction worked, but the summary service is unavailable: {e}")

    st.info(
        "This describes what the model calculated, not proven causes. "
        "The model is weak (ROC-AUC ≈ 0.46), so treat the result as illustrative."
    )