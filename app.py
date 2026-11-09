"""
Streamlit dashboard for the Student Performance Prediction System.

Run the training script first:
    python train_model.py
Then:
    streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_model.pkl"
DATA_PATH = ROOT / "student_data.csv"
METRICS_PATH = ROOT / "evaluation_results.json"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Student Performance Prediction System")
st.caption("Classroom prototype for exploring possible academic risk. "
           "Predictions are not diagnoses or final decisions.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("The model or dataset file is missing.")
    st.info("Run `python train_model.py` in the project folder first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
data = pd.read_csv(DATA_PATH)

if "G3" not in data.columns:
    st.error("The dataset does not contain the required G3 target column.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{len(data):,}")
c2.metric("Average final grade", f"{data['G3'].mean():.2f}")
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.subheader("Model evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision (At risk)", f"{metrics['precision_at_risk']:.3f}")
    m3.metric("Recall (At risk)", f"{metrics['recall_at_risk']:.3f}")
    m4.metric("F1-score (At risk)", f"{metrics['f1_at_risk']:.3f}")

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns used during training. "
         "Do not include G3.")
predictors = data.drop(columns=["G3"])
template = predictors.head(5)
st.download_button(
    "Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required_columns = list(model.feature_names_in_)
        missing = [column for column in required_columns
                   if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif new_data.empty:
            st.warning("The uploaded CSV has no student records.")
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records",
                      int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )
    except Exception as error:
        st.error(f"Could not process the uploaded file: {error}")

st.divider()
st.caption("Responsible use: Use predictions only as a prompt for supportive "
           "review by faculty. Do not use this prototype as the sole basis "
           "for decisions about students.")
