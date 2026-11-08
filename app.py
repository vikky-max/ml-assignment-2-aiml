"""Streamlit interface for evaluating five trained classifiers."""
from pathlib import Path
import subprocess, sys
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classification Model Evaluator", page_icon="🧪", layout="wide")
st.title("Wisconsin Breast Cancer — Classification Model Evaluator")
st.write("Upload labelled test data, select a classifier, and inspect its performance.")
st.info("Target encoding: 0 = malignant, 1 = benign. CSV must include all 30 feature columns and a target column.")

# Train models automatically on first deployment if model files are not present.
if not all((MODEL_DIR / filename).exists() for filename in FILES.values()):
    with st.spinner("Training the five classifiers for first-time setup..."):
        result = subprocess.run([sys.executable, str(ROOT / "train_models.py")], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("Model training failed: " + result.stderr)
        st.stop()
    st.success("All five models trained successfully.")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
choice = st.selectbox("Choose model", list(FILES))
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        if "target" not in data.columns:
            st.error("CSV must include a target column.")
            st.stop()
        y = data["target"].astype(int)
        X = data.drop(columns=["target"])
        model = joblib.load(MODEL_DIR / FILES[choice])
        expected = list(model.feature_names_in_)
        missing = sorted(set(expected) - set(X.columns))
        extra = sorted(set(X.columns) - set(expected))
        if missing or extra:
            st.error(f"Feature mismatch. Missing: {missing}; unexpected: {extra}")
            st.stop()
        X = X[expected]
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, prob),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1 Score": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        }
        st.subheader(choice + " — Evaluation Metrics")
        columns = st.columns(3)
        for i, (name, value) in enumerate(metrics.items()):
            columns[i % 3].metric(name, f"{value:.4f}")
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1]), index=["Actual malignant", "Actual benign"], columns=["Predicted malignant", "Predicted benign"]))
        st.subheader("Classification Report")
        report = classification_report(y, pred, labels=[0, 1], target_names=["malignant", "benign"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4))
        st.caption(f"Evaluated {len(data)} records.")
    except Exception as error:
        st.error(f"Unable to evaluate uploaded file: {error}")
