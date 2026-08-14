import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")


@st.cache_data
def load_reference_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def get_model_paths(model_dir: Path) -> dict:
    return {
        "Logistic Regression": model_dir / "logistic_regression.pkl",
        "Decision Tree": model_dir / "decision_tree.pkl",
        "KNN": model_dir / "knn.pkl",
        "Naive Bayes": model_dir / "naive_bayes.pkl",
        "Random Forest": model_dir / "random_forest.pkl",
    }


def compute_metrics(model, x_data, y_true):
    y_pred = model.predict(x_data)
    y_prob = model.predict_proba(x_data)[:, 1]

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }, y_pred


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    metrics_path = project_root / "model_metrics.csv"
    test_data_path = project_root / "test_data.csv"

    st.title("Breast Cancer Wisconsin - ML Classification App")
    st.write("Try different models and compare their performance on test data.")

    st.subheader("1) Overall model comparison")
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("model_metrics.csv not found. Run model/train_models.py first.")

    st.subheader("2) Upload test data (CSV)")
    st.caption("Upload the provided test_data.csv or similar file with diagnosis column.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
    elif test_data_path.exists():
        user_df = load_reference_data(test_data_path)
        st.info("No file uploaded. Using local test_data.csv by default.")
    else:
        st.stop()

    expected_feature_file = model_dir / "feature_columns.json"
    if not expected_feature_file.exists():
        st.error("feature_columns.json missing. Run model/train_models.py first.")
        st.stop()

    with open(expected_feature_file, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    if "diagnosis" not in user_df.columns:
        st.error("Uploaded CSV must include a diagnosis column with M/B labels.")
        st.stop()

    missing = [c for c in feature_cols if c not in user_df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

    x_user = user_df[feature_cols]
    y_user = user_df["diagnosis"].map({"M": 1, "B": 0}).astype(int)

    st.subheader("3) Select model")
    model_paths = get_model_paths(model_dir)
    model_name = st.selectbox("Choose a model", list(model_paths.keys()))

    model_path = model_paths[model_name]
    if not model_path.exists():
        st.error(f"Model file missing: {model_path.name}. Run model/train_models.py first.")
        st.stop()

    model = load_model(model_path)
    metrics, y_pred = compute_metrics(model, x_user, y_user)

    st.subheader("4) Metrics for selected model")
    metric_cols = st.columns(6)
    for idx, (metric_name, metric_value) in enumerate(metrics.items()):
        metric_cols[idx].metric(metric_name, metric_value)

    st.subheader("5) Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_user,
        y_pred,
        display_labels=["Benign (B)", "Malignant (M)"],
        cmap="Blues",
        ax=ax,
    )
    st.pyplot(fig)

    st.subheader("6) Classification report")
    report = classification_report(y_user, y_pred, target_names=["Benign", "Malignant"])
    st.code(report)


if __name__ == "__main__":
    main()
