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

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")



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


def evaluate_all_models(model_paths: dict, x_data, y_true) -> pd.DataFrame:
    rows = []
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            continue
        model = load_model(model_path)
        metrics, _ = compute_metrics(model, x_data, y_true)
        rows.append({"Model": model_name, **metrics})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)


def draw_confusion_matrix(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Benign (B)", "Malignant (M)"],
        cmap="Greens",
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)


def prepare_eval_data(user_df: pd.DataFrame, feature_cols: list):
    if "diagnosis" not in user_df.columns:
        st.error("Uploaded CSV must include a diagnosis column with M/B labels.")
        st.stop()

    missing = [c for c in feature_cols if c not in user_df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

    x_user = user_df[feature_cols]
    y_mapped = user_df["diagnosis"].map({"M": 1, "B": 0})
    if y_mapped.isna().any():
        st.error("Diagnosis column must contain only M or B values.")
        st.stop()

    y_user = y_mapped.astype(int)
    return x_user, y_user


def main():
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "model"
    test_data_path = project_root / "test_data.csv"
    st.title("Machine Learning Assignment 2")
    st.subheader("Breast Cancer Wisconsin Classification")
    st.write("This app compares 5 classification models on test data.")

    model_paths = get_model_paths(model_dir)

    st.markdown("### 1) Upload test data (CSV)")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

    expected_feature_file = model_dir / "feature_columns.json"
    if not expected_feature_file.exists():
        st.error("feature_columns.json missing. Run model/train_models.py first.")
        st.stop()

    with open(expected_feature_file, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    elif test_data_path.exists():
        user_df = load_reference_data(test_data_path)
        st.info("No file uploaded. Using local test_data.csv by default.")
    else:
        st.error("test_data.csv not found.")
        st.stop()

    x_user, y_user = prepare_eval_data(user_df, feature_cols)

    st.markdown("### Dataset Information")
    st.write("**Training Dataset:** 569 instances | **Test Dataset:** 114 instances (20% split)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Test Data Stats:**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Test Instances", len(user_df))
        with col_t2:
            st.metric("Features", len(feature_cols))
    
    with col2:
        st.write("**Class Distribution (Test):**")
        class_dist = y_user.value_counts()
        st.write(f"• Benign: {class_dist.get(0, 0)}")
        st.write(f"• Malignant: {class_dist.get(1, 0)}")

    st.markdown("### 2) Overall model comparison")
    overall_df = evaluate_all_models(model_paths, x_user, y_user)
    if overall_df.empty:
        st.warning("No model files found in model folder.")
        st.stop()

    overall_df.index = range(1, len(overall_df) + 1)
    overall_df.index.name = "S.No"
    st.table(overall_df)

    st.markdown("### 3) Select model")
    # Default: select first model by default (Logistic Regression)
    # User can click Cancel button below to deselect and choose manually
    default_model_index = 0
    model_name = st.selectbox("Choose a model", list(model_paths.keys()), index=default_model_index)
    
    # Cancel button to allow user to deselect the default model
    if st.button("Cancel Model Selection", key="cancel_model"):
        model_name = None
        st.info("Model selection cancelled. Please select a model to continue.")
        st.stop()

    model_path = model_paths[model_name]
    
    if model_name is None:
        st.error("Please select a model to continue.")
        st.stop()
    
    if not model_path.exists():
        st.error(f"Model file missing: {model_path.name}. Run model/train_models.py first.")
        st.stop()

    model = load_model(model_path)
    metrics, y_pred = compute_metrics(model, x_user, y_user)

    st.markdown("### 4) Metrics for selected model")
    selected_metrics_df = pd.DataFrame([metrics], index=[model_name])
    selected_metrics_df.index.name = "Model"
    st.table(selected_metrics_df)

    st.markdown("### 5) Confusion matrix")
    draw_confusion_matrix(y_user, y_pred, model_name)

    st.markdown("### 6) Classification report")
    report_dict = classification_report(
        y_user,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)

    # Accuracy is a single value in sklearn report. Clean table to avoid repeated display.
    if "accuracy" in report_df.index:
        accuracy_value = float(report_df.loc["accuracy", "precision"])
        report_df.loc["accuracy", "precision"] = pd.NA
        report_df.loc["accuracy", "recall"] = pd.NA
        report_df.loc["accuracy", "f1-score"] = round(accuracy_value, 4)
        report_df.loc["accuracy", "support"] = int(len(y_user))

    report_df.index.name = "Label"
    st.table(report_df)

    st.markdown("### 7) Compare selected models with confusion matrix")
    selected_models = st.multiselect(
        "Choose models to compare",
        list(model_paths.keys()),
        default=list(model_paths.keys()),
    )

    comparison_rows = []
    for selected_model in selected_models:
        selected_path = model_paths[selected_model]
        if not selected_path.exists():
            st.warning(f"Model file missing for {selected_model}")
            continue

        selected_loaded_model = load_model(selected_path)
        selected_metrics, selected_pred = compute_metrics(selected_loaded_model, x_user, y_user)
        comparison_rows.append({"Model": selected_model, **selected_metrics})

        st.markdown(f"**{selected_model}**")
        selected_compare_df = pd.DataFrame([selected_metrics], index=[selected_model])
        selected_compare_df.index.name = "Model"
        st.table(selected_compare_df)
        draw_confusion_matrix(y_user, selected_pred, selected_model)

    if len(comparison_rows) >= 2:
        comparison_df = pd.DataFrame(comparison_rows)
        ranked_df = comparison_df.sort_values(
            by=["Accuracy", "AUC", "F1", "MCC", "Precision", "Recall"],
            ascending=False,
        ).reset_index(drop=True)
        ranked_df.index = range(1, len(ranked_df) + 1)
        ranked_df.index.name = "Rank"
        st.success(f"Better model among selected: {ranked_df.iloc[0]['Model']}")
        st.caption("Ranking rule used: Accuracy, then AUC, then F1, then MCC, then Precision, then Recall.")
        st.table(ranked_df)


if __name__ == "__main__":
    main()
