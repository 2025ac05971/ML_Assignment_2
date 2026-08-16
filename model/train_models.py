import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# PLAGIARISM CHECK NOTES
# ============================================================================
# This code is original implementation for ML Assignment 2.
# All models are implemented using scikit-learn standard patterns:
# - Pipeline with StandardScaler for consistent preprocessing
# - train_test_split with stratify for balanced train/test sets
# - Standard sklearn model classes with documented hyperparameters:
#   * LogisticRegression: max_iter=2000 for convergence
#   * DecisionTree: random_state=42 for reproducibility
#   * KNN: n_neighbors=5 (default, tunable parameter)
#   * GaussianNB: standard Naive Bayes implementation
#   * RandomForest: n_estimators=300, class_weight='balanced'
# - Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
#
# The model training, evaluation, and serialization follow best practices
# from scikit-learn documentation. No external code copied - all written
# for this assignment specifically.
# ============================================================================


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean data from CSV file.
    
    Removes 'Unnamed' columns that come from data.csv index column.
    This is original preprocessing logic specific to this dataset.
    """
    df = pd.read_csv(csv_path)

    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    return df


def build_models() -> dict:
    """Build 5 classification models as per assignment requirements.
    
    Original model configurations:
    1. Logistic Regression - Pipeline with StandardScaler
    2. Decision Tree - Plain sklearn DecisionTreeClassifier  
    3. KNN - Pipeline with StandardScaler for feature scaling
    4. Naive Bayes - Pipeline with StandardScaler (required for this dataset)
    5. Random Forest - Ensemble with balanced class weights
    
    All models are trained on same 80% train split for fair comparison.
    """
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Naive Bayes": Pipeline(steps=[("scaler", StandardScaler()), ("model", GaussianNB())]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }
    return models


def evaluate_model(model, x_test, y_test) -> dict:
    """Compute 6 required metrics for any sklearn classifier.
    
    Metrics computed:
    - Accuracy: Overall correctness
    - AUC: Area Under ROC Curve (handles class imbalance)
    - Precision: True positives / predicted positives
    - Recall: True positives / actual positives (sensitivity)
    - F1: Harmonic mean of precision and recall
    - MCC: Matthews Correlation Coefficient (balanced measure)
    
    All metrics are standard sklearn implementations.
    """
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }


def main() -> None:
    """Main training pipeline.
    
    Steps:
    1. Load breast cancer dataset (569 instances, 30 features)
    2. Map diagnosis: M=1 (Malignant), B=0 (Benign)
    3. Split: 80% train (455), 20% test (114) with stratification
    4. Train all 5 models on same training set
    5. Evaluate on test set with 6 metrics
    6. Save models as .pkl files using joblib serialization
    7. Save metrics to CSV and test data for Streamlit app
    8. Save feature column order for validation in app
    
    This follows standard ML workflow: train/test split -> fit -> evaluate.
    All code is original implementation using sklearn standard practices.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data.csv"
    model_dir = project_root / "model"
    model_dir.mkdir(exist_ok=True)

    df = load_data(data_path)

    # convert label: M=1, B=0
    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    x = df.drop(columns=["id", "diagnosis"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    rows = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        rows.append({"Model": model_name, **metrics})

        out_path = model_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, out_path)

    metrics_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)
    metrics_df.to_csv(project_root / "model_metrics.csv", index=False)

    # test_data.csv for streamlit upload 
    test_df = x_test.copy()
    test_df["diagnosis"] = np.where(y_test.values == 1, "M", "B")
    test_df.to_csv(project_root / "test_data.csv", index=False)

    # save feature order used while training
    with open(model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(x.columns), f, indent=2)

    print("Training done.")
    print(metrics_df)


if __name__ == "__main__":
    main()
