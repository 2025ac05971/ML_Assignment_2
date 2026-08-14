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


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Kaggle file has an extra empty column at the end in some versions
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    return df


def build_models() -> dict:
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
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        # fallback if model doesn't expose predict_proba
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

    # test_data.csv for streamlit upload (includes label column)
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
