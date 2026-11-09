"""
Student Performance Prediction System
Downloads the UCI Student Performance dataset, prepares a binary risk target,
trains a Random Forest pipeline, evaluates it, and saves the model.

Run:
    pip install -r requirements.txt
    python train_model.py
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"
MODEL_PATH = ROOT / "student_model.pkl"
METRICS_PATH = ROOT / "evaluation_results.json"

def main():
    print("Downloading UCI Student Performance dataset (ID 320)...")
    student = fetch_ucirepo(id=320)

    # The laboratory specifies the Mathematics records. The UCI interface
    # returns the dataset's feature and target tables.
    df = pd.concat(
        [student.data.features, student.data.targets],
        axis=1
    )

    if "G3" not in df.columns:
        raise ValueError(
            "The downloaded dataset does not contain G3. "
            "Check the UCI dataset response before training."
        )

    df = df.drop_duplicates().copy()
    df.to_csv(DATA_PATH, index=False)

    print("Rows and columns:", df.shape)
    print(df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    # 1 = AT RISK when G3 < 10; 0 = NOT AT RISK otherwise.
    y = (df["G3"] < 10).astype(int)

    # G3 is the target and must never be an input feature.
    X = df.drop(columns=["G3"])

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    # Stratification may fail only if a class has fewer than two records.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_at_risk": round(float(precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        )), 4),
        "recall_at_risk": round(float(recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        )), 4),
        "f1_at_risk": round(float(f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        )), 4),
        "confusion_matrix_labels": ["Not at risk", "At risk"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions,
            target_names=["Not at risk", "At risk"],
            zero_division=0,
            output_dict=True
        )
    }

    print("\nMODEL EVALUATION")
    print("Accuracy:", metrics["accuracy"])
    print("Precision (At risk):", metrics["precision_at_risk"])
    print("Recall (At risk):", metrics["recall_at_risk"])
    print("F1-score (At risk):", metrics["f1_at_risk"])
    print("\n", classification_report(
        y_test, predictions,
        target_names=["Not at risk", "At risk"],
        zero_division=0
    ))

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"\nSaved: {MODEL_PATH.name}")
    print(f"Saved: {METRICS_PATH.name}")
    print(f"Saved: {DATA_PATH.name}")

if __name__ == "__main__":
    main()
