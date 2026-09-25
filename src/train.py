from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"

METADATA_PATH = ROOT / "models" / "metadata.json"


FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_calls",
    "usage_hours",
    "contract_months",
    "late_payments"
]

TARGET = "churn"


def load_data():
    """Load the training dataset."""

    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    return X, y


def build_pipeline():
    """Create preprocessing and model pipeline."""

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


def main():

    # Load data
    X, y = load_data()

    # Create train/test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create model pipeline
    pipeline = build_pipeline()

    # Train model
    pipeline.fit(
        X_train,
        y_train
    )

    # Evaluate model
    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    print("\nModel performance")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Create model directory if necessary
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save complete pipeline
    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    # Save useful information about model
    metadata = {
        "model_type": "LogisticRegression",
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics
    }

    with open(
        METADATA_PATH,
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"Metadata saved to: {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()