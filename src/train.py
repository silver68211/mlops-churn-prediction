from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

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


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "customer-churn"


FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_calls",
    "usage_hours",
    "contract_months",
    "late_payments"
]

TARGET = "churn"


MODEL_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42
}


def load_data():
    """Load training data."""

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
                **MODEL_PARAMS
            )
        )
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

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

    # Connect to MLflow
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # Load data
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()

    # Start one MLflow experiment run
    with mlflow.start_run(
        run_name="logistic_regression_baseline"
    ) as run:

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Evaluate
        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        # Log model parameters
        mlflow.log_params(
            MODEL_PARAMS
        )

        # Log evaluation metrics
        mlflow.log_metrics(
            metrics
        )

        # Create model signature
        signature = infer_signature(
            X_train,
            pipeline.predict(X_train)
        )

        # Log complete sklearn pipeline
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="churn_model",
            signature=signature,
            input_example=X_train.head(3)
        )

        print("\nModel performance:")

        for name, value in metrics.items():
            print(
                f"{name}: {value:.4f}"
            )

        print(
            f"\nMLflow run ID: {run.info.run_id}"
        )

        print(
            f"MLflow model URI: {model_info.model_uri}"
        )

    # Keep our local artifact for now
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    metadata = {
        "model_type": "LogisticRegression",
        "features": FEATURES,
        "target": TARGET,
        "parameters": MODEL_PARAMS,
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
        f"\nLocal model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()