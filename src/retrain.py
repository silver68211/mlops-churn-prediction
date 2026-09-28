from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
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


# -------------------------------------------------
# Configuration
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(
    os.getenv(
        "RETRAINING_DATA_PATH",
        ROOT / "data" / "retraining" / "churn_latest.csv"
    )
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

EXPERIMENT_NAME = "customer-churn-retraining"

REGISTERED_MODEL_NAME = "churn_classifier"

CHAMPION_URI = (
    "models:/churn_classifier@champion"
)

DECISION_THRESHOLD = float(
    os.getenv(
        "DECISION_THRESHOLD",
        "0.50"
    )
)


# Require at least this much ROC-AUC improvement.
MIN_ROC_AUC_GAIN = 0.005


FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_calls",
    "usage_hours",
    "contract_months",
    "late_payments"
]

TARGET = "churn"


# -------------------------------------------------
# Model
# -------------------------------------------------

def build_pipeline():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        )
    ])


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate_model(
    model,
    X,
    y
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    return {

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        )
    }


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    client = MlflowClient()

    # ---------------------------------------------
    # Load latest labeled data
    # ---------------------------------------------

    data = pd.read_csv(
        DATA_PATH
    )

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Validation rows: {len(X_test)}"
    )

    # ---------------------------------------------
    # Evaluate current champion
    # ---------------------------------------------

    print(
        "\nLoading current champion..."
    )

    champion = mlflow.sklearn.load_model(
        CHAMPION_URI
    )

    champion_metrics = evaluate_model(
        champion,
        X_test,
        y_test
    )

    print(
        "\nCHAMPION METRICS"
    )

    for key, value in champion_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    # ---------------------------------------------
    # Train new candidate
    # ---------------------------------------------

    candidate = build_pipeline()

    candidate.fit(
        X_train,
        y_train
    )

    candidate_metrics = evaluate_model(
        candidate,
        X_test,
        y_test
    )

    print(
        "\nCANDIDATE METRICS"
    )

    for key, value in candidate_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )

    # ---------------------------------------------
    # Promotion rule
    # ---------------------------------------------

    roc_auc_gain = (
        candidate_metrics["roc_auc"]
        - champion_metrics["roc_auc"]
    )

    recall_not_worse = (
        candidate_metrics["recall"]
        >= champion_metrics["recall"]
    )

    passes_quality_gate = (
        roc_auc_gain >= MIN_ROC_AUC_GAIN
        and recall_not_worse
    )

    print(
        "\nQUALITY GATE"
    )

    print(
        f"ROC-AUC gain: "
        f"{roc_auc_gain:.4f}"
    )

    print(
        f"Recall not worse: "
        f"{recall_not_worse}"
    )

    print(
        f"Candidate passes: "
        f"{passes_quality_gate}"
    )

    # ---------------------------------------------
    # Log retraining experiment
    # ---------------------------------------------

    with mlflow.start_run(
        run_name="logistic_regression_retraining"
    ):

        mlflow.log_param(
            "decision_threshold",
            DECISION_THRESHOLD
        )

        mlflow.log_param(
            "minimum_roc_auc_gain",
            MIN_ROC_AUC_GAIN
        )

        # Champion metrics
        for key, value in champion_metrics.items():

            mlflow.log_metric(
                f"champion_{key}",
                value
            )

        # Candidate metrics
        for key, value in candidate_metrics.items():

            mlflow.log_metric(
                f"candidate_{key}",
                value
            )

        mlflow.log_metric(
            "roc_auc_gain",
            roc_auc_gain
        )

        mlflow.set_tag(
            "quality_gate_passed",
            str(passes_quality_gate)
        )

        # Model signature
        signature_input = (
            X_train
            .head(100)
            .astype("float64")
        )

        signature = infer_signature(
            signature_input,
            candidate.predict(
                signature_input
            )
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=candidate,
            name="model",
            signature=signature,
            input_example=(
                signature_input.head(3)
            )
        )

    # ---------------------------------------------
    # Stop if candidate is not good enough
    # ---------------------------------------------

    if not passes_quality_gate:

        print(
            "\nCandidate rejected."
        )

        print(
            "Current champion remains unchanged."
        )

        return

    # ---------------------------------------------
    # Register successful candidate
    # ---------------------------------------------

    print(
        "\nCandidate passed quality gate."
    )

    model_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME
    )

    version = model_version.version

    print(
        f"Registered version: {version}"
    )

    # ---------------------------------------------
    # Candidate alias
    # ---------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="candidate",
        version=version
    )

    print(
        f"candidate -> version {version}"
    )

    # ---------------------------------------------
    # Promote candidate to champion
    # ---------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=version
    )

    print(
        f"champion -> version {version}"
    )

    print(
        "\nModel promotion completed."
    )


if __name__ == "__main__":
    main()