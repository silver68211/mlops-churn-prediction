from pathlib import Path
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
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
# Paths
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

MODEL_DIR = ROOT / "models"

COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"


# -------------------------------------------------
# MLflow configuration
# -------------------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "customer-churn"


# -------------------------------------------------
# Dataset configuration
# -------------------------------------------------

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
# Models
# -------------------------------------------------

MODELS = {
    "logistic_regression": {
        "model": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        "use_scaler": True,
        "skops_trusted_types": None
    },

    "random_forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "use_scaler": False,
        "skops_trusted_types": [
            "sklearn.tree._tree.Tree"
        ]
    },

    "gradient_boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        "use_scaler": False,
        "skops_trusted_types": [
            "sklearn.tree._tree.Tree"
        ]
    }
}


# -------------------------------------------------
# Data
# -------------------------------------------------

def load_data():
    """Load the churn dataset."""

    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    return X, y


# -------------------------------------------------
# Pipeline
# -------------------------------------------------

def build_pipeline(model, use_scaler):
    """Build the preprocessing and model pipeline."""

    steps = []

    if use_scaler:
        steps.append(
            ("scaler", StandardScaler())
        )

    steps.append(
        ("model", model)
    )

    return Pipeline(steps)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Calculate classification metrics."""

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


# -------------------------------------------------
# Train one model
# -------------------------------------------------

def train_model(
    model_name,
    model_config,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train, evaluate and log one model."""

    model = model_config["model"]

    use_scaler = model_config["use_scaler"]
    skops_trusted_types = model_config.get(
        "skops_trusted_types"
        )

    pipeline = build_pipeline(
        model,
        use_scaler
    )

    print(f"\nTraining: {model_name}")

    # Measure training time
    start_time = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter() - start_time
    )

    # Evaluate
    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    metrics["training_time_seconds"] = (
        training_time
    )

    # ---------------------------------------------
    # MLflow run
    # ---------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # General parameters
        mlflow.log_param(
            "model_type",
            model.__class__.__name__
        )

        mlflow.log_param(
            "use_scaler",
            use_scaler
        )

        # Model-specific parameters
        mlflow.log_params(
            model.get_params()
        )

        # Metrics
        mlflow.log_metrics(
            metrics
        )

        # Tags make runs easier to organize
        mlflow.set_tag(
            "experiment_stage",
            "model_comparison"
        )

        # Use float input for a serving-friendly schema.
        # This also avoids the integer/missing-value
        # warning we saw previously.
        signature_input = (
            X_train
            .head(100)
            .astype("float64")
        )

        signature = infer_signature(
            signature_input,
            pipeline.predict(
                signature_input
            )
        )

        # Save complete sklearn pipeline in MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=signature_input.head(3), 
            skops_trusted_types=skops_trusted_types
        )

        run_id = run.info.run_id

    # ---------------------------------------------
    # Save local copy
    # ---------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR /
        f"{model_name}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    # Print results
    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1 score:  {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   {metrics['roc_auc']:.4f}"
    )

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    return {
        "model": model_name,
        **metrics,
        "run_id": run_id
    }


# -------------------------------------------------
# Main
# -------------------------------------------------

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

    # IMPORTANT:
    # All models use exactly the same split.
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
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples:     {len(X_test)}"
    )

    # Store model results
    results = []

    # Train every candidate model
    for model_name, model_config in MODELS.items():

        result = train_model(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

        results.append(
            result
        )

    # ---------------------------------------------
    # Compare models
    # ---------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        by="roc_auc",
        ascending=False
    )

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "training_time_seconds"
            ]
        ].to_string(
            index=False
        )
    )

    # Save comparison table
    comparison.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print(
        f"\nComparison saved to:"
        f"\n{COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()