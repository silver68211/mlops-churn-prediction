from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# Paths
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

MODEL_PATH = ROOT / "models" / "logistic_regression.joblib"

FIGURE_DIR = ROOT / "reports" / "figures"

THRESHOLD_RESULTS_PATH = (
    ROOT / "reports" / "threshold_analysis.csv"
)


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
# Load data
# -------------------------------------------------

def load_test_data():
    """Recreate the same train/test split used during training."""

    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_test, y_test


# -------------------------------------------------
# Evaluate one threshold
# -------------------------------------------------

def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):
    """Calculate metrics for a classification threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    }


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    THRESHOLD_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load model
    model = joblib.load(
        MODEL_PATH
    )

    # Load test data
    X_test, y_test = load_test_data()

    # Get churn probabilities
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------
    # ROC-AUC
    # ---------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    # ---------------------------------------------
    # Threshold comparison
    # ---------------------------------------------

    thresholds_to_test = [
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20
    ]

    results = []

    for threshold in thresholds_to_test:

        result = evaluate_threshold(
            y_test,
            probabilities,
            threshold
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    print("\nTHRESHOLD COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_df.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False
    )

    # ---------------------------------------------
    # Find threshold with maximum F1
    # ---------------------------------------------

    precision, recall, thresholds = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        /
        (
            precision[:-1]
            + recall[:-1]
            + 1e-10
        )
    )

    best_index = np.argmax(
        f1_scores
    )

    best_threshold = thresholds[
        best_index
    ]

    print(
        "\nThreshold with highest F1:"
    )

    print(
        f"{best_threshold:.4f}"
    )

    print(
        f"Precision: "
        f"{precision[best_index]:.4f}"
    )

    print(
        f"Recall: "
        f"{recall[best_index]:.4f}"
    )

    print(
        f"F1: "
        f"{f1_scores[best_index]:.4f}"
    )

    # ---------------------------------------------
    # Confusion matrix: threshold 0.50
    # ---------------------------------------------

    predictions_05 = (
        probabilities >= 0.50
    ).astype(int)

    cm_05 = confusion_matrix(
        y_test,
        predictions_05
    )

    print(
        "\nConfusion matrix at threshold 0.50:"
    )

    print(
        cm_05
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm_05,
        display_labels=[
            "Stay",
            "Churn"
        ]
    )

    display.plot()

    plt.title(
        "Confusion Matrix - Threshold 0.50"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "confusion_matrix_threshold_050.png"
    )

    plt.close()

    # ---------------------------------------------
    # Confusion matrix: best F1 threshold
    # ---------------------------------------------

    predictions_best = (
        probabilities >= best_threshold
    ).astype(int)

    cm_best = confusion_matrix(
        y_test,
        predictions_best
    )

    print(
        "\nConfusion matrix at best-F1 threshold:"
    )

    print(
        cm_best
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm_best,
        display_labels=[
            "Stay",
            "Churn"
        ]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - Threshold "
        f"{best_threshold:.2f}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "confusion_matrix_best_threshold.png"
    )

    plt.close()

    # ---------------------------------------------
    # Precision-recall curve
    # ---------------------------------------------

    plt.figure()

    plt.plot(
        recall,
        precision
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        "precision_recall_curve.png"
    )

    plt.close()

    print(
        f"\nThreshold table saved to:"
        f"\n{THRESHOLD_RESULTS_PATH}"
    )

    print(
        f"\nFigures saved to:"
        f"\n{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()