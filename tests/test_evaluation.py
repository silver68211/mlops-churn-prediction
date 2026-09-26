import numpy as np

from src.evaluate import evaluate_threshold


def test_evaluation_metrics_are_valid():

    y_true = np.array([
        0,
        0,
        1,
        1
    ])

    probabilities = np.array([
        0.1,
        0.4,
        0.6,
        0.9
    ])

    result = evaluate_threshold(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.5
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:

        assert 0 <= result[metric] <= 1