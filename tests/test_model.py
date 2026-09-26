import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.make_data import generate_data
from src.train import (
    FEATURES,
    TARGET,
    build_pipeline
)


def create_test_dataset():
    """Create a small dataset for model tests."""

    data = generate_data(
        n_samples=500,
        random_state=42
    )

    X = data[FEATURES]
    y = data[TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def create_pipeline():
    """Create a lightweight model for testing."""

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return build_pipeline(
        model=model,
        use_scaler=True
    )


def test_model_can_train():
    """Pipeline should successfully train."""

    X_train, _, y_train, _ = (
        create_test_dataset()
    )

    pipeline = create_pipeline()

    pipeline.fit(
        X_train,
        y_train
    )


def test_prediction_shape():
    """Model should return one prediction per sample."""

    X_train, X_test, y_train, _ = (
        create_test_dataset()
    )

    pipeline = create_pipeline()

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    assert len(predictions) == len(X_test)


def test_predictions_are_binary():
    """Predictions must be either 0 or 1."""

    X_train, X_test, y_train, _ = (
        create_test_dataset()
    )

    pipeline = create_pipeline()

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    assert set(
        predictions
    ).issubset(
        {0, 1}
    )


def test_probability_range():
    """Churn probabilities should lie between 0 and 1."""

    X_train, X_test, y_train, _ = (
        create_test_dataset()
    )

    pipeline = create_pipeline()

    pipeline.fit(
        X_train,
        y_train
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )