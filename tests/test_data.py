from src.make_data import generate_data


EXPECTED_COLUMNS = [
    "tenure_months",
    "monthly_spend",
    "support_calls",
    "usage_hours",
    "contract_months",
    "late_payments",
    "churn"
]


def test_generate_data_shape():
    """Data generator should return requested number of rows."""

    data = generate_data(
        n_samples=100,
        random_state=42
    )

    assert len(data) == 100


def test_generate_data_columns():
    """Dataset should contain all expected columns."""

    data = generate_data(
        n_samples=100,
        random_state=42
    )

    assert list(data.columns) == EXPECTED_COLUMNS


def test_target_is_binary():
    """Churn target should contain only 0 and 1."""

    data = generate_data(
        n_samples=500,
        random_state=42
    )

    target_values = set(
        data["churn"].unique()
    )

    assert target_values.issubset(
        {0, 1}
    )


def test_no_missing_values():
    """Synthetic dataset should not contain missing values."""

    data = generate_data(
        n_samples=100,
        random_state=42
    )

    assert not data.isnull().any().any()