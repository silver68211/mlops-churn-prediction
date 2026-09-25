from pathlib import Path

import numpy as np
import pandas as pd


# Project root directory
ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = ROOT / "data" / "raw" / "churn.csv"


def generate_data(n_samples=5000, random_state=42):
    """Generate a simple synthetic customer churn dataset."""

    rng = np.random.default_rng(random_state)

    tenure_months = rng.integers(1, 73, n_samples)

    monthly_spend = rng.uniform(
        20,
        150,
        n_samples
    )

    support_calls = rng.poisson(
        2,
        n_samples
    )

    usage_hours = rng.normal(
        50,
        20,
        n_samples
    )

    usage_hours = np.clip(
        usage_hours,
        1,
        None
    )

    contract_months = rng.choice(
        [1, 12, 24],
        size=n_samples,
        p=[0.4, 0.35, 0.25]
    )

    late_payments = rng.poisson(
        1,
        n_samples
    )

    # Create an underlying relationship between
    # customer characteristics and churn.
    logit = (
        -1.2
        - 0.03 * tenure_months
        + 0.012 * (monthly_spend - 70)
        + 0.35 * support_calls
        - 0.015 * (usage_hours - 50)
        - 0.05 * contract_months
        + 0.30 * late_payments
    )

    churn_probability = 1 / (1 + np.exp(-logit))

    churn = rng.binomial(
        1,
        churn_probability
    )

    data = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "support_calls": support_calls,
        "usage_hours": usage_hours,
        "contract_months": contract_months,
        "late_payments": late_payments,
        "churn": churn
    })

    return data


def main():

    data = generate_data()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Number of rows: {len(data)}")
    print(f"Churn rate: {data['churn'].mean():.2%}")


if __name__ == "__main__":
    main()