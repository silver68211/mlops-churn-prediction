from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"


FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_calls",
    "usage_hours",
    "contract_months",
    "late_payments"
]


def load_model():
    """Load the trained model pipeline."""

    return joblib.load(MODEL_PATH)


def predict_customer(model, customer):
    """Predict churn for one customer."""

    customer_df = pd.DataFrame(
        [customer]
    )

    # Keep expected features and order
    customer_df = customer_df[FEATURES]

    prediction = model.predict(
        customer_df
    )[0]

    probability = model.predict_proba(
        customer_df
    )[0, 1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }


def main():

    model = load_model()

    customer = {
        "tenure_months": 8,
        "monthly_spend": 110,
        "support_calls": 5,
        "usage_hours": 30,
        "contract_months": 1,
        "late_payments": 3
    }

    result = predict_customer(
        model,
        customer
    )

    print(result)


if __name__ == "__main__":
    main()