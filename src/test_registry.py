import mlflow
import pandas as pd


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

MODEL_URI = "models:/churn_classifier@candidate"


def main():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    customer = pd.DataFrame([
        {
            "tenure_months": 8.0,
            "monthly_spend": 110.0,
            "support_calls": 5.0,
            "usage_hours": 30.0,
            "contract_months": 1.0,
            "late_payments": 3.0
        }
    ])

    prediction = model.predict(
        customer
    )

    probability = model.predict_proba(
        customer
    )[:, 1]

    print(
        f"Prediction: {prediction[0]}"
    )

    print(
        f"Churn probability: "
        f"{probability[0]:.4f}"
    )


if __name__ == "__main__":
    main()