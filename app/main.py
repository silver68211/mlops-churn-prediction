from contextlib import asynccontextmanager
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -------------------------------------------------
# Configuration
# -------------------------------------------------

# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# MODEL_URI = "models:/churn_classifier@candidate"

# DECISION_THRESHOLD = 0.50



MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/churn_classifier@candidate"
)

DECISION_THRESHOLD = float(
    os.getenv(
        "DECISION_THRESHOLD",
        "0.50"
    )
)

# -------------------------------------------------
# Global model variable
# -------------------------------------------------

model = None


# -------------------------------------------------
# Input schema
# -------------------------------------------------

class CustomerInput(BaseModel):

    tenure_months: float = Field(
        ge=0
    )

    monthly_spend: float = Field(
        ge=0
    )

    support_calls: float = Field(
        ge=0
    )

    usage_hours: float = Field(
        ge=0
    )

    contract_months: float = Field(
        ge=0
    )

    late_payments: float = Field(
        ge=0
    )


# -------------------------------------------------
# Output schema
# -------------------------------------------------

class PredictionOutput(BaseModel):

    prediction: int

    label: str

    churn_probability: float

    threshold: float


# -------------------------------------------------
# Application startup
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        f"Loading model: {MODEL_URI}"
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print(
        "Model loaded successfully."
    )

    yield


# -------------------------------------------------
# FastAPI application
# -------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "API for predicting customer churn "
        "using a model managed by MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(customer: CustomerInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    # Convert API input into a DataFrame
    customer_df = pd.DataFrame([
        customer.model_dump()
    ])

    # Probability of class 1 = churn
    probability = float(
        model.predict_proba(
            customer_df
        )[0, 1]
    )

    # Apply our decision threshold
    prediction = int(
        probability >= DECISION_THRESHOLD
    )

    label = (
        "churn"
        if prediction == 1
        else "stay"
    )

    return PredictionOutput(
        prediction=prediction,
        label=label,
        churn_probability=probability,
        threshold=DECISION_THRESHOLD
    )