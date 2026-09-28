from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response
)

from pydantic import BaseModel, Field

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST
)

from app.monitoring import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNT,
    CHURN_PROBABILITY,
    MODEL_LOADED
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/churn_classifier@champion"
)

DECISION_THRESHOLD = float(
    os.getenv(
        "DECISION_THRESHOLD",
        "0.50"
    )
)


# -------------------------------------------------
# Prediction logging
# -------------------------------------------------

PREDICTION_LOG_DIR = Path(
    os.getenv(
        "PREDICTION_LOG_DIR",
        "logs"
    )
)

PREDICTION_LOG_FILE = (
    PREDICTION_LOG_DIR
    / "predictions.jsonl"
)

PREDICTION_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def log_prediction(
    customer,
    probability,
    prediction
):
    """Save one prediction for later model monitoring."""

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "input": customer,

        "churn_probability": probability,

        "prediction": prediction,

        "threshold": DECISION_THRESHOLD
    }

    with open(
        PREDICTION_LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record)
            + "\n"
        )


# -------------------------------------------------
# Global model
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
# Application startup / shutdown
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    try:

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        print(
            f"Loading model: {MODEL_URI}"
        )

        model = mlflow.sklearn.load_model(
            MODEL_URI
        )

        # Prometheus:
        # 1 means that the model is available.
        MODEL_LOADED.set(1)

        print(
            "Model loaded successfully."
        )

    except Exception:

        # Model was not successfully loaded.
        MODEL_LOADED.set(0)

        raise

    yield

    # Application is shutting down.
    MODEL_LOADED.set(0)


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
# Request monitoring middleware
# -------------------------------------------------

@app.middleware("http")
async def monitor_requests(
    request: Request,
    call_next
):

    start_time = time.perf_counter()

    # Assume failure until a response is produced.
    status_code = 500

    try:

        response = await call_next(
            request
        )

        status_code = (
            response.status_code
        )

        return response

    finally:

        duration = (
            time.perf_counter()
            - start_time
        )

        # Count requests.
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code)
        ).inc()

        # Record request latency.
        REQUEST_LATENCY.labels(
            endpoint=request.url.path
        ).observe(
            duration
        )


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------

@app.get("/health")
def health():

    model_loaded = (
        model is not None
    )

    return {
        "status": (
            "healthy"
            if model_loaded
            else "unhealthy"
        ),
        "model_loaded": model_loaded
    }


# -------------------------------------------------
# Prometheus metrics endpoint
# -------------------------------------------------

@app.get(
    "/metrics",
    include_in_schema=False
)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    customer: CustomerInput
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not available."
        )

    # -------------------------------------------------
    # Convert input to DataFrame
    # -------------------------------------------------

    customer_data = (
        customer.model_dump()
    )

    customer_df = pd.DataFrame([
        customer_data
    ])

    # -------------------------------------------------
    # Predict churn probability
    # -------------------------------------------------

    probability = float(
        model.predict_proba(
            customer_df
        )[0, 1]
    )

    # -------------------------------------------------
    # Apply decision threshold
    # -------------------------------------------------

    prediction = int(
        probability
        >= DECISION_THRESHOLD
    )

    label = (
        "churn"
        if prediction == 1
        else "stay"
    )

    # -------------------------------------------------
    # Model monitoring
    # -------------------------------------------------

    # Count model predictions.
    PREDICTION_COUNT.labels(
        label=label
    ).inc()

    # Record probability distribution.
    CHURN_PROBABILITY.observe(
        probability
    )

    # Save input and prediction for
    # later drift/model analysis.
    log_prediction(
        customer=customer_data,
        probability=probability,
        prediction=prediction
    )

    # -------------------------------------------------
    # API response
    # -------------------------------------------------

    return PredictionOutput(
        prediction=prediction,
        label=label,
        churn_probability=probability,
        threshold=DECISION_THRESHOLD
    )