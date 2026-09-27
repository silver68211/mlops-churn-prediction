from prometheus_client import (
    Counter,
    Histogram,
    Gauge
)


# Number of HTTP requests
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)


# Request latency
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"]
)


# Predictions
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["label"]
)


# Predicted churn probability
CHURN_PROBABILITY = Histogram(
    "model_churn_probability",
    "Distribution of predicted churn probabilities"
)


# Whether model successfully loaded
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the ML model is loaded"
)