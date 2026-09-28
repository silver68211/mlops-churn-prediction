# Customer Churn Prediction — MLOps Project

An end-to-end MLOps project for customer churn prediction. The project focuses on the operational lifecycle of a machine learning model: reproducible training, experiment tracking, model versioning, API deployment, automated testing, monitoring, and retraining.

## Overview

The workflow follows a typical production ML lifecycle:

```text
Data
  ↓
Training
  ↓
Experiment Tracking
  ↓
Model Registry
  ↓
Testing
  ↓
FastAPI
  ↓
Docker
  ↓
CI/CD
  ↓
Deployment
  ↓
Monitoring
  ↓
New Labeled Data
  ↓
Retraining
  ↓
Model Validation and Promotion
```

The prediction task is binary customer churn classification. Several models are evaluated, while MLflow is used to track experiments and manage model versions.

## Main Components

- **Training:** Logistic Regression, Random Forest, and Gradient Boosting
- **Experiment tracking:** MLflow
- **Model registry:** MLflow Model Registry with `candidate` and `champion` aliases
- **API:** FastAPI
- **Containerization:** Docker
- **Testing:** pytest
- **CI/CD:** GitHub Actions
- **Monitoring:** Prometheus metrics and prediction logs
- **Retraining:** Candidate model evaluation and controlled promotion

## Project Structure

```text
mlops-churn/
├── app/
│   ├── main.py
│   └── monitoring.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── retraining/
├── models/
├── reports/
├── src/
│   ├── make_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── register_model.py
│   ├── monitor_predictions.py
│   └── retrain.py
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/silver68211/mlops-churn-prediction.git
cd mlops-churn-prediction

pip install -r requirements.txt
```

Generate the dataset:

```bash
python -m src.make_data
```

Train and compare the models:

```bash
python -m src.train
```

## MLflow

Start the MLflow server:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "localhost,127.0.0.1,host.docker.internal"
```

The MLflow UI is available at:

```text
http://127.0.0.1:5000
```

MLflow stores training parameters, evaluation metrics, model artifacts, and registered model versions.

## Run the API

Start the FastAPI application:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "monthly_spend": 110,
    "support_calls": 4,
    "usage_hours": 35,
    "contract_months": 1,
    "late_payments": 2
  }'
```

## Docker

Build the Docker image:

```bash
docker build -t mlops-churn-api:1.0 .
```

Run the container:

```bash
docker run \
  --rm \
  -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_URI=models:/churn_classifier@champion \
  mlops-churn-api:1.0
```

## Testing

Run the automated tests with:

```bash
pytest -v
```

Tests cover the data pipeline, model behavior, evaluation functions, and application components.

## Monitoring

The deployed API exposes Prometheus-compatible metrics at:

```text
http://127.0.0.1:8000/metrics
```

The monitoring layer tracks:

- API request counts
- request latency
- model availability
- prediction counts
- churn probability distributions

Prediction inputs and outputs are also logged for later analysis and drift detection.

## Retraining

When new labeled data becomes available, the retraining pipeline trains a new candidate model and compares it with the current production model.

Run:

```bash
python -m src.retrain
```

The general workflow is:

```text
New Labeled Data
       ↓
Train New Candidate
       ↓
Evaluate Candidate
       ↓
Compare with Champion
       ↓
Quality Gate
     /       \
   Pass      Fail
    ↓          ↓
 Promote    Keep Current Model
    ↓
Champion
```

A candidate is promoted to the `champion` alias only if it satisfies the defined quality checks. Otherwise, the existing production model remains unchanged.

## Model Lifecycle

The Model Registry separates model development from production deployment.

```text
Training Run
     ↓
Registered Model
     ↓
candidate
     ↓
Evaluation
     ↓
champion
     ↓
Production API
```

The application loads:

```text
models:/churn_classifier@champion
```

This avoids hard-coding a specific model version. A newer model can be promoted by moving the `champion` alias to the new version.

Previous model versions remain available, allowing rollback if necessary.

## CI/CD

GitHub Actions is used to automate the software delivery process.

The CI pipeline performs:

```text
Code Push
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
Build Docker Image
```

If the tests pass, the deployment pipeline can continue:

```text
Docker Image
    ↓
Container Registry
    ↓
Self-hosted Runner
    ↓
Start New Container
    ↓
Health Check
```

This reduces manual deployment steps and ensures that code changes are tested before reaching production.

## MLOps Lifecycle

The overall project demonstrates the complete feedback loop of an ML system:

```text
Build Model
    ↓
Track Experiments
    ↓
Register Model
    ↓
Test
    ↓
Deploy
    ↓
Monitor
    ↓
Collect New Data
    ↓
Retrain
    ↓
Validate
    ↓
Promote
    ↓
Deploy Again
```

The goal of MLOps is not only to build an accurate model, but also to make the complete ML system reproducible, deployable, observable, maintainable, and safely updatable.