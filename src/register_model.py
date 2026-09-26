import mlflow

from mlflow import MlflowClient


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "customer-churn"

REGISTERED_MODEL_NAME = "churn_classifier"

RUN_NAME = "logistic_regression"


def main():

    # Connect to MLflow
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    # -------------------------------------------------
    # Find experiment
    # -------------------------------------------------

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found."
        )

    # -------------------------------------------------
    # Find latest successful Logistic Regression run
    # -------------------------------------------------

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],

        filter_string=(
            f"tags.mlflow.runName = '{RUN_NAME}' "
            "and attributes.status = 'FINISHED'"
        ),

        order_by=[
            "attributes.start_time DESC"
        ],

        max_results=1
    )

    if runs.empty:
        raise RuntimeError(
            f"No successful run found for "
            f"'{RUN_NAME}'."
        )

    run_id = runs.iloc[0]["run_id"]

    print(
        f"Selected run ID: {run_id}"
    )

    # -------------------------------------------------
    # Model URI
    # -------------------------------------------------

    model_uri = (
        f"runs:/{run_id}/model"
    )

    print(
        f"Model URI: {model_uri}"
    )

    # -------------------------------------------------
    # Register model
    # -------------------------------------------------

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print(
        f"\nRegistered model:"
        f"\nName: {model_version.name}"
        f"\nVersion: {model_version.version}"
    )

    # -------------------------------------------------
    # Add useful model version tags
    # -------------------------------------------------

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="model_type",
        value="LogisticRegression"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="task",
        value="customer_churn_classification"
    )

    # -------------------------------------------------
    # Assign candidate alias
    # -------------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="candidate",
        version=model_version.version
    )

    print(
        f"\nAlias 'candidate' now points to "
        f"version {model_version.version}."
    )


if __name__ == "__main__":
    main()