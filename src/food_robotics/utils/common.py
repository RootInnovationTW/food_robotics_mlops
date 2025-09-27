import os
import mlflow
from dotenv import load_dotenv

def set_mlflow_tracking_uri() -> None:
    """
    Sets the MLflow tracking URI based on the execution environment.
    - Databricks: No action needed, as it's auto-configured.
    - Kubeflow: Reads MLFLOW_TRACKING_URI from environment variables set in the K8s cluster.
    - Local: Reads Databricks profile from .env file to connect to a Databricks workspace.
    """
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        print("Running on Databricks. MLflow is auto-configured.")
        return
    elif "KFP_POD_NAME" in os.environ:
        print("Running in Kubeflow Pipelines environment.")
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()} (from KFP environment variable)")
        else:
            print("Warning: KFP environment detected but MLFLOW_TRACKING_URI is not set.")
        return
    else:
        print("Running in a local environment. Assuming local or .env setup.")
        load_dotenv()
        profile = os.environ.get("PROFILE")
        if profile:
            mlflow.set_tracking_uri(f"databricks://{profile}")
            mlflow.set_registry_uri(f"databricks-uc://{profile}")
            print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
