"""MLflow tracking utilities"""
import mlflow
import os

def setup_tracking():
    """設置MLflow追蹤"""
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("file:./mlruns")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
