#!/bin/bash
echo "Starting MLflow server..."
python scripts/setup_mlflow_experiments.py
mlflow ui --host 0.0.0.0 --port 5000
