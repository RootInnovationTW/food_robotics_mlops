#!/bin/bash

# food_robotics_mlops_refactor.sh（續篇）
# 加入 MLflow 專案結構、模型註冊、自動化訓練與 CI/CD 工作流程

set -e

echo "🔧 添加 MLflow 專案結構與自動化訓練..."

# ==============================
# 5. 建立 MLflow 專案結構
# ==============================
echo "📦 建立 MLflow 專案結構..."

# MLproject
cat > MLproject << 'EOF'
name: food_robotics_mlops

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "data/raw/raw.csv"}
    command: "python src/train.py --data_path {data_path}"
EOF

# conda.yaml
cat > conda.yaml << 'EOF'
name: food_robotics_env
channels:
  - defaults
dependencies:
  - python=3.10
  - scikit-learn
  - pandas
  - numpy
  - mlflow
  - matplotlib
  - seaborn
EOF

# ==============================
# 6. 建立訓練腳本（src/train.py）
# ==============================
echo "🧠 建立訓練腳本..."

cat > src/train.py << 'EOF'
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.sklearn.autolog()

def main(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc}")

    mlflow.sklearn.log_model(model, "model", registered_model_name="food_robotics_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/raw.csv")
    args = parser.parse_args()
    main(args.data_path)
EOF

# ==============================
# 7. 建立 CI/CD 工作流程
# ==============================
echo "⚙️ 建立 GitHub Actions 工作流程..."

cat > .github/workflows/mlops.yml << 'EOF'
name: MLOps Pipeline

on:
  push:
    branches: [ main ]

jobs:
  train-and-log:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install mlflow scikit-learn pandas numpy
      - name: Run training
        run: |
          python src/train.py --data_path data/raw/raw.csv
EOF

# ==============================
# 8. 初始化 MLflow 跟踪資料夾
# ==============================
echo "📁 初始化 MLflow 跟踪資料夾..."
mkdir -p mlruns

