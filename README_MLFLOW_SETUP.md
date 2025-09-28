# Food Robotics MLflow Integration

## 快速開始

1. 安裝依賴：
   ```bash
   pip install -r requirements-complete.txt
   ```

2. 初始化MLflow實驗：
   ```bash
   python scripts/setup_mlflow_experiments.py
   ```

3. 啟動MLflow UI：
   ```bash
   ./scripts/start_mlflow.sh
   ```

4. 訪問 http://localhost:5000 查看MLflow界面

## 目錄結構
- `src/food_robotics/mlflow/` - MLflow工具
- `scripts/` - 設置和維護腳本
- `tests/` - MLflow集成測試
- `notebooks/` - 交互式探索
- `docker/` - 容器化部署
