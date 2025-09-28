#!/usr/bin/env python3
"""初始化食品機器人MLflow實驗"""
import mlflow
import yaml

def setup_experiments():
    with open('mlflow_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 設置追蹤URI
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # 創建實驗
    for exp_name in config['experiments'].values():
        try:
            mlflow.create_experiment(exp_name)
            print(f"Created experiment: {exp_name}")
        except:
            print(f"Experiment {exp_name} already exists")

if __name__ == "__main__":
    setup_experiments()
