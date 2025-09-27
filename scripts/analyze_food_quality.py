#!/usr/bin/env python3
import sys
sys.path.append('src')

import mlflow
import numpy as np
from src.sensors.food_quality_sensor import FoodQualitySensor, VegetableCookingModel

def main():
"""分析食品质量传感器数据"""

text
mlflow.set_experiment("food_quality_analysis")

with mlflow.start_run(run_name="vegetable_cooking_analysis"):
    # 创建烹饪模型和传感器
    model = VegetableCookingModel()
    sensor = FoodQualitySensor(model)
    
    # 模拟烹饪过程
    cooking_times = np.linspace(0, 1200, 10)  # 0-20分钟
    raw_radius = 0.03  # 生蔬菜半径
    
    stiffness_values = []
    for time in cooking_times:
        stiffness = sensor.calculate_stiffness(raw_radius, time)
        stiffness_values.append(stiffness)
        
        print(f"时间 {time:.0f}秒 - 刚度: {stiffness:.2e}")
    
    # 估计烹饪完成时间
    target_stiffness = 5e4  # 目标刚度
    estimated_time = sensor.estimate_cooking_time(target_stiffness, stiffness_values[-1])
    
    print(f"估计剩余烹饪时间: {estimated_time:.1f}秒")
    
    # 记录分析结果
    mlflow.log_metric("final_stiffness", stiffness_values[-1])
    mlflow.log_metric("estimated_remaining_time", estimated_time)
if name == "main":
main()
