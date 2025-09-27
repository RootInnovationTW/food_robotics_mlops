#!/usr/bin/env python3
import sys
sys.path.append('src')

import numpy as np
import mlflow
from src.robots.delta_robot import DeltaRobot, DeltaRobotConfig

def main():
"""运行Delta机器人仿真"""

# 设置MLflow实验
mlflow.set_experiment("delta_robot_simulation")

with mlflow.start_run(run_name="delta_performance_analysis"):
    # 创建Delta机器人
    config = DeltaRobotConfig()
    delta = DeltaRobot(config)
    
    # 测试运动学
    test_angles = np.array([np.pi/6, np.pi/4, np.pi/3])
    end_effector_pos = delta.forward_kinematics(test_angles)
    
    print(f"末端执行器位置: {end_effector_pos}")
    
    # 计算周期时间
    trajectory = [np.array([0.1, 0.1, -0.2]), 
                 np.array([0.2, 0.1, -0.3]),
                 np.array([0.1, 0.2, -0.4])]
    
    cycle_time = delta.calculate_cycle_time(trajectory)
    productivity = delta.productivity_analysis(cycle_time)
    
    print(f"周期时间: {cycle_time:.2f} 秒")
    print(f"生产率: {productivity:.2f} 单位/小时")
    
    # 记录关键指标
    mlflow.log_metric("simulation_success", 1)
    mlflow.log_param("robot_type", "delta_parallel")
if name == "main":
main()
