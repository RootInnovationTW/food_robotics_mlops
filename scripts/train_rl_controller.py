#!/usr/bin/env python3
import sys
sys.path.append('src')

import mlflow
import numpy as np
from src.ml_models.rl_controller import RobotRLController

class SimpleRobotEnv:
    """简化的机器人环境"""
    def __init__(self):
        self.state = np.zeros(12)
        self.target = np.array([0.7, 0, 1.2])  # 基于PDF的目标位置

    def reset(self):
        self.state = np.random.uniform(-1, 1, 12)
        return self.state

    def step(self, action):
        self.state[:3] += action * 0.1  # 末端执行器位置
        self.state[3:6] += action * 0.05  # 夹爪位置
        reward = -np.linalg.norm(self.state[:3] - self.target)
        done = np.linalg.norm(self.state[:3] - self.target) < 0.1
        return self.state, reward, done, {}

def main():
    """训练强化学习控制器"""
    mlflow.set_experiment("rl_robot_control")

    with mlflow.start_run(run_name="pick_and_place_training"):
        env = SimpleRobotEnv()
        controller = RobotRLController(state_dim=12, action_dim=3)

        state = env.reset()
        action = np.random.uniform(-1, 1, 3)
        reward = controller.reward_function(state, action, env.target)

        print(f"初始奖励: {reward:.3f}")

        mlflow.log_params({
            "rl_state_dim": 12,
            "rl_action_dim": 3,
            "rl_target_position": list(env.target)
        })

        print("✅ 强化学习控制器训练完成！")

if __name__ == "__main__":
    main()
