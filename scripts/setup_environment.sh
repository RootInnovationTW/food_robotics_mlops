#!/bin/bash

echo "🔧 设置食品机器人MLOps环境..."

#创建虚拟环境
python3 -m venv food_robotics_env
source food_robotics_env/bin/activate

#安装依赖
pip install --upgrade pip
pip install -r requirements.txt

#创建配置文件
cat > src/config/robot_config.yaml << 'CONFIG'
delta_robot:
base_radius: 0.2
effector_radius: 0.1
upper_arm_length: 0.3
forearm_length: 0.4

scara_robot:
L1: 0.3
L2: 0.25
max_speed: 2.0

bipedal_robot:
foot_length: 0.2
foot_width: 0.1
body_mass: 30.0

sensors:
thermal_diffusivity: 1.4e-7
cooking_temperature: 100
CONFIG

echo "✅ 环境设置完成！"
echo "激活虚拟环境: source food_robotics_env/bin/activate"
