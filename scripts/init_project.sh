#!/bin/bash
# 项目初始化脚本

echo "🔧 初始化食品机器人MLOps项目..."

# 检查Python版本
python3 --version || { echo "❌ Python 3未安装"; exit 1; }

# 创建虚拟环境
python3 -m venv food_robotics_env
source food_robotics_env/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 设置数据目录
mkdir -p data/raw data/processed mlruns

# 运行基础测试
python -m pytest tests/test_robots.py -v

echo "✅ 项目初始化完成!"
echo "激活虚拟环境: source food_robotics_env/bin/activate"
