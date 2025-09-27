# 食品机器人MLOps项目 (重构版)

这是一个完整的食品机器人机器学习运维平台，集成了Delta机器人、SCARA机器人和双足机器人的控制与自动化。

## 🚀 新特性

### 新增组件
- **🤖 真实硬件接口**: UR、HIWIN、ROS机器人支持
- **🎮 物理仿真**: PyBullet集成，真实物理模拟
- **📊 数据流水线**: 完整的传感器和图像数据处理
- **🐳 容器化部署**: Docker和Kubernetes支持
- **⚙️ CI/CD流水线**: GitHub Actions自动化测试和部署
- **🔧 工具脚本**: 项目初始化和数据管理

### 项目结构


food_robotics_mlops/
├── src/ # 源代码
│ ├── hardware/ # 真实机器人接口
│ ├── simulation/ # 物理仿真环境
│ ├── robots/ # 机器人模型
│ ├── control/ # 控制算法
│ ├── ml_models/ # 机器学习模型
│ └── sensors/ # 传感器处理
├── tests/ # 测试套件
├── scripts/ # 工具脚本
├── config/ # 配置文件
├── deployment/ # 部署配置
├── data/ # 数据目录
└── .github/workflows/ # CI/CD配置

## 🏃 快速开始

### 1. 环境设置
```bash
# 运行初始化脚本
./scripts/init_project.sh

# 激活虚拟环境
source food_robotics_env/bin/activate
2. 下载示例数据
bash
复制下载
./scripts/download_data.sh
3. 运行测试
bash
复制下载
python -m pytest tests/ -v
4. 启动仿真
bash
复制下载
python scripts/run_delta_simulation.py
5. 使用Docker
bash
复制下载
# 构建镜像
docker build -t food-robotics-mlops .

# 运行容器
docker-compose up
📊 功能模块
机器人系统
Delta机器人: 高速并行机器人，适用于食品分拣
SCARA机器人: 平面运动机器人，适用于装配任务
双足机器人: 人类步态模仿，适用于服务场景
机器学习
强化学习控制: 基于PPO、DDPG算法的机器人控制
PID优化: 使用MCIWO、PSO算法优化控制器参数
食品质量检测: 基于传感器数据的质量评估
仿真与部署
物理仿真: PyBullet真实物理环境
容器化: Docker一键部署
云原生: Kubernetes集群部署
🔧 开发工具
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 类型检查
mypy src/
测试覆盖
# 运行测试
pytest tests/ --cov=src --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
🤝 贡献指南
Fork本项目
创建特性分支: git checkout -b feature/AmazingFeature
提交更改: git commit -m 'Add some AmazingFeature'
推送到分支: git push origin feature/AmazingFeature
提交Pull Request
📄 许可证
本项目采用MIT许可证 - 查看 LICENSE 文件了解详情
🙏 致谢
感谢所有贡献者的支持
基于最新的MLOps最佳实践
集成先进的机器人控制算法

# ==============================
# 12. 创建Git推送脚本
# ==============================
echo "🔗 创建Git推送脚本..."

cat > scripts/git_push.sh << 'EOF'
#!/bin/bash
# Git自动推送脚本

set -e

echo "🚀 开始Git推送流程..."

# 检查Git状态
if [ ! -d ".git" ]; then
    echo "❌ 当前目录不是Git仓库"
    exit 1
fi

# 添加所有文件
echo "📁 添加文件到Git..."
git add .

# 提交更改
COMMIT_MSG="${1:-'refactor: 项目重构 - 添加缺失组件'}"
echo "💾 提交更改: $COMMIT_MSG"
git commit -m "$COMMIT_MSG" || {
    echo "⚠️ 没有更改需要提交"
    exit 0
}

# 推送到远程仓库
echo "📤 推送到远程仓库..."
git push origin main

echo "✅ Git推送完成!"
echo "📊 查看项目: https://github.com/RootInnovationTW/food_robotics_mlops"
