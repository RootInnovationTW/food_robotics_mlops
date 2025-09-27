#!/bin/bash
# 数据下载脚本

echo "📥 下载示例数据..."

# 创建数据目录
mkdir -p data/raw/images data/raw/sensor_data

# 下载示例传感器数据（这里使用生成示例数据代替真实下载）
python3 -c "
import pandas as pd
import numpy as np

# 生成示例传感器数据
sensor_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='S'),
    'temperature': np.random.normal(25, 5, 1000),
    'humidity': np.random.normal(60, 10, 1000),
    'pressure': np.random.normal(100, 5, 1000)
})

sensor_data.to_csv('data/raw/sensor_data/example_sensor.csv', index=False)
print('示例传感器数据已生成')
"

echo "✅ 数据准备完成!"
