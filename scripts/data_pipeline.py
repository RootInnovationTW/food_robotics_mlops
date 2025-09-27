#!/usr/bin/env python3
"""
食品机器人数据流水线
处理传感器数据、图像数据、机器人状态数据等
"""

import pandas as pd
import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

class FoodDataPipeline:
    """食品数据处理流水线"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.setup_directories()
        self.logger = self._setup_logging()
    
    def setup_directories(self):
        """设置数据目录结构"""
        directories = [
            "raw/images",
            "raw/sensor_data",
            "raw/robot_states",
            "processed/images",
            "processed/features",
            "processed/models",
            "external"
        ]
        
        for dir_path in directories:
            (self.data_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("FoodDataPipeline")
    
    def process_sensor_data(self, sensor_file: str) -> pd.DataFrame:
        """处理传感器数据"""
        self.logger.info(f"处理传感器数据: {sensor_file}")
        
        # 读取传感器数据（假设是CSV格式）
        try:
            df = pd.read_csv(self.data_dir / "raw/sensor_data" / sensor_file)
            
            # 数据清洗和预处理
            df_clean = self._clean_sensor_data(df)
            
            # 特征工程
            df_features = self._extract_sensor_features(df_clean)
            
            # 保存处理后的数据
            output_file = self.data_dir / "processed/features" / f"processed_{sensor_file}"
            df_features.to_csv(output_file, index=False)
            
            self.logger.info(f"传感器数据处理完成: {output_file}")
            return df_features
            
        except Exception as e:
            self.logger.error(f"传感器数据处理失败: {e}")
            raise
    
    def _clean_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗传感器数据"""
        # 移除重复值
        df_clean = df.drop_duplicates()
        
        # 处理缺失值
        df_clean = df_clean.fillna(method='ffill')
        
        # 过滤异常值
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean
    
    def _extract_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取传感器特征"""
        # 基本统计特征
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        features = {}
        for col in numeric_columns:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()
        
        # 时间序列特征（如果有时间戳）
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            for col in numeric_columns:
                if col != 'timestamp':
                    # 差分特征
                    features[f"{col}_diff_mean"] = df_sorted[col].diff().mean()
        
        return pd.DataFrame([features])
    
    def process_food_images(self, image_dir: str, output_size: tuple = (224, 224)):
        """处理食品图像数据"""
        self.logger.info(f"处理食品图像: {image_dir}")
        
        image_path = self.data_dir / "raw/images" / image_dir
        output_path = self.data_dir / "processed/images" / image_dir
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        for img_file in image_path.glob("*.jpg") + image_path.glob("*.png"):
            try:
                # 读取图像
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # 图像预处理
                img_processed = self._preprocess_image(img, output_size)
                
                # 保存处理后的图像
                output_file = output_path / img_file.name
                cv2.imwrite(str(output_file), img_processed)
                
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"图像处理失败 {img_file}: {e}")
        
        self.logger.info(f"图像处理完成: {processed_count} 张图片")
    
    def _preprocess_image(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """图像预处理"""
        # 调整大小
        img_resized = cv2.resize(img, size)
        
        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 颜色空间转换 (BGR to RGB)
        img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        
        return (img_rgb * 255).astype(np.uint8)
    
    def process_robot_states(self, state_file: str) -> Dict[str, Any]:
        """处理机器人状态数据"""
        self.logger.info(f"处理机器人状态: {state_file}")
        
        try:
            with open(self.data_dir / "raw/robot_states" / state_file, 'r') as f:
                state_data = json.load(f)
            
            # 状态数据分析和特征提取
            processed_states = self._analyze_robot_states(state_data)
            
            # 保存分析结果
            output_file = self.data_dir / "processed/features" / f"robot_states_{state_file}"
            with open(output_file, 'w') as f:
                json.dump(processed_states, f, indent=2)
            
            self.logger.info(f"机器人状态处理完成: {output_file}")
            return processed_states
            
        except Exception as e:
            self.logger.error(f"机器人状态处理失败: {e}")
            raise
    
    def _analyze_robot_states(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析机器人状态数据"""
        analysis = {
            "summary": {},
            "performance_metrics": {},
            "anomalies": []
        }
        
        # 这里可以添加具体的状态分析逻辑
        if 'joint_positions' in state_data:
            joints = np.array(state_data['joint_positions'])
            analysis['summary']['num_states'] = len(joints)
            analysis['performance_metrics']['joint_range'] = {
                'min': joints.min(axis=0).tolist(),
                'max': joints.max(axis=0).tolist(),
                'mean': joints.mean(axis=0).tolist()
            }
        
        return analysis


class DataPipelineManager:
    """数据流水线管理器"""
    
    def __init__(self, config_file: str = "config/data_pipeline.yaml"):
        self.config = self._load_config(config_file)
        self.pipeline = FoodDataPipeline(self.config.get('data_dir', 'data'))
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"配置文件加载失败: {e}")
            return {}
    
    def run_full_pipeline(self):
        """运行完整数据流水线"""
        self.logger.info("开始运行完整数据流水线")
        
        # 处理传感器数据
        for sensor_file in self.config.get('sensor_files', []):
            self.pipeline.process_sensor_data(sensor_file)
        
        # 处理图像数据
        for image_dir in self.config.get('image_dirs', []):
            self.pipeline.process_food_images(image_dir)
        
        # 处理机器人状态
        for state_file in self.config.get('state_files', []):
            self.pipeline.process_robot_states(state_file)
        
        self.logger.info("数据流水线运行完成")

if __name__ == "__main__":
    # 测试数据流水线
    pipeline = FoodDataPipeline()
    
    # 创建测试数据
    test_sensor_data = pd.DataFrame({
        'timestamp': range(100),
        'temperature': np.random.normal(25, 5, 100),
        'pressure': np.random.normal(100, 10, 100)
    })
    
    test_file = pipeline.data_dir / "raw/sensor_data/test_sensor.csv"
    test_sensor_data.to_csv(test_file, index=False)
    
    # 测试处理流程
    processed_data = pipeline.process_sensor_data("test_sensor.csv")
    print("传感器数据处理测试完成")
    print(processed_data.head())
