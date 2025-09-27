#!/usr/bin/env python3
"""
食品机器人测试套件
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from robots.delta_robot import DeltaRobot, DeltaRobotConfig
from robots.scara_robot import SCARARobot, SCARAConfig
from robots.bipedal_robot import BipedalRobot, BipedalConfig

class TestDeltaRobot(unittest.TestCase):
    """Delta机器人测试"""
    
    def setUp(self):
        self.config = DeltaRobotConfig()
        self.robot = DeltaRobot(self.config)
    
    def test_forward_kinematics(self):
        """测试前向运动学"""
        test_angles = np.array([np.pi/6, np.pi/4, np.pi/3])
        result = self.robot.forward_kinematics(test_angles)
        
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_inverse_kinematics(self):
        """测试逆运动学"""
        target_pos = np.array([0.1, 0.1, -0.3])
        result = self.robot.inverse_kinematics(target_pos)
        
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(np.isfinite(result)))

class TestSCARARobot(unittest.TestCase):
    """SCARA机器人测试"""
    
    def setUp(self):
        self.config = SCARAConfig()
        self.robot = SCARARobot(self.config)
    
    def test_kinematics(self):
        """测试运动学"""
        theta1, theta2 = 0.5, 0.3
        pos = self.robot.forward_kinematics(theta1, theta2)
        
        self.assertEqual(len(pos), 3)
        self.assertAlmostEqual(pos[2], 0.1)  # z坐标固定

class TestBipedalRobot(unittest.TestCase):
    """双足机器人测试"""
    
    def setUp(self):
        self.config = BipedalConfig()
        self.robot = BipedalRobot(self.config)
    
    def test_zmp_calculation(self):
        """测试零力矩点计算"""
        positions = [np.array([0, 0, 0.5]) for _ in range(3)]
        accelerations = [np.array([0, 0, 0]) for _ in range(3)]
        masses = [10.0, 5.0, 15.0]
        
        zmp = self.robot.calculate_zmp(positions, accelerations, masses)
        
        self.assertEqual(len(zmp), 2)
        self.assertTrue(np.all(np.isfinite(zmp)))

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
