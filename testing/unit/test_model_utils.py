"""
模型工具函数的单元测试
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training.utils.model_utils import (
    evaluate_model, 
    save_model_and_metadata, 
    load_model,
    load_model_features,
    load_model_config,
    get_latest_model_info
)


class TestModelUtils(unittest.TestCase):
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试数据
        np.random.seed(42)
        self.X_test = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y_test = pd.Series(np.random.randint(0, 2, 100))
        
        # 创建模拟模型
        self.mock_model = MagicMock()
        self.mock_model.predict_proba.return_value = np.column_stack([
            np.random.rand(100),  # 类别0的概率
            np.random.rand(100)   # 类别1的概率
        ])
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """清理测试数据"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluate_model(self):
        """测试模型评估函数"""
        with patch('sklearn.metrics.roc_auc_score') as mock_auc, \
             patch('sklearn.metrics.confusion_matrix') as mock_cm, \
             patch('sklearn.metrics.classification_report') as mock_report:
            
            mock_auc.return_value = 0.85
            mock_cm.return_value = np.array([[80, 10], [5, 5]])
            mock_report.return_value = {'accuracy': 0.85}
            
            result = evaluate_model(self.mock_model, self.X_test, self.y_test, threshold=0.5)
            
            self.assertIn('confusion_matrix', result)
            self.assertIn('classification_report', result)
            self.assertIn('auc_roc', result)
            self.assertIn('y_pred_proba', result)
            self.assertIn('y_pred', result)
            self.assertEqual(result['auc_roc'], 0.85)
    
    def test_save_and_load_model(self):
        """测试模型保存和加载"""
        config = {
            "data": {
                "model_output_path": self.temp_dir
            }
        }
        
        # 测试保存模型
        model_name, model_path = save_model_and_metadata(
            self.mock_model, self.X_test, self.y_test, config, "test_model"
        )
        
        self.assertEqual(model_name, "test_model")
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_model_features.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_model_config.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_model_stats.pkl")))
        
        # 测试加载模型
        loaded_model = load_model(model_path)
        self.assertEqual(loaded_model, self.mock_model)
        
        # 测试加载特征
        features = load_model_features(os.path.join(self.temp_dir, "test_model_features.pkl"))
        self.assertEqual(features, list(self.X_test.columns))
        
        # 测试加载配置
        loaded_config = load_model_config(os.path.join(self.temp_dir, "test_model_config.pkl"))
        self.assertEqual(loaded_config, config)
    
    def test_get_latest_model_info(self):
        """测试获取最新模型信息"""
        # 创建一些测试模型文件
        test_files = [
            "model1.pkl",
            "model1_features.pkl", 
            "model1_config.pkl",
            "model1_stats.pkl",
            "model2.pkl",
            "model2_features.pkl",
            "model2_config.pkl", 
            "model2_stats.pkl"
        ]
        
        for filename in test_files:
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write("test")
        
        # 测试获取最新模型信息
        latest_info = get_latest_model_info(self.temp_dir)
        
        self.assertIn('model_name', latest_info)
        self.assertIn('model_path', latest_info)
        self.assertIn('feature_path', latest_info)
        self.assertIn('config_path', latest_info)
        self.assertIn('stats_path', latest_info)
        
        # 验证文件路径存在
        for key in ['model_path', 'feature_path', 'config_path', 'stats_path']:
            self.assertTrue(os.path.exists(latest_info[key]))


if __name__ == '__main__':
    unittest.main()
