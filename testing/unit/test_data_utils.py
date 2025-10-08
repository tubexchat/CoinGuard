"""
数据处理工具函数的单元测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training.utils.data_utils import (
    create_target_variable,
    prepare_data,
    split_data_temporal,
    validate_data_quality
)


class TestDataUtils(unittest.TestCase):
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试数据
        np.random.seed(42)
        self.df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 100,
            'open_time': pd.date_range('2023-01-01', periods=100, freq='H'),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(150, 250, 100),
            'low': np.random.uniform(50, 150, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'close_time': pd.date_range('2023-01-01', periods=100, freq='H'),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        # 确保 high >= low
        self.df['high'] = np.maximum(self.df['high'], self.df['low'] + 1)
        
    def test_create_target_variable(self):
        """测试目标变量创建"""
        # 测试lookahead=1的情况
        df_with_target = create_target_variable(self.df.copy(), lookahead=1, drop_percent=-0.1)
        
        self.assertIn('target', df_with_target.columns)
        self.assertTrue(df_with_target['target'].dtype in [np.int64, np.int32])
        self.assertTrue(df_with_target['target'].isin([0, 1]).all())
        
        # 测试lookahead>1的情况
        df_with_target_2 = create_target_variable(self.df.copy(), lookahead=3, drop_percent=-0.1)
        self.assertIn('target', df_with_target_2.columns)
    
    def test_prepare_data(self):
        """测试数据准备函数"""
        df_with_target = create_target_variable(self.df.copy(), lookahead=1, drop_percent=-0.1)
        X, y = prepare_data(df_with_target)
        
        # 检查特征矩阵
        self.assertIsInstance(X, pd.DataFrame)
        self.assertNotIn('target', X.columns)
        self.assertNotIn('symbol', X.columns)
        self.assertNotIn('open_time', X.columns)
        
        # 检查目标变量
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(y), len(df_with_target))
        
        # 检查是否处理了无穷大和NaN值
        self.assertFalse(np.isinf(X).any().any())
        self.assertFalse(np.isnan(X).any().any())
    
    def test_split_data_temporal(self):
        """测试时间序列数据分割"""
        df_with_target = create_target_variable(self.df.copy(), lookahead=1, drop_percent=-0.1)
        X, y = prepare_data(df_with_target)
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
            X, y, train_ratio=0.7, val_ratio=0.15
        )
        
        # 检查分割比例
        total_samples = len(X)
        self.assertEqual(len(X_train), int(total_samples * 0.7))
        self.assertEqual(len(X_val), int(total_samples * 0.15))
        self.assertEqual(len(X_test), total_samples - len(X_train) - len(X_val))
        
        # 检查数据完整性
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))
        
        # 检查时间顺序（训练集应该在验证集之前，验证集应该在测试集之前）
        self.assertTrue(X_train.index.max() < X_val.index.min())
        self.assertTrue(X_val.index.max() < X_test.index.min())
    
    def test_validate_data_quality(self):
        """测试数据质量验证"""
        # 测试正常数据
        self.assertTrue(validate_data_quality(self.df))
        
        # 测试缺少必要列的数据
        df_missing_col = self.df.drop(columns=['symbol'])
        self.assertFalse(validate_data_quality(df_missing_col))
        
        # 测试空数据
        df_empty = pd.DataFrame()
        self.assertFalse(validate_data_quality(df_empty))
        
        # 测试包含非正值的数据
        df_negative = self.df.copy()
        df_negative.loc[0, 'close'] = -1
        self.assertTrue(validate_data_quality(df_negative))  # 应该通过，只是警告
        
        # 测试high < low的数据
        df_invalid_high_low = self.df.copy()
        df_invalid_high_low.loc[0, 'high'] = 50
        df_invalid_high_low.loc[0, 'low'] = 100
        self.assertTrue(validate_data_quality(df_invalid_high_low))  # 应该通过，只是警告


if __name__ == '__main__':
    unittest.main()
