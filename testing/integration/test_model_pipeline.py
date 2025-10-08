"""
模型训练和预测管道的集成测试
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

from training.utils.data_utils import load_data, create_target_variable, prepare_data, split_data_temporal
from training.utils.model_utils import train_xgboost_model, evaluate_model, save_model_and_metadata, load_model


class TestModelPipeline(unittest.TestCase):
    
    def setUp(self):
        """设置测试数据"""
        # 创建更真实的测试数据
        np.random.seed(42)
        n_samples = 1000
        
        # 创建多个交易对的数据
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        data = []
        
        for symbol in symbols:
            symbol_data = pd.DataFrame({
                'symbol': [symbol] * n_samples,
                'open_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
                'open': np.random.uniform(100, 200, n_samples),
                'high': np.random.uniform(150, 250, n_samples),
                'low': np.random.uniform(50, 150, n_samples),
                'close': np.random.uniform(100, 200, n_samples),
                'volume': np.random.uniform(1000, 5000, n_samples),
                'close_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
                'feature1': np.random.randn(n_samples),
                'feature2': np.random.randn(n_samples),
                'feature3': np.random.randn(n_samples)
            })
            
            # 确保 high >= low
            symbol_data['high'] = np.maximum(symbol_data['high'], symbol_data['low'] + 1)
            data.append(symbol_data)
        
        self.df = pd.concat(data, ignore_index=True)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """清理测试数据"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_training_pipeline(self):
        """测试完整的训练管道"""
        # 1. 创建目标变量
        df_with_target = create_target_variable(self.df.copy(), lookahead=1, drop_percent=-0.1)
        
        # 2. 准备数据
        X, y = prepare_data(df_with_target)
        
        # 3. 分割数据
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
            X, y, train_ratio=0.7, val_ratio=0.15
        )
        
        # 4. 训练模型
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_estimators': 100,  # 使用较小的值以加快测试
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = train_xgboost_model(X_train, y_train, X_val, y_val, model_params)
        
        # 5. 评估模型
        evaluation_result = evaluate_model(model, X_test, y_test, threshold=0.5)
        
        # 验证结果
        self.assertIsNotNone(model)
        self.assertIn('auc_roc', evaluation_result)
        self.assertIn('confusion_matrix', evaluation_result)
        self.assertIn('classification_report', evaluation_result)
        
        # 验证AUC分数在合理范围内
        self.assertGreater(evaluation_result['auc_roc'], 0.0)
        self.assertLessEqual(evaluation_result['auc_roc'], 1.0)
    
    def test_model_save_and_load_pipeline(self):
        """测试模型保存和加载管道"""
        # 准备数据
        df_with_target = create_target_variable(self.df.copy(), lookahead=1, drop_percent=-0.1)
        X, y = prepare_data(df_with_target)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
            X, y, train_ratio=0.7, val_ratio=0.15
        )
        
        # 训练模型
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = train_xgboost_model(X_train, y_train, X_val, y_val, model_params)
        
        # 保存模型
        config = {
            "data": {
                "model_output_path": self.temp_dir
            }
        }
        
        model_name, model_path = save_model_and_metadata(
            model, X, y, config, "test_pipeline_model"
        )
        
        # 加载模型
        loaded_model = load_model(model_path)
        
        # 验证加载的模型与原模型预测结果一致
        original_predictions = model.predict(X_test.head(10))
        loaded_predictions = loaded_model.predict(X_test.head(10))
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_model_prediction_consistency(self):
        """测试模型预测的一致性"""
        # 准备数据
        df_with_target = create_target_variable(self.df.copy(), lookahead=1, drop_percent=-0.1)
        X, y = prepare_data(df_with_target)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
            X, y, train_ratio=0.7, val_ratio=0.15
        )
        
        # 训练模型
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = train_xgboost_model(X_train, y_train, X_val, y_val, model_params)
        
        # 多次预测相同数据，验证结果一致性
        predictions_1 = model.predict(X_test.head(10))
        predictions_2 = model.predict(X_test.head(10))
        
        np.testing.assert_array_equal(predictions_1, predictions_2)
        
        # 验证概率预测的一致性
        proba_1 = model.predict_proba(X_test.head(10))
        proba_2 = model.predict_proba(X_test.head(10))
        
        np.testing.assert_array_almost_equal(proba_1, proba_2, decimal=10)
    
    def test_data_preprocessing_consistency(self):
        """测试数据预处理的一致性"""
        # 多次处理相同数据，验证结果一致性
        df_copy1 = self.df.copy()
        df_copy2 = self.df.copy()
        
        # 创建目标变量
        df_with_target1 = create_target_variable(df_copy1, lookahead=1, drop_percent=-0.1)
        df_with_target2 = create_target_variable(df_copy2, lookahead=1, drop_percent=-0.1)
        
        # 准备数据
        X1, y1 = prepare_data(df_with_target1)
        X2, y2 = prepare_data(df_with_target2)
        
        # 验证结果一致性
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


if __name__ == '__main__':
    unittest.main()
