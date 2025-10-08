"""
模型管理器
负责模型的加载、预测和管理
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from training.utils.model_utils import get_latest_model_info, load_model, load_model_features, load_model_config

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器类"""
    
    def __init__(self, model_dir: str = "../data/models"):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = os.path.abspath(model_dir)
        self.model = None
        self.feature_names = None
        self.model_config = None
        self.model_stats = None
        self.model_info = None
        
    def load_latest_model(self) -> bool:
        """
        加载最新的模型
        
        Returns:
            bool: 是否加载成功
        """
        try:
            # 获取最新模型信息
            latest_info = get_latest_model_info(self.model_dir)
            
            if not latest_info:
                logger.error(f"在目录 {self.model_dir} 中未找到模型文件")
                return False
            
            # 检查所有必需的文件是否存在
            required_files = ['model_path', 'feature_path', 'config_path', 'stats_path']
            for file_key in required_files:
                if not os.path.exists(latest_info[file_key]):
                    logger.error(f"模型文件不存在: {latest_info[file_key]}")
                    return False
            
            # 加载模型
            self.model = load_model(latest_info['model_path'])
            self.feature_names = load_model_features(latest_info['feature_path'])
            self.model_config = load_model_config(latest_info['config_path'])
            
            # 加载统计信息
            with open(latest_info['stats_path'], 'rb') as f:
                self.model_stats = pickle.load(f)
            
            # 构建模型信息
            self.model_info = {
                'model_name': latest_info['model_name'],
                'model_path': latest_info['model_path'],
                'model_type': self.model_stats.get('model_type', 'Unknown'),
                'training_date': self.model_stats.get('training_date', 'Unknown'),
                'features_count': self.model_stats.get('features_count', 0),
                'training_samples': self.model_stats.get('training_samples', 0),
                'target_distribution': self.model_stats.get('target_distribution', {}),
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"模型加载成功: {latest_info['model_name']}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return self.model_info
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names or []
    
    def predict(self, data: pd.DataFrame, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        进行预测
        
        Args:
            data: 输入数据
            threshold: 预测阈值
            
        Returns:
            List[Dict]: 预测结果列表
        """
        if not self.is_model_loaded():
            raise RuntimeError("模型未加载")
        
        try:
            # 数据预处理
            processed_data = self._preprocess_data(data)
            
            # 进行预测
            probabilities = self.model.predict_proba(processed_data)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            # 格式化结果
            results = []
            for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
                result = {
                    'index': i,
                    'symbol': data.iloc[i]['symbol'] if 'symbol' in data.columns else 'unknown',
                    'prediction': int(pred),
                    'probability': float(prob),
                    'risk_level': 'high' if pred == 1 else 'low',
                    'confidence': float(abs(prob - 0.5) * 2)  # 置信度：距离0.5的距离
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"预测过程中发生错误: {e}")
            raise RuntimeError(f"预测失败: {str(e)}")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        try:
            # 创建数据副本
            df = data.copy()
            
            # 确保时间列是datetime类型
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
            
            # 按symbol分组进行特征工程（简化版本，实际应该使用完整的特征工程流程）
            df = self._build_basic_features(df)
            
            # 选择模型需要的特征
            if self.feature_names:
                # 确保所有必需的特征都存在
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    logger.warning(f"缺少特征: {missing_features}")
                    # 用0填充缺少的特征
                    for feature in missing_features:
                        df[feature] = 0
                
                # 选择特征并确保顺序正确
                df_features = df[self.feature_names]
            else:
                # 如果没有特征名称，使用所有数值列
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df_features = df[numeric_columns]
            
            # 处理无穷大和NaN值
            df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_features.fillna(0, inplace=True)
            
            return df_features
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise RuntimeError(f"数据预处理失败: {str(e)}")
    
    def _build_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建基础特征（简化版本）
        
        Args:
            df: 输入数据
            
        Returns:
            pd.DataFrame: 包含特征的数据
        """
        try:
            # 基础价格特征
            df['price_change_1h'] = df.groupby('symbol')['close'].diff(1)
            df['return_1h'] = df.groupby('symbol')['close'].pct_change(1)
            df['log_return_1h'] = df.groupby('symbol')['return_1h'].transform(lambda x: np.log(1 + x))
            df['volume_change_1h'] = df.groupby('symbol')['volume'].pct_change(1)
            df['price_range'] = df['high'] - df['low']
            df['candle_body'] = abs(df['close'] - df['open'])
            
            # 滞后特征
            for lag in [1, 2, 3]:
                df[f'lag_return_{lag}h'] = df.groupby('symbol')['return_1h'].shift(lag)
                df[f'lag_volume_{lag}h'] = df.groupby('symbol')['volume'].shift(lag)
            
            # 滚动窗口特征
            for window in [6, 12, 24]:
                df[f'rolling_mean_close_{window}h'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(window).mean()
                )
                df[f'rolling_std_close_{window}h'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.rolling(window).std()
                )
                df[f'rolling_mean_volume_{window}h'] = df.groupby('symbol')['volume'].transform(
                    lambda x: x.rolling(window).mean()
                )
            
            # 简单的技术指标（这里只是示例，实际应该使用TA-Lib）
            df['RSI'] = 50  # 简化为固定值
            df['ATR'] = df['price_range']  # 简化为价格范围
            df['MACD_12_26_9'] = 0  # 简化为0
            df['MACDh_12_26_9'] = 0
            df['MACDs_12_26_9'] = 0
            df['BBL_20_2.0'] = df['close'] * 0.95
            df['BBM_20_2.0'] = df['close']
            df['BBU_20_2.0'] = df['close'] * 1.05
            df['BBB_20_2.0'] = 0.1
            df['BBP_20_2.0'] = 0.5
            
            return df
            
        except Exception as e:
            logger.error(f"特征构建失败: {e}")
            raise RuntimeError(f"特征构建失败: {str(e)}")
    
    def get_prediction_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取预测统计信息
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            Dict: 统计信息
        """
        if not predictions:
            return {}
        
        total_predictions = len(predictions)
        high_risk_count = sum(1 for p in predictions if p['prediction'] == 1)
        low_risk_count = total_predictions - high_risk_count
        
        probabilities = [p['probability'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        return {
            'total_predictions': total_predictions,
            'high_risk_count': high_risk_count,
            'low_risk_count': low_risk_count,
            'high_risk_ratio': high_risk_count / total_predictions if total_predictions > 0 else 0,
            'avg_probability': np.mean(probabilities),
            'avg_confidence': np.mean(confidences),
            'min_probability': np.min(probabilities),
            'max_probability': np.max(probabilities)
        }
