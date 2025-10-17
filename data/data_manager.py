"""
数据管理器
统一管理所有CSV数据的存储、读取和验证
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器类"""
    
    def __init__(self, base_dir: str = "."):
        """
        初始化数据管理器
        
        Args:
            base_dir: 项目根目录
        """
        self.base_dir = os.path.abspath(base_dir)
        self.data_dir = os.path.join(self.base_dir, "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.models_dir = os.path.join(self.data_dir, "models")
        
        # 确保目录存在
        self._ensure_directories()
        
        # 数据文件路径配置
        self.file_paths = {
            "raw_klines": os.path.join(self.raw_dir, "crypto_klines_data.csv"),
            "processed_features": os.path.join(self.processed_dir, "enhanced_features_crypto_data.csv"),
            "data_metadata": os.path.join(self.data_dir, "data_metadata.json")
        }
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [self.data_dir, self.raw_dir, self.processed_dir, self.models_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "crypto_klines_data.csv") -> str:
        """
        保存原始数据
        
        Args:
            df: 要保存的DataFrame
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.raw_dir, filename)
        
        try:
            # 验证数据
            if not self._validate_raw_data(df):
                raise ValueError("原始数据验证失败")
            
            # 保存数据
            df.to_csv(filepath, index=False)
            
            # 更新元数据
            self._update_metadata("raw_data", {
                "filepath": filepath,
                "rows": len(df),
                "columns": len(df.columns),
                "columns_list": list(df.columns),
                "last_updated": datetime.now().isoformat(),
                "data_types": df.dtypes.to_dict()
            })
            
            logger.info(f"原始数据已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存原始数据失败: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "enhanced_features_crypto_data.csv") -> str:
        """
        保存处理后的数据
        
        Args:
            df: 要保存的DataFrame
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.processed_dir, filename)
        
        try:
            # 验证数据
            if not self._validate_processed_data(df):
                raise ValueError("处理后数据验证失败")
            
            # 保存数据
            df.to_csv(filepath, index=False)
            
            # 更新元数据
            self._update_metadata("processed_data", {
                "filepath": filepath,
                "rows": len(df),
                "columns": len(df.columns),
                "columns_list": list(df.columns),
                "last_updated": datetime.now().isoformat(),
                "data_types": df.dtypes.to_dict(),
                "feature_count": len([col for col in df.columns if col not in ['symbol', 'open_time', 'close_time']])
            })
            
            logger.info(f"处理后数据已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存处理后数据失败: {e}")
            raise
    
    def load_raw_data(self, filename: str = "crypto_klines_data.csv") -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            filename: 文件名
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        filepath = os.path.join(self.raw_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"原始数据文件不存在: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"原始数据已加载: {len(df)} 行, {len(df.columns)} 列")
            return df
        except Exception as e:
            logger.error(f"加载原始数据失败: {e}")
            raise
    
    def load_processed_data(self, filename: str = "enhanced_features_crypto_data.csv") -> pd.DataFrame:
        """
        加载处理后的数据
        
        Args:
            filename: 文件名
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        filepath = os.path.join(self.processed_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"处理后数据文件不存在: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"处理后数据已加载: {len(df)} 行, {len(df.columns)} 列")
            return df
        except Exception as e:
            logger.error(f"加载处理后数据失败: {e}")
            raise
    
    def _validate_raw_data(self, df: pd.DataFrame) -> bool:
        """验证原始数据"""
        required_columns = [
            'symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'
        ]
        
        # 检查必需列
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"原始数据缺少必需列: {missing_columns}")
            return False
        
        # 检查数据完整性
        if df.empty:
            logger.error("原始数据为空")
            return False
        
        # 检查价格数据合理性
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                logger.error(f"原始数据包含非正值: {col}")
                return False
        
        # 检查high >= low
        if (df['high'] < df['low']).any():
            logger.error("原始数据包含high < low的记录")
            return False
        
        return True
    
    def _validate_processed_data(self, df: pd.DataFrame) -> bool:
        """验证处理后的数据"""
        # 检查基本结构
        if df.empty:
            logger.error("处理后数据为空")
            return False
        
        # 检查是否有特征列
        feature_columns = [col for col in df.columns if col not in ['symbol', 'open_time', 'close_time']]
        if len(feature_columns) < 10:  # 至少应该有10个特征
            logger.warning(f"处理后数据特征数量较少: {len(feature_columns)}")
        
        # 检查是否有无穷大或NaN值
        if df.isin([np.inf, -np.inf]).any().any():
            logger.warning("处理后数据包含无穷大值")
        
        if df.isna().any().any():
            logger.warning("处理后数据包含NaN值")
        
        return True
    
    def _update_metadata(self, data_type: str, metadata: Dict[str, Any]):
        """更新数据元数据"""
        metadata_file = self.file_paths["data_metadata"]
        
        try:
            # 加载现有元数据
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # 更新元数据
            all_metadata[data_type] = metadata
            
            # 保存元数据
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"更新元数据失败: {e}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        info = {
            "data_directory": self.data_dir,
            "raw_data_directory": self.raw_dir,
            "processed_data_directory": self.processed_dir,
            "models_directory": self.models_dir
        }
        
        # 检查文件存在性
        for key, filepath in self.file_paths.items():
            if os.path.exists(filepath):
                if filepath.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath)
                        info[key] = {
                            "exists": True,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "file_size": os.path.getsize(filepath)
                        }
                    except:
                        info[key] = {"exists": True, "error": "无法读取"}
                else:
                    info[key] = {"exists": True}
            else:
                info[key] = {"exists": False}
        
        return info
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "data_info": self.get_data_info()
        }
        
        # 尝试加载元数据
        metadata_file = self.file_paths["data_metadata"]
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    summary["metadata"] = json.load(f)
            except:
                summary["metadata"] = {"error": "无法读取元数据"}
        
        return summary
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for directory in [self.raw_dir, self.processed_dir]:
            for filename in os.listdir(directory):
                if filename.endswith('.csv'):
                    filepath = os.path.join(directory, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_date:
                        try:
                            os.remove(filepath)
                            logger.info(f"已删除旧数据文件: {filepath}")
                        except Exception as e:
                            logger.error(f"删除文件失败 {filepath}: {e}")
    
    def export_data_sample(self, data_type: str = "processed", n_samples: int = 100) -> str:
        """
        导出数据样本
        
        Args:
            data_type: 数据类型 ("raw" 或 "processed")
            n_samples: 样本数量
            
        Returns:
            str: 导出文件路径
        """
        try:
            if data_type == "raw":
                df = self.load_raw_data()
                filename = f"raw_data_sample_{n_samples}.csv"
            else:
                df = self.load_processed_data()
                filename = f"processed_data_sample_{n_samples}.csv"
            
            # 随机采样
            sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
            
            # 保存样本
            sample_path = os.path.join(self.data_dir, filename)
            sample_df.to_csv(sample_path, index=False)
            
            logger.info(f"数据样本已导出: {sample_path}")
            return sample_path
            
        except Exception as e:
            logger.error(f"导出数据样本失败: {e}")
            raise
