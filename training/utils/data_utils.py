"""
数据处理工具函数
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """从CSV文件加载数据并进行初步处理。"""
    print(f"正在从 {filepath} 加载数据...")
    try:
        df = pd.read_csv(filepath)
        df['open_time'] = pd.to_datetime(df['open_time'])
        return df
    except FileNotFoundError:
        print(f"错误: 输入文件 '{filepath}' 未找到。请确保路径正确。")
        return None


def create_target_variable(df: pd.DataFrame, lookahead: int, drop_percent: float) -> pd.DataFrame:
    """为数据集创建目标变量 y。"""
    print(f"正在创建目标变量 (未来{lookahead}小时内下跌超过 {-drop_percent:.0%})...")
    
    if lookahead == 1:
        # 当只看未来1小时时，不需要滚动窗口，直接shift即可，效率更高
        future_lows = df.groupby('symbol')['low'].shift(-1)
    else:
        future_lows = df.groupby('symbol')['low'].transform(
            lambda x: x.shift(-lookahead).rolling(window=lookahead, min_periods=1).min()
        )
    
    price_drop_ratio = (future_lows / df['close']) - 1
    df['target'] = (price_drop_ratio < drop_percent).astype(int)
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """准备特征矩阵X和目标向量y。"""
    features_to_drop = [
        'symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'target'
    ]
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['target']
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    print("特征 (X) 和目标 (y) 已准备就绪。")
    print(f"特征数量: {len(X.columns)}")
    return X, y


def split_data_temporal(X: pd.DataFrame, y: pd.Series, train_ratio: float, val_ratio: float) -> Tuple:
    """按时间顺序划分数据集。"""
    print("\n正在按时间顺序划分数据集...")
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size + val_size], y.iloc[train_size:train_size + val_size]
    X_test, y_test = X.iloc[train_size + val_size:], y.iloc[train_size + val_size:]

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def validate_data_quality(df: pd.DataFrame) -> bool:
    """验证数据质量"""
    print("正在验证数据质量...")
    
    # 检查必要的列是否存在
    required_columns = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: 缺少必要的列: {missing_columns}")
        return False
    
    # 检查数据是否为空
    if df.empty:
        print("错误: 数据为空")
        return False
    
    # 检查是否有重复的时间戳
    duplicates = df.duplicated(subset=['symbol', 'open_time']).sum()
    if duplicates > 0:
        print(f"警告: 发现 {duplicates} 个重复的时间戳")
    
    # 检查价格数据的合理性
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if (df[col] <= 0).any():
            print(f"警告: {col} 列包含非正值")
    
    # 检查high >= low
    if (df['high'] < df['low']).any():
        print("警告: 发现 high < low 的数据点")
    
    print("数据质量验证完成")
    return True
