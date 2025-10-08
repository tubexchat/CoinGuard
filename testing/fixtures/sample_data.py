"""
测试数据夹具
提供各种测试场景的样本数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_crypto_data(n_samples=1000, n_symbols=3, start_date='2023-01-01'):
    """创建样本加密货币数据"""
    np.random.seed(42)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'][:n_symbols]
    data = []
    
    for symbol in symbols:
        # 生成基础价格数据
        base_price = np.random.uniform(100, 200)
        price_changes = np.random.normal(0, 0.02, n_samples)  # 2%的标准差
        
        prices = [base_price]
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # 确保价格为正
        
        # 生成OHLCV数据
        symbol_data = pd.DataFrame({
            'symbol': [symbol] * n_samples,
            'open_time': pd.date_range(start_date, periods=n_samples, freq='H'),
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'close': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'volume': np.random.uniform(1000, 5000, n_samples),
            'close_time': pd.date_range(start_date, periods=n_samples, freq='H'),
            'quote_asset_volume': np.random.uniform(100000, 500000, n_samples),
            'number_of_trades': np.random.randint(100, 1000, n_samples),
            'taker_buy_base_asset_volume': np.random.uniform(500, 2500, n_samples),
            'taker_buy_quote_asset_volume': np.random.uniform(50000, 250000, n_samples)
        })
        
        # 确保 high >= low
        symbol_data['high'] = np.maximum(symbol_data['high'], symbol_data['low'] + 0.01)
        
        data.append(symbol_data)
    
    return pd.concat(data, ignore_index=True)


def create_sample_features_data(n_samples=1000, n_symbols=3):
    """创建样本特征数据"""
    np.random.seed(42)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'][:n_symbols]
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
            
            # 基础特征
            'price_change_1h': np.random.normal(0, 2, n_samples),
            'return_1h': np.random.normal(0, 0.02, n_samples),
            'log_return_1h': np.random.normal(0, 0.02, n_samples),
            'volume_change_1h': np.random.normal(0, 0.1, n_samples),
            'price_range': np.random.uniform(1, 10, n_samples),
            'candle_body': np.random.uniform(0.5, 5, n_samples),
            
            # 滞后特征
            'lag_return_1h': np.random.normal(0, 0.02, n_samples),
            'lag_return_2h': np.random.normal(0, 0.02, n_samples),
            'lag_return_3h': np.random.normal(0, 0.02, n_samples),
            'lag_volume_1h': np.random.uniform(1000, 5000, n_samples),
            'lag_volume_2h': np.random.uniform(1000, 5000, n_samples),
            'lag_volume_3h': np.random.uniform(1000, 5000, n_samples),
            
            # 滚动窗口特征
            'rolling_mean_close_6h': np.random.uniform(100, 200, n_samples),
            'rolling_std_close_6h': np.random.uniform(1, 10, n_samples),
            'rolling_mean_volume_6h': np.random.uniform(1000, 5000, n_samples),
            'rolling_mean_close_12h': np.random.uniform(100, 200, n_samples),
            'rolling_std_close_12h': np.random.uniform(1, 10, n_samples),
            'rolling_mean_volume_12h': np.random.uniform(1000, 5000, n_samples),
            'rolling_mean_close_24h': np.random.uniform(100, 200, n_samples),
            'rolling_std_close_24h': np.random.uniform(1, 10, n_samples),
            'rolling_mean_volume_24h': np.random.uniform(1000, 5000, n_samples),
            
            # 技术指标
            'RSI': np.random.uniform(20, 80, n_samples),
            'ATR': np.random.uniform(1, 10, n_samples),
            'MACD_12_26_9': np.random.normal(0, 1, n_samples),
            'MACDh_12_26_9': np.random.normal(0, 0.5, n_samples),
            'MACDs_12_26_9': np.random.normal(0, 1, n_samples),
            'BBL_20_2.0': np.random.uniform(90, 110, n_samples),
            'BBM_20_2.0': np.random.uniform(100, 120, n_samples),
            'BBU_20_2.0': np.random.uniform(110, 130, n_samples),
            'BBB_20_2.0': np.random.uniform(0.1, 0.3, n_samples),
            'BBP_20_2.0': np.random.uniform(0, 1, n_samples),
            
            # 多空比数据
            'long_short_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'long_short_position_ratio': np.random.uniform(0.5, 2.0, n_samples)
        })
        
        # 确保 high >= low
        symbol_data['high'] = np.maximum(symbol_data['high'], symbol_data['low'] + 1)
        
        data.append(symbol_data)
    
    return pd.concat(data, ignore_index=True)


def create_sample_data_with_target(n_samples=1000, n_symbols=3, drop_percent=-0.1):
    """创建包含目标变量的样本数据"""
    df = create_sample_features_data(n_samples, n_symbols)
    
    # 创建目标变量（模拟价格下跌超过阈值的情况）
    np.random.seed(42)
    df['target'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])  # 5%的高风险样本
    
    return df


def create_malformed_data():
    """创建格式错误的数据用于测试错误处理"""
    return pd.DataFrame({
        'symbol': ['BTCUSDT', 'ETHUSDT'],
        'open_time': ['invalid_date', '2023-01-01'],
        'open': [100, -50],  # 负价格
        'high': [50, 200],   # high < low
        'low': [150, 100],
        'close': [np.inf, 150],  # 无穷大值
        'volume': [np.nan, 1000]  # NaN值
    })


def create_empty_data():
    """创建空数据用于测试边界情况"""
    return pd.DataFrame()


def create_single_symbol_data(n_samples=100):
    """创建单个交易对的数据"""
    return create_sample_crypto_data(n_samples=n_samples, n_symbols=1)


def create_high_volatility_data(n_samples=1000):
    """创建高波动率的数据"""
    np.random.seed(42)
    
    data = []
    for symbol in ['BTCUSDT']:
        # 生成高波动率的价格数据
        base_price = 20000
        price_changes = np.random.normal(0, 0.1, n_samples)  # 10%的标准差
        
        prices = [base_price]
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
        
        symbol_data = pd.DataFrame({
            'symbol': [symbol] * n_samples,
            'open_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.1) for p in prices],
            'low': [p * np.random.uniform(0.9, 1.0) for p in prices],
            'close': [p * np.random.uniform(0.95, 1.05) for p in prices],
            'volume': np.random.uniform(1000, 10000, n_samples),
            'close_time': pd.date_range('2023-01-01', periods=n_samples, freq='H')
        })
        
        symbol_data['high'] = np.maximum(symbol_data['high'], symbol_data['low'] + 0.01)
        data.append(symbol_data)
    
    return pd.concat(data, ignore_index=True)
