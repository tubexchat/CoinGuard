import pandas as pd
import numpy as np

# --- 配置 ---
INPUT_CSV_FILE = "data/raw/data/crypto_klines_data.csv"
OUTPUT_CSV_FILE = "data/features_crypto_data.csv"


def calculate_rsi(prices, period=14):
    """
    计算相对强弱指数 (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = 平均上涨幅度 / 平均下跌幅度
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high, low, close, period=14):
    """
    计算平均真实波幅 (ATR)
    ATR = True Range 的移动平均
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD)
    Histogram = MACD - Signal
    """
    ema_fast = prices.ewm(span=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    计算布林带
    Middle Band = SMA(period)
    Upper Band = SMA + (std_dev * std)
    Lower Band = SMA - (std_dev * std)
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对输入的K线数据DataFrame进行特征工程构建。
    核心要点：所有计算都必须在每个 'symbol' 分组内独立进行。
    """
    
    # 1. 基础数据准备
    # 确保数据按交易对和时间顺序排列，这是时间序列计算的必要前提
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.sort_values(by=['symbol', 'open_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("开始构建特征...")

    # 2. 价格和成交量衍生特征
    # 使用 .groupby('symbol') 来确保计算在每个交易对内部进行
    df['price_change_1h'] = df.groupby('symbol')['close'].diff(1)
    df['return_1h'] = df.groupby('symbol')['close'].pct_change(1)
    df['log_return_1h'] = df.groupby('symbol')['return_1h'].transform(lambda x: np.log(1 + x))
    df['volume_change_1h'] = df.groupby('symbol')['volume'].pct_change(1)
    df['price_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])

    # 3. 滞后特征 (Lag Features)
    # 滞后特征让模型可以看到过去的信息
    for lag in [1, 2, 3, 6, 12]: # 过去1, 2, 3, 6, 12小时
        df[f'lag_return_{lag}h'] = df.groupby('symbol')['return_1h'].shift(lag)
        df[f'lag_volume_{lag}h'] = df.groupby('symbol')['volume'].shift(lag)

    # 4. 滚动窗口特征 (Rolling Window Features)
    # 滚动窗口特征可以捕捉一段时间内的趋势和波动性
    for window in [6, 12, 24]: # 6小时, 12小时, 24小时窗口
        # 滚动均值
        df[f'rolling_mean_close_{window}h'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window).mean())
        # 滚动标准差 (波动率)
        df[f'rolling_std_close_{window}h'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window).std())
        # 滚动成交量均值
        df[f'rolling_mean_volume_{window}h'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean())
    
    # 5. 技术分析指标 (使用纯 pandas/numpy 实现)
    print("正在计算技术分析指标 (RSI, MACD, BBands, ATR)...")
    
    # 计算 RSI (相对强弱指数)
    def calculate_rsi_group(group):
        rsi_values = calculate_rsi(group['close'], period=14)
        return pd.Series(rsi_values, index=group.index)
    
    df['RSI'] = df.groupby('symbol', group_keys=False).apply(calculate_rsi_group, include_groups=False)
    
    # 计算 ATR (平均真实波幅) - 一个非常重要的波动率/风险指标
    def calculate_atr_group(group):
        atr_values = calculate_atr(group['high'], group['low'], group['close'], period=14)
        return pd.Series(atr_values, index=group.index)

    df['ATR'] = df.groupby('symbol', group_keys=False).apply(calculate_atr_group, include_groups=False)

    # 计算 MACD (异同移动平均线)
    def calculate_macd_group(group):
        macd, macd_signal, macd_hist = calculate_macd(group['close'], fast_period=12, slow_period=26, signal_period=9)
        result_df = pd.DataFrame({
            'MACD_12_26_9': macd,
            'MACDh_12_26_9': macd_hist,
            'MACDs_12_26_9': macd_signal
        }, index=group.index)
        return result_df

    macd_df = df.groupby('symbol', group_keys=False).apply(calculate_macd_group, include_groups=False)
    df = pd.concat([df, macd_df], axis=1)

    # 计算布林带 (Bollinger Bands)
    def calculate_bbands_group(group):
        upper, middle, lower = calculate_bollinger_bands(group['close'], period=20, std_dev=2)
        result_df = pd.DataFrame({
            'BBL_20_2.0': lower,
            'BBM_20_2.0': middle,
            'BBU_20_2.0': upper,
            'BBB_20_2.0': (upper - lower) / middle,  # 布林带宽度
            'BBP_20_2.0': (group['close'] - lower) / (upper - lower)  # 布林带位置
        }, index=group.index)
        return result_df

    bbands_df = df.groupby('symbol', group_keys=False).apply(calculate_bbands_group, include_groups=False)
    df = pd.concat([df, bbands_df], axis=1)
    
    # 6. 清理数据
    # 由于滞后和滚动计算，数据的前几行会包含NaN值，需要清理掉
    print(f"构建特征前数据量: {len(df)}")
    df.dropna(inplace=True)
    print(f"移除NaN后数据量: {len(df)}")
    
    return df


def main():
    """主函数，加载数据、构建特征并保存。"""
    print(f"正在从 {INPUT_CSV_FILE} 加载数据...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_CSV_FILE}' 未找到。请确保文件与脚本在同一目录下。")
        return
        
    featured_df = build_features(df)
    
    print(f"\n特征工程完成。总共生成了 {len(featured_df.columns)} 个特征列。")
    print("特征列展示:", featured_df.columns.tolist())
    
    print(f"\n正在将结果保存到 {OUTPUT_CSV_FILE}...")
    featured_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"✅ 数据成功保存！")

if __name__ == "__main__":
    main()