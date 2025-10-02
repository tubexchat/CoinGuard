import pandas as pd
import talib
import numpy as np

# --- 配置 ---
INPUT_CSV_FILE = "data/crypto_klines_data.csv"
OUTPUT_CSV_FILE = "data/features_crypto_data.csv"


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
    
    # 5. 技术分析指标 (使用 TA-Lib 库)
    # TA-Lib 需要 numpy 数组作为输入，所以需要转换数据类型
    print("正在计算技术分析指标 (RSI, MACD, BBands, ATR)...")
    
    # 计算 RSI (相对强弱指数)
    def calculate_rsi(group):
        close_array = group['close'].values.astype(np.float64)
        rsi_values = talib.RSI(close_array, timeperiod=14)
        return pd.Series(rsi_values, index=group.index)
    
    df['RSI'] = df.groupby('symbol', group_keys=False).apply(calculate_rsi)
    
    # 计算 ATR (平均真实波幅) - 一个非常重要的波动率/风险指标
    def calculate_atr(group):
        high_array = group['high'].values.astype(np.float64)
        low_array = group['low'].values.astype(np.float64)
        close_array = group['close'].values.astype(np.float64)
        atr_values = talib.ATR(high_array, low_array, close_array, timeperiod=14)
        return pd.Series(atr_values, index=group.index)

    df['ATR'] = df.groupby('symbol', group_keys=False).apply(calculate_atr)

    # 计算 MACD (异同移动平均线)
    def calculate_macd(group):
        close_array = group['close'].values.astype(np.float64)
        macd, macd_signal, macd_hist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
        result_df = pd.DataFrame({
            'MACD_12_26_9': macd,
            'MACDh_12_26_9': macd_hist,
            'MACDs_12_26_9': macd_signal
        }, index=group.index)
        return result_df

    macd_df = df.groupby('symbol', group_keys=False).apply(calculate_macd)
    df = pd.concat([df, macd_df], axis=1)

    # 计算布林带 (Bollinger Bands)
    def calculate_bbands(group):
        close_array = group['close'].values.astype(np.float64)
        upper, middle, lower = talib.BBANDS(close_array, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        result_df = pd.DataFrame({
            'BBL_20_2.0': lower,
            'BBM_20_2.0': middle,
            'BBU_20_2.0': upper,
            'BBB_20_2.0': (upper - lower) / middle,  # 布林带宽度
            'BBP_20_2.0': (close_array - lower) / (upper - lower)  # 布林带位置
        }, index=group.index)
        return result_df

    bbands_df = df.groupby('symbol', group_keys=False).apply(calculate_bbands)
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