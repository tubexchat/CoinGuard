import requests
import pandas as pd
import time
from tqdm import tqdm

# --- 配置 ---
TICKER_API_URL = "https://api.lewiszhang.top/ticker/24hr"
KLINES_API_URL = "https://api.lewiszhang.top/klines"
OUTPUT_CSV_FILE = "data/crypto_klines_data.csv"
REQUEST_DELAY_SECONDS = 0.1 # 每次API请求之间的延迟，防止请求过于频繁

# K线数据在CSV文件中的列名
# 原始K线有12个字段，我们在最前面加上了'symbol'
KLINE_COLUMNS = [
    'symbol',           # 交易对名称
    'open_time',        # 开盘时间 (Unix apoch)
    'open',             # 开盘价
    'high',             # 最高价
    'low',              # 最低价
    'close',            # 收盘价
    'volume',           # 成交量
    'close_time',       # 收盘时间 (Unix apoch)
    'quote_asset_volume',# 成交额
    'number_of_trades', # 成交笔数
    'taker_buy_base_asset_volume',  # 主动买入成交量
    'taker_buy_quote_asset_volume', # 主动买入成交额
    'ignore'            # 忽略字段
]


def fetch_all_symbols():
    """从API获取所有可用的交易对符号。"""
    print("正在获取所有交易对列表...")
    try:
        response = requests.get(TICKER_API_URL)
        response.raise_for_status()  # 如果请求失败 (例如 404, 500), 则会抛出异常
        tickers_data = response.json()
        symbols = [item['symbol'] for item in tickers_data]
        print(f"成功获取到 {len(symbols)} 个交易对。")
        return symbols
    except requests.exceptions.RequestException as e:
        print(f"获取交易对列表失败: {e}")
        return []

def fetch_klines_for_symbol(symbol: str, interval: str = '1h', limit: int = 1000):
    """为单个交易对获取K线数据。"""
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    try:
        response = requests.get(KLINES_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # 打印错误信息但程序不中断
        print(f"  - 警告: 获取 {symbol} 的K线数据失败: {e}")
        return None

def main():
    """主函数，执行整个数据获取和存储流程。"""
    # 1. 获取所有交易对
    symbols = fetch_all_symbols()
    if not symbols:
        print("没有获取到任何交易对，程序退出。")
        return

    all_klines_data = []

    # 2. 遍历每个交易对，获取K线数据
    # 使用tqdm创建进度条
    for symbol in tqdm(symbols, desc="正在获取K线数据"):
        klines = fetch_klines_for_symbol(symbol)
        
        if klines:
            # 为每一条K线数据前面加上交易对名称
            for kline_record in klines:
                processed_record = [symbol] + kline_record
                all_klines_data.append(processed_record)
        
        # 友好请求，在每次API调用后稍作等待
        time.sleep(REQUEST_DELAY_SECONDS)

    # 3. 将所有数据转换为Pandas DataFrame并保存到CSV
    if not all_klines_data:
        print("未能获取到任何K线数据，无法生成CSV文件。")
        return

    print("\n正在将数据写入CSV文件...")
    df = pd.DataFrame(all_klines_data, columns=KLINE_COLUMNS)

    # 数据类型转换，将价格和交易量等字段转换为数值类型以便后续分析
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # errors='coerce' 会将无法转换的值变为NaN

    # 将时间戳转换为更易读的日期时间格式 (可选，如果需要)
    # df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    # df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # 忽略最后一个字段 'ignore'
    df = df.drop(columns=['ignore'])

    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"✅ 数据成功保存到文件: {OUTPUT_CSV_FILE}")
    print(f"总共获取并保存了 {len(df)} 条K线记录。")


if __name__ == "__main__":
    main()