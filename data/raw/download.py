import requests
import pandas as pd
import time
from tqdm import tqdm
import os
import json
from datetime import datetime

# --- 配置 ---
TICKER_API_URL = "https://api.lewiszhang.top/ticker/24hr"
KLINES_API_URL = "https://api.lewiszhang.top/klines"
RATIO_API_URL = "https://api.lewiszhang.top/topLongShortAccountRatio"
POSITION_RATIO_API_URL = "https://api.lewiszhang.top/topLongShortPositionRatio"
OUTPUT_CSV_FILE = "data/crypto_klines_data.csv"
REQUEST_DELAY_SECONDS = 0.1 # 每次API请求之间的延迟，防止请求过于频繁

def make_request_with_progress(url, params=None, desc="下载中"):
    """带进度条的HTTP请求函数"""
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建进度条
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
                    pbar.update(len(chunk))
        
        # 解析JSON响应 - 使用收集到的内容而不是response.json()
        return json.loads(content.decode('utf-8'))
    except requests.exceptions.RequestException as e:
        print(f"  - 警告: 请求失败: {e}")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  - 警告: 解析响应失败: {e}")
        return None

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
    tickers_data = make_request_with_progress(TICKER_API_URL, desc="获取交易对列表")
    if tickers_data:
        symbols = [item['symbol'] for item in tickers_data]
        print(f"✅ 成功获取到 {len(symbols)} 个交易对。")
        return symbols
    else:
        print("❌ 获取交易对列表失败")
        return []

def fetch_klines_for_symbol(symbol: str, interval: str = '1h', limit: int = 1000):
    """为单个交易对获取K线数据。"""
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    return make_request_with_progress(KLINES_API_URL, params=params, desc=f"获取 {symbol} K线数据")

def fetch_account_ratio_for_symbol(symbol: str, period: str = '1h', limit: int = 1000):
    """为单个交易对获取账户多空比时间序列。"""
    params = {
        'symbol': symbol,
        'period': period,
        'limit': limit
    }
    return make_request_with_progress(RATIO_API_URL, params=params, desc=f"获取 {symbol} 账户多空比")

def fetch_position_ratio_for_symbol(symbol: str, period: str = '1h', limit: int = 1000):
    """为单个交易对获取持仓多空比时间序列。"""
    params = {
        'symbol': symbol,
        'period': period,
        'limit': limit
    }
    return make_request_with_progress(POSITION_RATIO_API_URL, params=params, desc=f"获取 {symbol} 持仓多空比")

def main():
    """主函数，执行整个数据获取和存储流程。"""
    start_time = datetime.now()
    print(f"🚀 开始下载原始数据 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 获取所有交易对
    symbols = fetch_all_symbols()
    if not symbols:
        print("❌ 没有获取到任何交易对，程序退出。")
        return

    all_klines_data = []
    failed_symbols = []

    # 2. 遍历每个交易对，获取K线数据
    # 使用tqdm创建进度条，显示更详细的信息
    with tqdm(symbols, desc="📊 获取K线数据", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for symbol in pbar:
            pbar.set_postfix_str(f"当前: {symbol}")
            klines = fetch_klines_for_symbol(symbol)
            
            if klines:
                # 为每一条K线数据前面加上交易对名称
                for kline_record in klines:
                    processed_record = [symbol] + kline_record
                    all_klines_data.append(processed_record)
            else:
                failed_symbols.append(symbol)
            
            # 友好请求，在每次API调用后稍作等待
            time.sleep(REQUEST_DELAY_SECONDS)

    # 显示K线数据获取结果
    print(f"\n📈 K线数据获取完成:")
    print(f"   ✅ 成功: {len(symbols) - len(failed_symbols)} 个交易对")
    if failed_symbols:
        print(f"   ❌ 失败: {len(failed_symbols)} 个交易对")
        print(f"   失败的交易对: {', '.join(failed_symbols[:5])}{'...' if len(failed_symbols) > 5 else ''}")

    # 3. 将所有数据转换为Pandas DataFrame并保存到CSV
    if not all_klines_data:
        print("❌ 未能获取到任何K线数据，无法生成CSV文件。")
        return

    print(f"\n💾 正在处理 {len(all_klines_data)} 条K线记录...")
    with tqdm(total=1, desc="转换数据格式", bar_format='{l_bar}{bar}| {desc}') as pbar:
        df = pd.DataFrame(all_klines_data, columns=KLINE_COLUMNS)
        pbar.update(1)

    # 数据类型转换，将价格和交易量等字段转换为数值类型以便后续分析
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    
    with tqdm(numeric_cols, desc="转换数据类型", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {desc}') as pbar:
        for col in pbar:
            pbar.set_postfix_str(f"处理: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce') # errors='coerce' 会将无法转换的值变为NaN

    # 将时间戳转换为更易读的日期时间格式 (可选，如果需要)
    # df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    # df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # 忽略最后一个字段 'ignore'
    df = df.drop(columns=['ignore'])

    # 4. 获取并合并账户多空比(按 symbol + open_time 对齐)
    print(f"\n📊 正在获取账户多空比数据...")
    ratio_records = []
    unique_symbols = df['symbol'].unique().tolist()
    failed_ratio_symbols = []
    
    with tqdm(unique_symbols, desc="获取账户多空比", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for symbol in pbar:
            pbar.set_postfix_str(f"当前: {symbol}")
            ratios = fetch_account_ratio_for_symbol(symbol, period='1h', limit=1000)
            if ratios:
                for r in ratios:
                    ratio_records.append({
                        'symbol': r.get('symbol', symbol),
                        'timestamp': r.get('timestamp'),
                        'longShortRatio': r.get('longShortRatio')
                    })
            else:
                failed_ratio_symbols.append(symbol)
            time.sleep(REQUEST_DELAY_SECONDS)

    if ratio_records:
        ratio_df = pd.DataFrame(ratio_records)
        ratio_df['timestamp'] = pd.to_numeric(ratio_df['timestamp'], errors='coerce')
        ratio_df['longShortRatio'] = pd.to_numeric(ratio_df['longShortRatio'], errors='coerce')

        df = df.merge(
            ratio_df[['symbol', 'timestamp', 'longShortRatio']],
            left_on=['symbol', 'open_time'],
            right_on=['symbol', 'timestamp'],
            how='left'
        )
        df = df.drop(columns=['timestamp'])
        df = df.rename(columns={'longShortRatio': 'long_short_ratio'})
    else:
        df['long_short_ratio'] = float('nan')

    # 5. 获取并合并持仓多空比(按 symbol + open_time 对齐)
    print(f"\n📊 正在获取持仓多空比数据...")
    pos_ratio_records = []
    failed_pos_ratio_symbols = []
    
    with tqdm(unique_symbols, desc="获取持仓多空比", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for symbol in pbar:
            pbar.set_postfix_str(f"当前: {symbol}")
            pos_ratios = fetch_position_ratio_for_symbol(symbol, period='1h', limit=1000)
            if pos_ratios:
                for r in pos_ratios:
                    pos_ratio_records.append({
                        'symbol': r.get('symbol', symbol),
                        'timestamp': r.get('timestamp'),
                        'longShortRatio': r.get('longShortRatio')
                    })
            else:
                failed_pos_ratio_symbols.append(symbol)
            time.sleep(REQUEST_DELAY_SECONDS)

    if pos_ratio_records:
        pos_ratio_df = pd.DataFrame(pos_ratio_records)
        pos_ratio_df['timestamp'] = pd.to_numeric(pos_ratio_df['timestamp'], errors='coerce')
        pos_ratio_df['longShortRatio'] = pd.to_numeric(pos_ratio_df['longShortRatio'], errors='coerce')

        df = df.merge(
            pos_ratio_df[['symbol', 'timestamp', 'longShortRatio']],
            left_on=['symbol', 'open_time'],
            right_on=['symbol', 'timestamp'],
            how='left',
            suffixes=(None, '_pos')
        )
        df = df.drop(columns=['timestamp'])
        df = df.rename(columns={'longShortRatio': 'long_short_position_ratio'})
    else:
        df['long_short_position_ratio'] = float('nan')

    # 6. 保存数据到CSV文件
    print(f"\n💾 正在保存数据到文件...")
    with tqdm(total=1, desc="保存CSV文件", bar_format='{l_bar}{bar}| {desc}') as pbar:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        pbar.update(1)
    
    # 计算总耗时
    end_time = datetime.now()
    total_time = end_time - start_time
    
    # 显示最终结果
    print("\n" + "=" * 60)
    print("🎉 数据下载完成!")
    print(f"📁 文件保存位置: {OUTPUT_CSV_FILE}")
    print(f"📊 总记录数: {len(df):,} 条K线记录")
    print(f"⏱️  总耗时: {total_time}")
    print(f"🕐 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示统计信息
    print(f"\n📈 数据统计:")
    print(f"   • 交易对数量: {df['symbol'].nunique()}")
    print(f"   • 时间范围: {df['open_time'].min()} - {df['open_time'].max()}")
    print(f"   • 账户多空比记录: {df['long_short_ratio'].notna().sum():,}")
    print(f"   • 持仓多空比记录: {df['long_short_position_ratio'].notna().sum():,}")
    
    # 显示失败统计
    if failed_symbols or failed_ratio_symbols or failed_pos_ratio_symbols:
        print(f"\n⚠️  失败统计:")
        if failed_symbols:
            print(f"   • K线数据失败: {len(failed_symbols)} 个交易对")
        if failed_ratio_symbols:
            print(f"   • 账户多空比失败: {len(failed_ratio_symbols)} 个交易对")
        if failed_pos_ratio_symbols:
            print(f"   • 持仓多空比失败: {len(failed_pos_ratio_symbols)} 个交易对")
    
    print("=" * 60)


if __name__ == "__main__":
    main()