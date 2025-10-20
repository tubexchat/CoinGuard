"""
Enhanced Feature Engineering Module for Cryptocurrency Time Series Analysis

This module provides comprehensive feature engineering capabilities specifically
designed for cryptocurrency market analysis, including advanced technical indicators,
market microstructure features, and statistical measures suitable for academic research.

Authors: Research Team
License: MIT
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import talib

warnings.filterwarnings('ignore')


class EnhancedFeatureEngineering:
    """Enhanced feature engineering for cryptocurrency time series data."""
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize feature engineering class.
        
        Args:
            symbols: List of cryptocurrency symbols to process
        """
        self.symbols = symbols
        self.feature_names = []
        
    def build_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive feature set for cryptocurrency analysis.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with enhanced features
        """
        print("Building comprehensive feature set...")
        
        # Ensure proper data structure
        df = self._prepare_data(df)
        
        # Apply feature engineering by symbol
        enhanced_dfs = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('open_time')
            
            print(f"Processing features for {symbol}...")
            
            # Apply all feature engineering methods
            symbol_df = self._create_basic_features(symbol_df)
            symbol_df = self._create_technical_indicators(symbol_df)
            symbol_df = self._create_microstructure_features(symbol_df)
            symbol_df = self._create_volatility_features(symbol_df)
            symbol_df = self._create_momentum_features(symbol_df)
            symbol_df = self._create_pattern_features(symbol_df)
            symbol_df = self._create_statistical_features(symbol_df)
            symbol_df = self._create_regime_features(symbol_df)
            symbol_df = self._create_fractal_features(symbol_df)
            symbol_df = self._create_network_features(symbol_df)
            
            enhanced_dfs.append(symbol_df)
        
        # Combine all symbols
        result_df = pd.concat(enhanced_dfs, ignore_index=True)
        
        # Clean data
        result_df = self._clean_features(result_df)
        
        print(f"Feature engineering completed. Total features: {len(result_df.columns)}")
        self.feature_names = [col for col in result_df.columns if col not in 
                            ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
        
        return result_df
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature engineering."""
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values(['symbol', 'open_time']).reset_index(drop=True)
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic price and volume features."""
        
        # Price relationships
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        df['hc_ratio'] = df['high'] / df['close']
        df['lc_ratio'] = df['low'] / df['close']
        
        # Price ranges and spreads
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body_size'] / df['price_range']
        
        # Returns
        df['simple_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_return'] = df['volume'].pct_change()
        df['volume_price_ratio'] = df['volume'] / df['close']
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators using TA-Lib."""
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Momentum Indicators
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        df['RSI_7'] = talib.RSI(close, timeperiod=7)
        df['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        df['MOM_10'] = talib.MOM(close, timeperiod=10)
        df['MOM_20'] = talib.MOM(close, timeperiod=20)
        
        df['ROC_10'] = talib.ROC(close, timeperiod=10)
        df['ROC_20'] = talib.ROC(close, timeperiod=20)
        
        df['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)
        df['CCI_20'] = talib.CCI(high, low, close, timeperiod=20)
        
        df['WILLR_14'] = talib.WILLR(high, low, close, timeperiod=14)
        df['WILLR_20'] = talib.WILLR(high, low, close, timeperiod=20)
        
        # Trend Indicators
        df['SMA_5'] = talib.SMA(close, timeperiod=5)
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        
        df['EMA_5'] = talib.EMA(close, timeperiod=5)
        df['EMA_10'] = talib.EMA(close, timeperiod=10)
        df['EMA_20'] = talib.EMA(close, timeperiod=20)
        df['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(high, low)
        
        # Average Directional Index
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Volatility Indicators
        df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_20'] = talib.ATR(high, low, close, timeperiod=20)
        df['NATR_14'] = talib.NATR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(close, volume)
        df['AD'] = talib.AD(high, low, close, volume)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        
        # Bid-ask spread proxies
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['effective_spread'] = 2 * abs(df['close'] - (df['high'] + df['low']) / 2) / df['close']
        
        # Price impact measures
        df['amihud_illiquidity'] = abs(df['log_return']) / (df['volume'] * df['close'])
        df['price_impact'] = abs(df['log_return']) / np.log(1 + df['volume'])
        
        # Order flow imbalance proxies
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        # Volume-weighted measures
        df['vwap'] = (df['volume'] * df['close']).rolling(24).sum() / df['volume'].rolling(24).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Microstructure noise
        df['microstructure_noise'] = df['log_return'].rolling(5).std()
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced volatility features."""

        # Realized volatility
        for window in [6, 12, 24, 48, 168]:  # 6h, 12h, 1d, 2d, 1w
            df[f'realized_vol_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(window)
            df[f'vol_of_vol_{window}'] = df[f'realized_vol_{window}'].rolling(window//2).std()

        # Add specific windows for compatibility (12h, 24h naming)
        df['realized_vol_12h'] = df['realized_vol_12']
        df['realized_vol_24h'] = df['realized_vol_24']

        # High-frequency volatility estimators
        # Garman-Klass estimator
        for window in [12, 24]:
            df[f'gk_vol_{window}'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low']) ** 2).rolling(window).mean() -
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2).rolling(window).mean()
            )
        df['gk_vol_12h'] = df['gk_vol_12']
        df['gk_vol_24h'] = df['gk_vol_24']

        # Parkinson estimator
        for window in [12, 24]:
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(window).mean()
            )
        df['parkinson_vol_12h'] = df['parkinson_vol_12']
        df['parkinson_vol_24h'] = df['parkinson_vol_24']

        # Rogers-Satchell estimator
        df['rs_vol_24'] = np.sqrt(
            (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
             np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])).rolling(24).mean()
        )

        # Volatility clustering
        df['vol_cluster'] = df['realized_vol_24'] / df['realized_vol_24'].rolling(168).mean()

        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend features."""

        # Multi-timeframe momentum
        for period in [6, 12, 24, 48, 72, 168]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'momentum_vol_adj_{period}'] = df[f'momentum_{period}'] / df[f'realized_vol_{min(period, 48)}']

        # Momentum acceleration
        df['momentum_accel_12'] = df['momentum_12'] - df['momentum_12'].shift(6)
        df['momentum_accel_24'] = df['momentum_24'] - df['momentum_24'].shift(12)

        # Trend strength
        for window in [12, 24, 48, 168]:
            time_index = pd.Series(range(len(df)))
            df[f'trend_strength_{window}'] = df['close'].rolling(window).corr(time_index)

        # Add specific windows for compatibility (12h, 24h, 48h naming)
        df['trend_strength_12h'] = df['trend_strength_12']
        df['trend_strength_24h'] = df['trend_strength_24']
        df['trend_strength_48h'] = df['trend_strength_48']

        # Momentum oscillators
        df['price_oscillator'] = (df['EMA_10'] - df['EMA_20']) / df['EMA_20']
        df['volume_oscillator'] = (df['volume'].rolling(10).mean() - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).mean()

        return df
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick pattern features."""
        
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        close = df['close'].values
        
        # Candlestick patterns
        df['doji'] = talib.CDLDOJI(open_price, high, low, close)
        df['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        df['engulfing_bullish'] = talib.CDLENGULFING(open_price, high, low, close)
        df['harami'] = talib.CDLHARAMI(open_price, high, low, close)
        df['piercing'] = talib.CDLPIERCING(open_price, high, low, close)
        df['dark_cloud'] = talib.CDLDARKCLOUDCOVER(open_price, high, low, close)
        df['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        df['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        
        # Pattern aggregation
        pattern_columns = ['doji', 'hammer', 'shooting_star', 'engulfing_bullish', 
                          'harami', 'piercing', 'dark_cloud', 'morning_star', 'evening_star']
        df['bullish_patterns'] = df[['hammer', 'engulfing_bullish', 'piercing', 'morning_star']].sum(axis=1)
        df['bearish_patterns'] = df[['shooting_star', 'dark_cloud', 'evening_star']].sum(axis=1)
        df['neutral_patterns'] = df[['doji', 'harami']].sum(axis=1)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        
        # Skewness and kurtosis
        for window in [24, 48, 168]:
            df[f'skew_{window}'] = df['log_return'].rolling(window).skew()
            df[f'kurt_{window}'] = df['log_return'].rolling(window).kurt()
        
        # Percentile features
        for window in [24, 48]:
            for percentile in [10, 25, 75, 90]:
                df[f'price_pct_{percentile}_{window}'] = df['close'].rolling(window).quantile(percentile/100)
                df[f'volume_pct_{percentile}_{window}'] = df['volume'].rolling(window).quantile(percentile/100)
        
        # Statistical tests
        df['jarque_bera_24'] = df['log_return'].rolling(24).apply(
            lambda x: stats.jarque_bera(x.dropna())[1] if len(x.dropna()) > 3 else np.nan
        )
        
        # Autocorrelation
        for lag in [1, 6, 12, 24]:
            df[f'autocorr_lag_{lag}'] = df['log_return'].rolling(48).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan
            )
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime identification features."""
        
        # Volatility regimes
        vol_24h = df['realized_vol_24']
        df['vol_regime_low'] = (vol_24h <= vol_24h.rolling(168).quantile(0.33)).astype(int)
        df['vol_regime_high'] = (vol_24h >= vol_24h.rolling(168).quantile(0.67)).astype(int)
        
        # Trend regimes
        sma_fast = df['SMA_10']
        sma_slow = df['SMA_50']
        df['trend_regime_bull'] = (sma_fast > sma_slow).astype(int)
        df['trend_regime_bear'] = (sma_fast < sma_slow).astype(int)
        
        # Market microstructure regimes
        spread_24h = df['hl_spread'].rolling(24).mean()
        df['liquidity_regime_low'] = (spread_24h >= spread_24h.rolling(168).quantile(0.75)).astype(int)
        df['liquidity_regime_high'] = (spread_24h <= spread_24h.rolling(168).quantile(0.25)).astype(int)
        
        return df
    
    def _create_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fractal and complexity features."""
        
        # Hurst exponent approximation
        def hurst_approx(series, max_lag=20):
            """Approximate Hurst exponent calculation."""
            series = series.dropna()
            if len(series) < max_lag * 2:
                return np.nan
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['hurst_exp_48'] = df['log_return'].rolling(96).apply(
            lambda x: hurst_approx(x, max_lag=20)
        )
        
        # Fractal dimension approximation
        def fractal_dim_approx(series, max_lag=10):
            """Approximate fractal dimension."""
            series = series.dropna()
            if len(series) < max_lag * 2:
                return np.nan
            
            # Simplified box-counting method
            scales = np.arange(1, max_lag)
            counts = []
            for scale in scales:
                boxes = len(series) // scale
                if boxes > 1:
                    counts.append(boxes)
                else:
                    counts.append(1)
            
            if len(counts) < 2:
                return np.nan
            
            try:
                poly = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
                return -poly[0]
            except:
                return np.nan
        
        df['fractal_dim_48'] = df['close'].rolling(96).apply(
            lambda x: fractal_dim_approx(x, max_lag=10)
        )
        
        return df
    
    def _create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create network-based features (correlations and relationships)."""
        
        # Auto-correlation features
        for lag in [1, 6, 12, 24]:
            df[f'price_autocorr_{lag}'] = df['close'].rolling(48).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            df[f'volume_autocorr_{lag}'] = df['volume'].rolling(48).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Price-volume correlation
        df['price_volume_corr_24'] = df['close'].rolling(24).corr(df['volume'])
        df['return_volume_corr_24'] = df['log_return'].rolling(24).corr(df['volume_return'])
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with too many NaN values
        nan_threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=nan_threshold)
        
        print(f"Data shape after cleaning: {df.shape}")
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis."""
        
        feature_groups = {
            'basic': [f for f in self.feature_names if any(x in f for x in ['ratio', 'return', 'range', 'body', 'shadow'])],
            'technical': [f for f in self.feature_names if any(x in f for x in ['RSI', 'MACD', 'BB', 'SMA', 'EMA', 'ATR', 'ADX'])],
            'microstructure': [f for f in self.feature_names if any(x in f for x in ['spread', 'impact', 'pressure', 'vwap', 'noise'])],
            'volatility': [f for f in self.feature_names if any(x in f for x in ['vol', 'gk_', 'parkinson', 'rs_'])],
            'momentum': [f for f in self.feature_names if any(x in f for x in ['momentum', 'trend', 'oscillator'])],
            'patterns': [f for f in self.feature_names if any(x in f for x in ['doji', 'hammer', 'star', 'engulf', 'pattern'])],
            'statistical': [f for f in self.feature_names if any(x in f for x in ['skew', 'kurt', 'pct_', 'jarque', 'autocorr'])],
            'regime': [f for f in self.feature_names if 'regime' in f],
            'fractal': [f for f in self.feature_names if any(x in f for x in ['hurst', 'fractal'])],
            'network': [f for f in self.feature_names if any(x in f for x in ['corr', 'network'])]
        }
        
        return feature_groups


def main():
    """Main function for testing feature engineering."""
    
    # Configuration
    INPUT_CSV_FILE = "../raw/data/crypto_klines_data.csv"
    OUTPUT_CSV_FILE = "../enhanced_features_crypto_data.csv"
    
    print(f"Loading data from {INPUT_CSV_FILE}...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV_FILE}' not found.")
        return
    
    # Initialize feature engineering
    feature_engineer = EnhancedFeatureEngineering()
    
    # Build comprehensive features
    enhanced_df = feature_engineer.build_comprehensive_features(df)
    
    # Get feature groups
    feature_groups = feature_engineer.get_feature_groups()
    
    print(f"\nFeature engineering completed!")
    print(f"Total features created: {len(feature_engineer.feature_names)}")
    print("\nFeature groups:")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)} features")
    
    # Save enhanced dataset
    print(f"\nSaving enhanced dataset to {OUTPUT_CSV_FILE}...")
    enhanced_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print("✅ Enhanced dataset saved successfully!")
    
    # Save feature information
    feature_info = {
        'total_features': len(feature_engineer.feature_names),
        'feature_groups': feature_groups,
        'all_features': feature_engineer.feature_names
    }
    
    import json
    with open("../enhanced_feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Feature information saved!")


if __name__ == "__main__":
    main()