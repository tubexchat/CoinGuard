"""
Advanced Risk Management and Backtesting Framework

This module provides comprehensive risk management tools and backtesting capabilities
for cryptocurrency trading strategies based on machine learning predictions.

Authors: Research Team
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for trading strategies."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    annualized_volatility: float
    downside_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    
    # Win/Loss metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    
    # Advanced metrics
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    tail_ratio: float
    skewness: float
    kurtosis: float
    
    # Statistical tests
    jarque_bera_stat: float
    jarque_bera_pvalue: float


@dataclass
class BacktestConfig:
    """Configuration for backtesting strategy."""
    
    # Basic settings
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% slippage
    
    # Position sizing
    position_sizing_method: str = "fixed_fraction"  # "fixed_fraction", "kelly", "risk_parity"
    max_position_size: float = 0.1  # 10% of capital per position
    risk_per_trade: float = 0.02   # 2% risk per trade
    
    # Risk management
    stop_loss: Optional[float] = 0.05     # 5% stop loss
    take_profit: Optional[float] = 0.10   # 10% take profit
    max_holding_period: Optional[int] = 168  # 7 days in hours
    
    # Strategy parameters
    prediction_threshold: float = 0.6     # Minimum prediction confidence
    min_prediction_edge: float = 0.1      # Minimum edge over random
    
    # Execution settings
    rebalance_frequency: str = "1H"       # Rebalancing frequency
    lookback_window: int = 24             # Hours for rolling calculations


class PositionManager:
    """Manages trading positions and risk."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize position manager.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.positions = {}  # {symbol: position_info}
        self.cash = config.initial_capital
        self.total_value = config.initial_capital
        
    def calculate_position_size(self, symbol: str, price: float, 
                              prediction_confidence: float,
                              current_volatility: float = 0.02) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            price: Current price
            prediction_confidence: Model confidence (0-1)
            current_volatility: Current asset volatility
            
        Returns:
            Position size in base currency
        """
        if self.config.position_sizing_method == "fixed_fraction":
            # Fixed fraction of total portfolio value
            max_position_value = self.total_value * self.config.max_position_size
            return min(max_position_value, self.cash)
        
        elif self.config.position_sizing_method == "kelly":
            # Kelly criterion
            # Simplified: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            win_prob = prediction_confidence
            loss_prob = 1 - prediction_confidence
            avg_win_loss_ratio = 1.5  # Assumed average win/loss ratio
            
            kelly_fraction = (avg_win_loss_ratio * win_prob - loss_prob) / avg_win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
            
            return self.total_value * kelly_fraction
        
        elif self.config.position_sizing_method == "risk_parity":
            # Risk parity based on volatility
            target_risk = self.config.risk_per_trade
            position_value = (self.total_value * target_risk) / current_volatility
            
            return min(position_value, self.total_value * self.config.max_position_size)
        
        else:
            raise ValueError(f"Unknown position sizing method: {self.config.position_sizing_method}")
    
    def open_position(self, symbol: str, size: float, price: float, 
                     timestamp: datetime, prediction_confidence: float) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            size: Position size in base currency
            price: Entry price
            timestamp: Entry timestamp
            prediction_confidence: Model confidence
            
        Returns:
            True if position opened successfully
        """
        # Calculate costs
        commission_cost = size * self.config.commission
        slippage_cost = size * self.config.slippage
        total_cost = size + commission_cost + slippage_cost
        
        # Check if we have enough cash
        if total_cost > self.cash:
            return False
        
        # Open position
        self.positions[symbol] = {
            'size': size,
            'entry_price': price,
            'entry_time': timestamp,
            'confidence': prediction_confidence,
            'unrealized_pnl': 0.0,
            'stop_loss': price * (1 - self.config.stop_loss) if self.config.stop_loss else None,
            'take_profit': price * (1 + self.config.take_profit) if self.config.take_profit else None
        }
        
        # Update cash
        self.cash -= total_cost
        
        return True
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, 
                      reason: str = "signal") -> Optional[Dict]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
            
        Returns:
            Trade result dictionary
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        size = position['size']
        entry_price = position['entry_price']
        pnl = size * (price - entry_price) / entry_price
        
        # Calculate costs
        commission_cost = size * self.config.commission
        slippage_cost = size * self.config.slippage
        total_cost = commission_cost + slippage_cost
        
        # Net P&L
        net_pnl = pnl - total_cost
        
        # Create trade record
        trade_result = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'size': size,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / size,
            'holding_period': timestamp - position['entry_time'],
            'confidence': position['confidence'],
            'exit_reason': reason
        }
        
        # Update cash
        self.cash += size + net_pnl
        
        # Remove position
        del self.positions[symbol]
        
        return trade_result
    
    def update_positions(self, market_data: Dict[str, float], timestamp: datetime) -> List[Dict]:
        """
        Update all positions with current market data.
        
        Args:
            market_data: {symbol: current_price}
            timestamp: Current timestamp
            
        Returns:
            List of closed trades (if any)
        """
        closed_trades = []
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]
            entry_price = position['entry_price']
            
            # Update unrealized P&L
            size = position['size']
            unrealized_pnl = size * (current_price - entry_price) / entry_price
            position['unrealized_pnl'] = unrealized_pnl
            
            # Check stop loss
            if (position['stop_loss'] is not None and 
                current_price <= position['stop_loss']):
                trade = self.close_position(symbol, current_price, timestamp, "stop_loss")
                if trade:
                    closed_trades.append(trade)
                symbols_to_close.append(symbol)
                continue
            
            # Check take profit
            if (position['take_profit'] is not None and 
                current_price >= position['take_profit']):
                trade = self.close_position(symbol, current_price, timestamp, "take_profit")
                if trade:
                    closed_trades.append(trade)
                symbols_to_close.append(symbol)
                continue
            
            # Check max holding period
            if (self.config.max_holding_period is not None and
                (timestamp - position['entry_time']).total_seconds() / 3600 >= self.config.max_holding_period):
                trade = self.close_position(symbol, current_price, timestamp, "max_holding")
                if trade:
                    closed_trades.append(trade)
                symbols_to_close.append(symbol)
        
        # Update total value
        unrealized_total = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        position_values = sum(pos['size'] for pos in self.positions.values())
        self.total_value = self.cash + position_values + unrealized_total
        
        return closed_trades


class RiskAnalyzer:
    """Analyzes trading strategy risk metrics."""
    
    @staticmethod
    def calculate_comprehensive_metrics(trades_df: pd.DataFrame, 
                                      returns_series: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            trades_df: DataFrame with trade results
            returns_series: Time series of strategy returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            RiskMetrics object
        """
        if len(trades_df) == 0 or len(returns_series) == 0:
            return RiskMetrics(
                total_return=0.0, annualized_return=0.0, cumulative_return=0.0,
                volatility=0.0, annualized_volatility=0.0, downside_volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                max_drawdown=0.0, max_drawdown_duration=0, current_drawdown=0.0,
                win_rate=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0, win_loss_ratio=0.0,
                var_95=0.0, cvar_95=0.0, tail_ratio=0.0, skewness=0.0, kurtosis=0.0,
                jarque_bera_stat=0.0, jarque_bera_pvalue=1.0
            )
        
        # Basic return metrics
        total_return = returns_series.iloc[-1] - returns_series.iloc[0]
        cumulative_return = (1 + returns_series).cumprod().iloc[-1] - 1
        
        # Annualized metrics (assuming hourly data)
        periods_per_year = 365 * 24  # Hours per year
        n_periods = len(returns_series)
        years = n_periods / periods_per_year
        
        annualized_return = (1 + cumulative_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility metrics
        volatility = returns_series.std()
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        
        # Downside volatility
        downside_returns = returns_series[returns_series < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
        
        # Risk-adjusted returns
        risk_free_rate = 0.02  # Assumed 2% annual risk-free rate
        excess_returns = returns_series - risk_free_rate / periods_per_year
        
        sharpe_ratio = (excess_returns.mean() * periods_per_year) / annualized_volatility if annualized_volatility > 0 else 0
        sortino_ratio = (excess_returns.mean() * periods_per_year) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns_series).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1]
        
        # Max drawdown duration
        is_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade-based metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Value at Risk and Conditional VaR
        var_95 = np.percentile(returns_series, 5)
        cvar_95 = returns_series[returns_series <= var_95].mean()
        
        # Tail ratio
        top_10_pct = np.percentile(returns_series, 90)
        bottom_10_pct = np.percentile(returns_series, 10)
        tail_ratio = abs(top_10_pct / bottom_10_pct) if bottom_10_pct < 0 else 0
        
        # Distribution moments
        skewness = stats.skew(returns_series)
        kurtosis = stats.kurtosis(returns_series)
        
        # Normality test
        jb_stat, jb_pvalue = stats.jarque_bera(returns_series)
        
        return RiskMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            downside_volatility=downside_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue
        )


class Backtester:
    """Comprehensive backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.position_manager = PositionManager(config)
        self.trades = []
        self.portfolio_history = []
        
    def run_backtest(self, market_data: pd.DataFrame, 
                    predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive backtest.
        
        Args:
            market_data: DataFrame with OHLCV data
            predictions: DataFrame with model predictions
            
        Returns:
            Comprehensive backtest results
        """
        print("Starting backtest...")
        print(f"Market data shape: {market_data.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        # Ensure data alignment
        market_data = market_data.sort_values(['symbol', 'open_time'])
        predictions = predictions.sort_values(['symbol', 'open_time'])
        
        # Get unique timestamps
        timestamps = sorted(market_data['open_time'].unique())
        
        for i, timestamp in enumerate(timestamps):
            if i % 1000 == 0:
                print(f"Processing timestamp {i+1}/{len(timestamps)}: {timestamp}")
            
            # Get current market data
            current_market = market_data[market_data['open_time'] == timestamp]
            current_predictions = predictions[predictions['open_time'] == timestamp]
            
            if current_market.empty:
                continue
            
            # Create price dictionary
            price_data = dict(zip(current_market['symbol'], current_market['close']))
            
            # Update existing positions
            closed_trades = self.position_manager.update_positions(price_data, timestamp)
            self.trades.extend(closed_trades)
            
            # Process new signals
            if not current_predictions.empty:
                self._process_signals(current_predictions, price_data, timestamp)
            
            # Record portfolio state
            self._record_portfolio_state(timestamp, price_data)
        
        # Close all remaining positions
        final_timestamp = timestamps[-1]
        final_market = market_data[market_data['open_time'] == final_timestamp]
        final_prices = dict(zip(final_market['symbol'], final_market['close']))
        
        for symbol in list(self.position_manager.positions.keys()):
            if symbol in final_prices:
                trade = self.position_manager.close_position(
                    symbol, final_prices[symbol], final_timestamp, "backtest_end"
                )
                if trade:
                    self.trades.append(trade)
        
        # Prepare results
        results = self._prepare_results()
        
        print(f"Backtest completed. Total trades: {len(self.trades)}")
        
        return results
    
    def _process_signals(self, predictions: pd.DataFrame, 
                        price_data: Dict[str, float], timestamp: datetime) -> None:
        """Process trading signals from model predictions."""
        
        for _, row in predictions.iterrows():
            symbol = row['symbol']
            prediction_prob = row.get('prediction_proba', 0.5)
            
            # Skip if below threshold
            if prediction_prob < self.config.prediction_threshold:
                continue
            
            # Skip if already have position
            if symbol in self.position_manager.positions:
                continue
            
            # Skip if no price data
            if symbol not in price_data:
                continue
            
            # Calculate position size
            current_price = price_data[symbol]
            
            # Estimate volatility (simplified)
            volatility = 0.02  # Assume 2% daily volatility
            
            position_size = self.position_manager.calculate_position_size(
                symbol, current_price, prediction_prob, volatility
            )
            
            # Open position if size is significant
            if position_size > self.config.initial_capital * 0.001:  # Minimum 0.1% of capital
                self.position_manager.open_position(
                    symbol, position_size, current_price, timestamp, prediction_prob
                )
    
    def _record_portfolio_state(self, timestamp: datetime, 
                              price_data: Dict[str, float]) -> None:
        """Record current portfolio state."""
        
        # Update position values
        self.position_manager.update_positions(price_data, timestamp)
        
        portfolio_state = {
            'timestamp': timestamp,
            'cash': self.position_manager.cash,
            'total_value': self.position_manager.total_value,
            'num_positions': len(self.position_manager.positions),
            'leverage': (self.position_manager.total_value - self.position_manager.cash) / self.position_manager.total_value
        }
        
        self.portfolio_history.append(portfolio_state)
    
    def _prepare_results(self) -> Dict[str, Any]:
        """Prepare comprehensive backtest results."""
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Calculate returns
        if len(portfolio_df) > 1:
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
            returns_series = portfolio_df['returns'].dropna()
        else:
            returns_series = pd.Series([0])
        
        # Calculate risk metrics
        risk_metrics = RiskAnalyzer.calculate_comprehensive_metrics(
            trades_df, returns_series
        )
        
        return {
            'trades': trades_df,
            'portfolio_history': portfolio_df,
            'risk_metrics': risk_metrics,
            'config': self.config,
            'summary': {
                'total_trades': len(trades_df),
                'final_portfolio_value': portfolio_df['total_value'].iloc[-1] if len(portfolio_df) > 0 else self.config.initial_capital,
                'total_return_pct': (portfolio_df['total_value'].iloc[-1] / self.config.initial_capital - 1) * 100 if len(portfolio_df) > 0 else 0,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown_pct': risk_metrics.max_drawdown * 100,
                'win_rate_pct': risk_metrics.win_rate * 100
            }
        }


class BacktestVisualizer:
    """Visualization tools for backtest results."""
    
    @staticmethod
    def plot_portfolio_performance(results: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> None:
        """Plot comprehensive portfolio performance."""
        
        portfolio_df = results['portfolio_history']
        risk_metrics = results['risk_metrics']
        
        if portfolio_df.empty:
            print("No portfolio history to plot.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Portfolio value over time
        ax1.plot(portfolio_df['timestamp'], portfolio_df['total_value'], 
                linewidth=2, color='blue', label='Portfolio Value')
        ax1.axhline(y=results['config'].initial_capital, color='red', 
                   linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        portfolio_df['cumulative_returns'] = portfolio_df['total_value'] / results['config'].initial_capital
        portfolio_df['rolling_max'] = portfolio_df['cumulative_returns'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['cumulative_returns'] - portfolio_df['rolling_max']) / portfolio_df['rolling_max']
        
        ax2.fill_between(portfolio_df['timestamp'], portfolio_df['drawdown'] * 100, 0,
                        color='red', alpha=0.3)
        ax2.plot(portfolio_df['timestamp'], portfolio_df['drawdown'] * 100, 
                color='red', linewidth=1)
        ax2.set_title(f'Drawdown (Max: {risk_metrics.max_drawdown*100:.2f}%)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Daily returns distribution
        if 'returns' in portfolio_df.columns:
            returns = portfolio_df['returns'].dropna() * 100
            ax3.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {returns.mean():.3f}%')
            ax3.set_title('Returns Distribution')
            ax3.set_xlabel('Returns (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Number of positions over time
        ax4.plot(portfolio_df['timestamp'], portfolio_df['num_positions'], 
                color='green', linewidth=1)
        ax4.set_title('Number of Active Positions')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Number of Positions')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio performance plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_trade_analysis(results: Dict[str, Any], 
                          save_path: Optional[str] = None) -> None:
        """Plot trade analysis."""
        
        trades_df = results['trades']
        
        if trades_df.empty:
            print("No trades to analyze.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # P&L distribution
        ax1.hist(trades_df['pnl_pct'] * 100, bins=30, alpha=0.7, 
                color='lightblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L (%)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative P&L
        trades_df_sorted = trades_df.sort_values('exit_time')
        cumulative_pnl = trades_df_sorted['pnl'].cumsum()
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue')
        ax2.set_title('Cumulative P&L by Trade')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # Holding period analysis
        trades_df['holding_hours'] = trades_df['holding_period'].dt.total_seconds() / 3600
        ax3.hist(trades_df['holding_hours'], bins=30, alpha=0.7, 
                color='lightgreen', edgecolor='black')
        ax3.set_title('Holding Period Distribution')
        ax3.set_xlabel('Holding Period (Hours)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # P&L vs Confidence
        ax4.scatter(trades_df['confidence'], trades_df['pnl_pct'] * 100, 
                   alpha=0.6, color='purple')
        ax4.set_title('P&L vs Prediction Confidence')
        ax4.set_xlabel('Prediction Confidence')
        ax4.set_ylabel('P&L (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation
        correlation = trades_df['confidence'].corr(trades_df['pnl_pct'])
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trade analysis plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def generate_backtest_report(results: Dict[str, Any], 
                               save_path: Optional[str] = None) -> str:
        """Generate comprehensive backtest report."""
        
        risk_metrics = results['risk_metrics']
        summary = results['summary']
        config = results['config']
        
        report = f"""
# Backtest Report

## Strategy Configuration
- **Initial Capital**: ${config.initial_capital:,.2f}
- **Commission**: {config.commission*100:.3f}%
- **Position Sizing**: {config.position_sizing_method}
- **Max Position Size**: {config.max_position_size*100:.1f}%
- **Prediction Threshold**: {config.prediction_threshold:.2f}

## Performance Summary
- **Total Trades**: {summary['total_trades']:,}
- **Final Portfolio Value**: ${summary['final_portfolio_value']:,.2f}
- **Total Return**: {summary['total_return_pct']:.2f}%
- **Win Rate**: {summary['win_rate_pct']:.1f}%

## Risk Metrics

### Return Metrics
- **Annualized Return**: {risk_metrics.annualized_return*100:.2f}%
- **Cumulative Return**: {risk_metrics.cumulative_return*100:.2f}%
- **Volatility (Annual)**: {risk_metrics.annualized_volatility*100:.2f}%

### Risk-Adjusted Returns
- **Sharpe Ratio**: {risk_metrics.sharpe_ratio:.3f}
- **Sortino Ratio**: {risk_metrics.sortino_ratio:.3f}
- **Calmar Ratio**: {risk_metrics.calmar_ratio:.3f}

### Drawdown Analysis
- **Maximum Drawdown**: {risk_metrics.max_drawdown*100:.2f}%
- **Max Drawdown Duration**: {risk_metrics.max_drawdown_duration} periods
- **Current Drawdown**: {risk_metrics.current_drawdown*100:.2f}%

### Trading Performance
- **Profit Factor**: {risk_metrics.profit_factor:.2f}
- **Average Win**: {risk_metrics.avg_win:.2f}
- **Average Loss**: {risk_metrics.avg_loss:.2f}
- **Win/Loss Ratio**: {risk_metrics.win_loss_ratio:.2f}

### Risk Measures
- **Value at Risk (95%)**: {risk_metrics.var_95*100:.2f}%
- **Conditional VaR (95%)**: {risk_metrics.cvar_95*100:.2f}%
- **Skewness**: {risk_metrics.skewness:.3f}
- **Kurtosis**: {risk_metrics.kurtosis:.3f}

## Statistical Analysis
- **Jarque-Bera Test**: {risk_metrics.jarque_bera_stat:.3f} (p-value: {risk_metrics.jarque_bera_pvalue:.3f})
- **Returns Distribution**: {"Normal" if risk_metrics.jarque_bera_pvalue > 0.05 else "Non-normal"}

## Risk Assessment
"""
        
        # Risk assessment
        if risk_metrics.sharpe_ratio >= 1.0:
            report += "- **Excellent**: Sharpe ratio ≥ 1.0 indicates strong risk-adjusted returns\\n"
        elif risk_metrics.sharpe_ratio >= 0.5:
            report += "- **Good**: Sharpe ratio ≥ 0.5 indicates acceptable risk-adjusted returns\\n"
        else:
            report += "- **Poor**: Sharpe ratio < 0.5 indicates poor risk-adjusted returns\\n"
        
        if abs(risk_metrics.max_drawdown) <= 0.1:
            report += "- **Low Risk**: Maximum drawdown ≤ 10%\\n"
        elif abs(risk_metrics.max_drawdown) <= 0.2:
            report += "- **Moderate Risk**: Maximum drawdown ≤ 20%\\n"
        else:
            report += "- **High Risk**: Maximum drawdown > 20%\\n"
        
        if risk_metrics.win_rate >= 0.6:
            report += "- **High Accuracy**: Win rate ≥ 60%\\n"
        elif risk_metrics.win_rate >= 0.4:
            report += "- **Moderate Accuracy**: Win rate ≥ 40%\\n"
        else:
            report += "- **Low Accuracy**: Win rate < 40%\\n"
        
        report += f"""
---
*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Backtest report saved to: {save_path}")
        
        return report


def demonstrate_backtesting():
    """Demonstrate backtesting capabilities."""
    
    # Generate synthetic market data
    np.random.seed(42)
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='1H')
    
    symbols = ['BTC', 'ETH', 'BNB']
    
    market_data = []
    predictions = []
    
    for symbol in symbols:
        # Generate price data with random walk
        n_periods = len(date_range)
        returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift
        prices = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
        
        symbol_data = pd.DataFrame({
            'symbol': symbol,
            'open_time': date_range,
            'close': prices
        })
        
        # Generate predictions (with some skill)
        future_returns = np.roll(returns, -6)  # 6-hour forward returns
        prediction_proba = 0.5 + 0.3 * np.tanh(future_returns * 10)  # Sigmoid transformation
        prediction_proba += np.random.normal(0, 0.1, n_periods)  # Add noise
        prediction_proba = np.clip(prediction_proba, 0, 1)
        
        symbol_predictions = pd.DataFrame({
            'symbol': symbol,
            'open_time': date_range,
            'prediction_proba': prediction_proba
        })
        
        market_data.append(symbol_data)
        predictions.append(symbol_predictions)
    
    market_df = pd.concat(market_data, ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    
    print(f"Generated market data: {market_df.shape}")
    print(f"Generated predictions: {predictions_df.shape}")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        max_position_size=0.2,
        prediction_threshold=0.6,
        stop_loss=0.05,
        take_profit=0.10
    )
    
    # Run backtest
    backtester = Backtester(config)
    results = backtester.run_backtest(market_df, predictions_df)
    
    # Generate visualizations
    BacktestVisualizer.plot_portfolio_performance(results)
    BacktestVisualizer.plot_trade_analysis(results)
    
    # Generate report
    report = BacktestVisualizer.generate_backtest_report(results)
    print(report)
    
    print("Backtesting demonstration completed!")


if __name__ == "__main__":
    demonstrate_backtesting()