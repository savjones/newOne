#!/usr/bin/env python3
"""
Example script to run a backtest with the momentum strategy.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.momentum import MomentumStrategy
from backtests.engine import BacktestEngine
from data.ingestion import DataIngestion
from data.storage import DataStorage
from data.cleaner import DataCleaner
from utils.config import ConfigManager
from utils.logging import setup_logging


def generate_sample_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = '1H'
) -> pd.DataFrame:
    """Generate sample OHLCV data for testing.
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        timeframe: Timeframe
        
    Returns:
        DataFrame with sample OHLCV data
    """
    # Generate time range
    if timeframe == '1H':
        freq = 'H'
    elif timeframe == '4H':
        freq = '4H'
    elif timeframe == '1D':
        freq = 'D'
    else:
        freq = 'H'
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate sample price data with some trend and volatility
    np.random.seed(42)  # For reproducible results
    
    # Start price
    start_price = 100.0
    
    # Generate returns with trend and volatility
    returns = np.random.normal(0.0001, 0.02, len(timestamps))  # Small positive trend
    
    # Add some momentum
    for i in range(1, len(returns)):
        returns[i] += returns[i-1] * 0.1  # Momentum effect
    
    # Convert to prices
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Generate OHLC from price
        volatility = np.random.uniform(0.005, 0.02)
        
        open_price = price * (1 + np.random.uniform(-volatility, volatility))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, volatility))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, volatility))
        close_price = price
        
        # Generate volume
        base_volume = 1000000
        volume = base_volume * (1 + np.random.uniform(-0.5, 1.0))
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'symbol': symbol
        })
    
    return pd.DataFrame(data)


def main():
    """Main function to run the backtest."""
    print("🚀 Starting Crypto Trading Bot Backtest")
    print("=" * 50)
    
    # Setup logging
    setup_logging(
        name="backtest_example",
        level="INFO",
        log_dir="logs"
    )
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Strategy configuration
    strategy_config = {
        'strategy': {
            'name': 'momentum',
            'description': 'Momentum strategy example',
            'version': '1.0.0',
            'enabled': True
        },
        'signals': {
            'short_ma': 20,
            'long_ma': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'volatility_window': 30,
            'volume_ma_period': 20,
            'volume_threshold': 1.5
        },
        'rules': {
            'entry': {
                'ma_crossover': True,
                'volume_confirmation': True,
                'volatility_filter': True
            },
            'exit': {
                'stop_loss': True,
                'stop_loss_type': 'atr',
                'stop_loss_multiplier': 2.0,
                'take_profit': True,
                'risk_reward_ratio': 3.0
            }
        },
        'risk': {
            'max_position_size': 0.1,
            'max_drawdown': 0.2,
            'stop_loss_default': 0.05,
            'take_profit_default': 0.15
        }
    }
    
    # Initialize strategy
    print("📊 Initializing Momentum Strategy...")
    strategy = MomentumStrategy(
        strategy_name="momentum_example",
        config=strategy_config
    )
    
    # Generate sample data
    print("📈 Generating sample market data...")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    market_data = {}
    
    for symbol in symbols:
        data = generate_sample_data(symbol, start_date, end_date, '1H')
        market_data[symbol] = data
        print(f"  Generated {len(data)} data points for {symbol}")
    
    # Initialize backtesting engine
    print("⚙️  Initializing Backtesting Engine...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Add strategy
    engine.add_strategy(strategy)
    
    # Add market data
    for symbol, data in market_data.items():
        engine.add_market_data(symbol, data)
    
    # Run backtest
    print("🏃 Running backtest...")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Initial Capital: $100,000")
    print(f"  Commission: 0.1%")
    print(f"  Slippage: 0.05%")
    
    start_time = datetime.now()
    
    try:
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            timeframe='1H'
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ Backtest completed in {duration.total_seconds():.2f} seconds")
        
        # Display results
        print("\n📊 Backtest Results")
        print("=" * 50)
        
        # Portfolio performance
        portfolio_perf = results.get('portfolio_performance', {})
        if portfolio_perf:
            print("\nPortfolio Performance:")
            print(f"  Total Return: {portfolio_perf.get('total_return', 0):.2%}")
            print(f"  Annualized Return: {portfolio_perf.get('annualized_return', 0):.2%}")
            print(f"  Volatility: {portfolio_perf.get('volatility', 0):.2%}")
            print(f"  Sharpe Ratio: {portfolio_perf.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {portfolio_perf.get('max_drawdown', 0):.2%}")
            print(f"  Hit Rate: {portfolio_perf.get('hit_rate', 0):.2%}")
        
        # Strategy performance
        strategy_perf = results.get('strategy_performance', {})
        if strategy_perf:
            print("\nStrategy Performance:")
            for strategy_name, perf in strategy_perf.items():
                print(f"  {strategy_name}:")
                print(f"    Total Trades: {perf.get('total_trades', 0)}")
                print(f"    Win Rate: {perf.get('win_rate', 0):.2%}")
                print(f"    Total Signals: {perf.get('total_signals', 0)}")
        
        # Overall results
        backtest_info = results.get('backtest_info', {})
        if backtest_info:
            print("\nOverall Results:")
            print(f"  Initial Capital: ${backtest_info.get('initial_capital', 0):,.2f}")
            print(f"  Final Capital: ${backtest_info.get('final_capital', 0):,.2f}")
            print(f"  Total Return: {backtest_info.get('total_return', 0):.2%}")
            print(f"  Total Events: {results.get('events', 0)}")
            print(f"  Total Trades: {results.get('trades', 0)}")
        
        # Strategy summary
        print("\n📋 Strategy Summary:")
        strategy_summary = engine.get_strategy_summary()
        for strategy_name, summary in strategy_summary.items():
            print(f"  {strategy_name}:")
            print(f"    Positions: {len(summary.get('positions', {}))}")
            print(f"    Total Signals: {summary.get('total_signals', 0)}")
            print(f"    Total Trades: {summary.get('total_trades', 0)}")
        
        print("\n🎉 Backtest completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
