"""
Comprehensive Strategy Demonstration Script

This script demonstrates all implemented trading strategies:
- Momentum Strategy
- Mean Reversion Strategy  
- Breakout Strategy
- Microstructure Strategy
- Pairs Trading Strategy

It runs backtests for each strategy and provides performance comparisons.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List

from strategies import (
    MomentumStrategy, MeanReversionStrategy, BreakoutStrategy,
    MicrostructureStrategy, PairsTradingStrategy
)
from backtests.engine import BacktestEngine
from backtests.portfolio import Portfolio
from backtests.performance import PerformanceAnalyzer
from utils.config import ConfigManager
from utils.logging import setup_logging
from utils.risk import RiskManager


def generate_synthetic_data(symbols: List[str], periods: int = 500) -> Dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for multiple symbols"""
    np.random.seed(42)
    
    data = {}
    base_price = 100.0
    
    for symbol in symbols:
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0001, 0.02, periods)  # Daily returns
        
        # Add some trend
        trend = np.linspace(0, 0.1, periods)
        returns += trend
        
        # Add some mean reversion
        for i in range(1, periods):
            if abs(returns[i]) > 0.03:
                returns[i] *= -0.5
        
        # Generate prices
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, periods),
            'symbol': symbol
        }, index=dates)
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        data[symbol] = df
    
    return data


def load_strategy_configs() -> Dict:
    """Load configuration for all strategies"""
    configs = {}
    
    strategy_files = [
        'configs/strategies/momentum.yaml',
        'configs/strategies/mean_reversion.yaml',
        'configs/strategies/breakout.yaml',
        'configs/strategies/microstructure.yaml',
        'configs/strategies/pairs_trading.yaml'
    ]
    
    for config_file in strategy_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                strategy_name = config_file.split('/')[-1].replace('.yaml', '')
                configs[strategy_name] = config
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found")
    
    return configs


def run_strategy_backtest(strategy_class, strategy_config: Dict, 
                         data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
    """Run backtest for a single strategy"""
    try:
        # Initialize risk manager
        risk_manager = RiskManager({
            'max_position_size': 0.2,
            'max_drawdown': 0.15,
            'stop_loss_atr_multiplier': 2.0
        })
        
        # Initialize strategy
        strategy = strategy_class(strategy_config, risk_manager)
        
        # Initialize portfolio
        portfolio = Portfolio(initial_capital)
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Run backtest
        results = engine.run_backtest(strategy, data, portfolio)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        performance = analyzer.calculate_performance(results['trades'], results['portfolio_history'])
        
        return {
            'strategy_name': strategy.get_strategy_name(),
            'performance': performance,
            'trades': results['trades'],
            'portfolio_history': results['portfolio_history'],
            'success': True
        }
        
    except Exception as e:
        logging.error(f"Error running backtest for {strategy_class.__name__}: {str(e)}")
        return {
            'strategy_name': strategy_class.__name__,
            'performance': {},
            'trades': [],
            'portfolio_history': [],
            'success': False,
            'error': str(e)
        }


def compare_strategies(results: List[Dict]) -> pd.DataFrame:
    """Compare performance across all strategies"""
    comparison_data = []
    
    for result in results:
        if result['success'] and result['performance']:
            perf = result['performance']
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Total Return (%)': perf.get('total_return', 0) * 100,
                'Sharpe Ratio': perf.get('sharpe_ratio', 0),
                'Max Drawdown (%)': perf.get('max_drawdown', 0) * 100,
                'Win Rate (%)': perf.get('win_rate', 0) * 100,
                'Profit Factor': perf.get('profit_factor', 0),
                'Total Trades': perf.get('total_trades', 0),
                'Avg Trade Duration': perf.get('avg_trade_duration', 0)
            })
    
    return pd.DataFrame(comparison_data)


def main():
    """Main execution function"""
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting comprehensive strategy demonstration...")
    
    # Load configurations
    configs = load_strategy_configs()
    logger.info(f"Loaded configurations for {len(configs)} strategies")
    
    # Generate synthetic data
    symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
    data = generate_synthetic_data(symbols, periods=500)
    logger.info(f"Generated synthetic data for {len(symbols)} symbols")
    
    # Define strategies to test
    strategies = [
        (MomentumStrategy, 'momentum'),
        (MeanReversionStrategy, 'mean_reversion'),
        (BreakoutStrategy, 'breakout'),
        (MicrostructureStrategy, 'microstructure'),
        (PairsTradingStrategy, 'pairs_trading')
    ]
    
    # Run backtests for each strategy
    results = []
    initial_capital = 100000
    
    for strategy_class, config_key in strategies:
        if config_key in configs:
            logger.info(f"Running backtest for {strategy_class.__name__}...")
            
            # Use first symbol's data for single-symbol strategies
            if strategy_class == PairsTradingStrategy:
                # Pairs trading needs multiple symbols
                strategy_data = data
            else:
                strategy_data = data[symbols[0]]
            
            result = run_strategy_backtest(
                strategy_class, 
                configs[config_key], 
                strategy_data, 
                initial_capital
            )
            results.append(result)
            
            if result['success']:
                logger.info(f"✓ {result['strategy_name']} completed successfully")
            else:
                logger.warning(f"✗ {result['strategy_name']} failed: {result.get('error', 'Unknown error')}")
        else:
            logger.warning(f"Configuration not found for {strategy_class.__name__}")
    
    # Compare strategies
    logger.info("Comparing strategy performance...")
    comparison_df = compare_strategies(results)
    
    if not comparison_df.empty:
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Find best performing strategy
        best_strategy = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        print(f"\n🏆 Best Strategy by Sharpe Ratio: {best_strategy['Strategy']}")
        print(f"   Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
        print(f"   Total Return: {best_strategy['Total Return (%)']:.2f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_df.to_csv(f"strategy_comparison_{timestamp}.csv", index=False)
        logger.info(f"Results saved to strategy_comparison_{timestamp}.csv")
        
    else:
        logger.warning("No successful backtests to compare")
    
    # Generate summary report
    successful_strategies = sum(1 for r in results if r['success'])
    total_strategies = len(strategies)
    
    print(f"\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Strategies: {total_strategies}")
    print(f"Successful Backtests: {successful_strategies}")
    print(f"Failed Backtests: {total_strategies - successful_strategies}")
    print(f"Success Rate: {successful_strategies/total_strategies*100:.1f}%")
    
    if successful_strategies > 0:
        print(f"\nStrategies tested successfully:")
        for result in results:
            if result['success']:
                print(f"  ✓ {result['strategy_name']}")
    
    if total_strategies - successful_strategies > 0:
        print(f"\nStrategies with issues:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {result['strategy_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info("Strategy demonstration completed!")


if __name__ == "__main__":
    main()
