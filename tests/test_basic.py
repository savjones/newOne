"""
Basic tests for the crypto trading bot.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import ConfigManager
from utils.risk import RiskManager, PositionSide
from strategies.momentum import MomentumStrategy
from backtests.portfolio import Portfolio


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the trading bot."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.risk_manager = RiskManager()
        
        # Sample strategy config
        self.strategy_config = {
            'strategy': {
                'name': 'test_momentum',
                'description': 'Test momentum strategy',
                'version': '1.0.0',
                'enabled': True
            },
            'signals': {
                'short_ma': 20,
                'long_ma': 50,
                'rsi_period': 14,
                'atr_period': 14,
                'volatility_window': 30
            },
            'rules': {
                'entry': {
                    'ma_crossover': True,
                    'volume_confirmation': True
                },
                'exit': {
                    'stop_loss': True,
                    'take_profit': True
                }
            },
            'risk': {
                'max_position_size': 0.1,
                'max_drawdown': 0.2
            }
        }
        
        # Sample market data
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H')
        
        data = []
        price = 100.0
        
        for i, date in enumerate(dates):
            # Simple price movement
            price += np.random.normal(0, 0.5)
            
            # Generate OHLC
            open_price = price
            high_price = price + np.random.uniform(0, 1)
            low_price = price - np.random.uniform(0, 1)
            close_price = price + np.random.normal(0, 0.2)
            volume = np.random.uniform(1000000, 2000000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'symbol': 'BTC/USDT'
            })
        
        return pd.DataFrame(data)
    
    def test_config_manager(self):
        """Test configuration manager."""
        self.assertIsInstance(self.config_manager, ConfigManager)
        
        # Test getting risk config
        risk_config = self.config_manager.get_risk_config()
        self.assertIsNotNone(risk_config)
        self.assertEqual(risk_config.max_position_size, 0.1)
    
    def test_risk_manager(self):
        """Test risk manager."""
        self.assertIsInstance(self.risk_manager, RiskManager)
        
        # Test position size calculation
        position_size = self.risk_manager.calculate_position_size(
            portfolio_value=100000,
            symbol='BTC/USDT',
            price=50000,
            volatility=0.02,
            confidence=0.8
        )
        
        self.assertGreater(position_size, 0)
        
        # Test position limits
        is_valid, reason = self.risk_manager.check_position_limits(
            symbol='BTC/USDT',
            quantity=1.0,
            price=50000,
            portfolio_value=100000
        )
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(reason, str)
    
    def test_momentum_strategy(self):
        """Test momentum strategy."""
        strategy = MomentumStrategy(
            strategy_name="test_momentum",
            config=self.strategy_config
        )
        
        self.assertEqual(strategy.strategy_name, "test_momentum")
        self.assertIsNotNone(strategy.config)
        
        # Test data processing
        processed_data = strategy.process_data(self.sample_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertGreater(len(processed_data), 0)
        
        # Test signal generation
        signals = strategy.generate_signals(processed_data)
        self.assertIsInstance(signals, list)
    
    def test_portfolio(self):
        """Test portfolio management."""
        portfolio = Portfolio(initial_capital=100000)
        
        self.assertEqual(portfolio.initial_capital, 100000)
        self.assertEqual(portfolio.cash, 100000)
        
        # Test trade execution
        success = portfolio.execute_trade(
            symbol='BTC/USDT',
            side='BUY',
            quantity=1.0,
            price=50000,
            commission=50
        )
        
        self.assertTrue(success)
        self.assertLess(portfolio.cash, 100000)
        self.assertIn('BTC/USDT', portfolio.positions)
        
        # Test position update
        portfolio.update_price('BTC/USDT', 51000)
        position = portfolio.get_position('BTC/USDT')
        self.assertEqual(position.current_price, 51000)
        
        # Test metrics calculation
        metrics = portfolio.calculate_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_value', metrics)
    
    def test_data_validation(self):
        """Test data validation."""
        # Test valid data
        valid_data = self.sample_data.copy()
        self.assertTrue(len(valid_data) > 0)
        
        # Test required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, valid_data.columns)
        
        # Test data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(valid_data['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(valid_data['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(valid_data['close']))
    
    def test_strategy_config_validation(self):
        """Test strategy configuration validation."""
        # Test valid config
        strategy = MomentumStrategy(
            strategy_name="test_validation",
            config=self.strategy_config
        )
        
        self.assertIsNotNone(strategy)
        
        # Test invalid config (missing required keys)
        invalid_config = {'strategy': {'name': 'test'}}
        
        with self.assertRaises(ValueError):
            MomentumStrategy(
                strategy_name="test_invalid",
                config=invalid_config
            )
    
    def test_risk_calculations(self):
        """Test risk calculations."""
        # Test stop loss calculation
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=50000,
            side=PositionSide.LONG,
            atr=1000,
            multiplier=2.0
        )
        
        self.assertEqual(stop_loss, 48000)  # 50000 - (1000 * 2)
        
        # Test take profit calculation
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price=50000,
            side=PositionSide.LONG,
            risk_reward_ratio=3.0,
            stop_loss_pct=0.05
        )
        
        expected_take_profit = 50000 + (50000 * 0.05 * 3.0)
        self.assertAlmostEqual(take_profit, expected_take_profit, places=2)


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
