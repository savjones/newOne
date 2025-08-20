"""
Comprehensive Tests for All Trading Strategies

This test file validates the functionality of all implemented strategies:
- Momentum Strategy
- Mean Reversion Strategy
- Breakout Strategy
- Microstructure Strategy
- Pairs Trading Strategy
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from strategies import (
    MomentumStrategy, MeanReversionStrategy, BreakoutStrategy,
    MicrostructureStrategy, PairsTradingStrategy
)
from utils.risk import RiskManager
from utils.data_utils import calculate_sma, calculate_ema, calculate_rsi


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for Momentum Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategy': {
                'ma_short_period': 10,
                'ma_long_period': 20,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_threshold': 1.5,
                'volatility_threshold': 0.02
            }
        }
        self.risk_manager = Mock(spec=RiskManager)
        self.strategy = MomentumStrategy(self.config, self.risk_manager)
        
        # Generate test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(80, 90, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.test_data['high'] = self.test_data[['open', 'high', 'close']].max(axis=1)
        self.test_data['low'] = self.test_data[['open', 'low', 'close']].min(axis=1)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.get_strategy_name(), "Momentum (MA + RSI + Volume)")
        self.assertIsNotNone(self.strategy.strategy_config)
    
    def test_calculate_indicators(self):
        """Test indicator calculation"""
        result = self.strategy.calculate_indicators(self.test_data)
        
        # Check that indicators are calculated
        self.assertIn('ma_short', result.columns)
        self.assertIn('ma_long', result.columns)
        self.assertIn('rsi', result.columns)
        self.assertIn('volume_ma', result.columns)
        self.assertIn('volatility', result.columns)
        
        # Check that indicators have reasonable values
        self.assertTrue(all(result['ma_short'].notna().tail(50)))
        self.assertTrue(all(result['ma_long'].notna().tail(50)))
        self.assertTrue(all(result['rsi'].notna().tail(50)))
    
    def test_signal_generation(self):
        """Test signal generation"""
        # Calculate indicators first
        data_with_indicators = self.strategy.calculate_indicators(self.test_data)
        
        # Generate signals
        signals = self.strategy.generate_signals(data_with_indicators)
        
        # Check that signals are generated (may be empty if conditions not met)
        self.assertIsInstance(signals, list)
        
        # If signals are generated, check their structure
        if signals:
            signal = signals[0]
            self.assertIsNotNone(signal.timestamp)
            self.assertIsNotNone(signal.signal_type)
            self.assertIsNotNone(signal.price)
            self.assertIsNotNone(signal.confidence)
    
    def test_position_sizing(self):
        """Test position size calculation"""
        # Create a mock signal
        mock_signal = Mock()
        mock_signal.confidence = 0.8
        
        # Test position sizing
        position_size = self.strategy.calculate_position_size(
            mock_signal, 100.0, 100000
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.strategy.strategy_config.max_position_size)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test cases for Mean Reversion Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategy': {
                'vwap_period': 20,
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'entry_threshold': 2.5,
                'exit_threshold': 0.5
            }
        }
        self.risk_manager = Mock(spec=RiskManager)
        self.strategy = MeanReversionStrategy(self.config, self.risk_manager)
        
        # Generate test data with mean reversion characteristics
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 100.0
        
        # Create mean-reverting price series
        prices = [base_price]
        for i in range(1, 100):
            # Add mean reversion
            deviation = (prices[-1] - base_price) / base_price
            if abs(deviation) > 0.05:
                # Strong mean reversion
                prices.append(prices[-1] * (1 - np.sign(deviation) * 0.02))
            else:
                # Random walk
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.test_data['high'] = self.test_data[['open', 'high', 'close']].max(axis=1)
        self.test_data['low'] = self.test_data[['open', 'low', 'close']].min(axis=1)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.get_strategy_name(), "Mean Reversion (VWAP + Bollinger Bands)")
    
    def test_calculate_indicators(self):
        """Test indicator calculation"""
        result = self.strategy.calculate_indicators(self.test_data)
        
        # Check that indicators are calculated
        self.assertIn('vwap', result.columns)
        self.assertIn('bb_upper', result.columns)
        self.assertIn('bb_lower', result.columns)
        self.assertIn('bb_middle', result.columns)
        self.assertIn('rsi', result.columns)
        self.assertIn('vwap_deviation', result.columns)
        
        # Check that indicators have reasonable values
        self.assertTrue(all(result['vwap'].notna().tail(50)))
        self.assertTrue(all(result['bb_upper'].notna().tail(50)))
        self.assertTrue(all(result['bb_lower'].notna().tail(50)))
    
    def test_signal_generation(self):
        """Test signal generation"""
        # Calculate indicators first
        data_with_indicators = self.strategy.calculate_indicators(self.test_data)
        
        # Generate signals
        signals = self.strategy.generate_signals(data_with_indicators)
        
        # Check that signals are generated
        self.assertIsInstance(signals, list)
        
        # If signals are generated, check their structure
        if signals:
            signal = signals[0]
            self.assertIsNotNone(signal.timestamp)
            self.assertIsNotNone(signal.signal_type)
            self.assertIsNotNone(signal.price)
            self.assertIsNotNone(signal.confidence)


class TestBreakoutStrategy(unittest.TestCase):
    """Test cases for Breakout Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategy': {
                'donchian_period': 20,
                'keltner_period': 20,
                'keltner_multiplier': 2.0,
                'volume_threshold': 1.5,
                'volatility_threshold': 1.2,
                'breakout_confirmation_bars': 2
            }
        }
        self.risk_manager = Mock(spec=RiskManager)
        self.strategy = BreakoutStrategy(self.config, self.risk_manager)
        
        # Generate test data with breakout characteristics
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 100.0
        
        # Create price series with consolidation and breakout
        prices = [base_price]
        for i in range(1, 100):
            if i < 50:
                # Consolidation phase
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))
            elif i == 50:
                # Breakout
                prices.append(prices[-1] * 1.05)
            else:
                # Trending phase
                prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.01)))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        self.test_data['high'] = self.test_data[['open', 'high', 'close']].max(axis=1)
        self.test_data['low'] = self.test_data[['open', 'low', 'close']].min(axis=1)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.get_strategy_name(), "Breakout (Donchian + Keltner + Volume/Volatility)")
    
    def test_calculate_indicators(self):
        """Test indicator calculation"""
        result = self.strategy.calculate_indicators(self.test_data)
        
        # Check that indicators are calculated
        self.assertIn('donchian_upper', result.columns)
        self.assertIn('donchian_lower', result.columns)
        self.assertIn('keltner_upper', result.columns)
        self.assertIn('keltner_lower', result.columns)
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('volatility_ratio', result.columns)
        self.assertIn('breakout_strength', result.columns)
        
        # Check that indicators have reasonable values
        self.assertTrue(all(result['donchian_upper'].notna().tail(50)))
        self.assertTrue(all(result['donchian_lower'].notna().tail(50)))
        self.assertTrue(all(result['breakout_strength'].notna().tail(50)))


class TestMicrostructureStrategy(unittest.TestCase):
    """Test cases for Microstructure Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategy': {
                'order_flow_window': 10,
                'imbalance_threshold': 0.6,
                'flow_strength_threshold': 0.7,
                'vwap_period': 20,
                'vwap_drift_threshold': 0.001
            }
        }
        self.risk_manager = Mock(spec=RiskManager)
        self.strategy = MicrostructureStrategy(self.config, self.risk_manager)
        
        # Generate test data with microstructure characteristics
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 100.0
        
        # Create price series with order flow characteristics
        prices = [base_price]
        volumes = [10000]
        
        for i in range(1, 100):
            # Simulate order flow imbalance
            if i % 20 == 0:
                # Large order flow
                price_change = np.random.normal(0.01, 0.005)
                volume = volumes[-1] * 2.0
            else:
                # Normal order flow
                price_change = np.random.normal(0, 0.002)
                volume = volumes[-1] * np.random.uniform(0.8, 1.2)
            
            prices.append(prices[-1] * (1 + price_change))
            volumes.append(max(1000, volume))
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Ensure OHLC consistency
        self.test_data['high'] = self.test_data[['open', 'high', 'close']].max(axis=1)
        self.test_data['low'] = self.test_data[['open', 'low', 'close']].min(axis=1)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.get_strategy_name(), "Microstructure (Order Flow + Book Imbalance + VWAP Drift)")
    
    def test_calculate_indicators(self):
        """Test indicator calculation"""
        result = self.strategy.calculate_indicators(self.test_data)
        
        # Check that indicators are calculated
        self.assertIn('order_flow_imbalance', result.columns)
        self.assertIn('flow_strength', result.columns)
        self.assertIn('vwap_drift', result.columns)
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('microstructure_signal', result.columns)
        
        # Check that indicators have reasonable values
        self.assertTrue(all(result['order_flow_imbalance'].notna().tail(50)))
        self.assertTrue(all(result['flow_strength'].notna().tail(50)))
        self.assertTrue(all(result['microstructure_signal'].notna().tail(50)))


class TestPairsTradingStrategy(unittest.TestCase):
    """Test cases for Pairs Trading Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategy': {
                'cointegration_lookback': 252,
                'cointegration_pvalue': 0.05,
                'spread_lookback': 60,
                'z_score_entry': 2.0,
                'z_score_exit': 0.5
            }
        }
        self.risk_manager = Mock(spec=RiskManager)
        self.strategy = PairsTradingStrategy(self.config, self.risk_manager)
        
        # Generate test data for multiple symbols
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        
        # Create cointegrated price series
        np.random.seed(42)
        base_price1 = 100.0
        base_price2 = 50.0
        
        prices1 = [base_price1]
        prices2 = [base_price2]
        
        for i in range(1, 300):
            # Create cointegrated relationship
            if i < 100:
                # Normal relationship
                prices1.append(prices1[-1] * (1 + np.random.normal(0, 0.01)))
                prices2.append(prices2[-1] * (1 + np.random.normal(0, 0.01)))
            elif i < 200:
                # Spread widens
                prices1.append(prices1[-1] * (1 + np.random.normal(0.002, 0.01)))
                prices2.append(prices2[-1] * (1 + np.random.normal(-0.001, 0.01)))
            else:
                # Spread narrows (mean reversion)
                prices1.append(prices1[-1] * (1 + np.random.normal(-0.001, 0.01)))
                prices2.append(prices2[-1] * (1 + np.random.normal(0.002, 0.01)))
        
        self.test_data = {
            'BTC': pd.DataFrame({
                'open': prices1,
                'high': [p * 1.01 for p in prices1],
                'low': [p * 0.99 for p in prices1],
                'close': prices1,
                'volume': np.random.uniform(1000, 10000, 300)
            }, index=dates),
            'ETH': pd.DataFrame({
                'open': prices2,
                'high': [p * 1.01 for p in prices2],
                'low': [p * 0.99 for p in prices2],
                'close': prices2,
                'volume': np.random.uniform(1000, 10000, 300)
            }, index=dates)
        }
        
        # Ensure OHLC consistency
        for symbol in self.test_data:
            df = self.test_data[symbol]
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.get_strategy_name(), "Pairs Trading (Cointegration + Z-Score)")
    
    def test_pairs_analysis(self):
        """Test pairs analysis functionality"""
        # Analyze pairs
        pairs_analysis = self.strategy.analyze_pairs(self.test_data)
        
        # Check that pairs analysis is performed
        self.assertIsInstance(pairs_analysis, dict)
        
        # If pairs are found, check their structure
        if pairs_analysis:
            pair_key = list(pairs_analysis.keys())[0]
            pair_data = pairs_analysis[pair_key]
            
            self.assertIn('symbol1', pair_data)
            self.assertIn('symbol2', pair_data)
            self.assertIn('cointegration', pair_data)
            self.assertIn('spread_statistics', pair_data)
            self.assertIn('correlation', pair_data)
    
    def test_signal_generation(self):
        """Test signal generation for pairs"""
        # Analyze pairs first
        pairs_analysis = self.strategy.analyze_pairs(self.test_data)
        
        if pairs_analysis:
            # Generate signals
            signals = self.strategy.generate_signals(pairs_analysis)
            
            # Check that signals are generated
            self.assertIsInstance(signals, list)
            
            # If signals are generated, check their structure
            if signals:
                signal = signals[0]
                self.assertIsNotNone(signal.timestamp)
                self.assertIsNotNone(signal.signal_type)
                self.assertIsNotNone(signal.confidence)


class TestStrategyIntegration(unittest.TestCase):
    """Test cases for strategy integration and common functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_config = {
            'strategy': {
                'base_position_size': 0.1,
                'max_position_size': 0.3,
                'volatility_scaling': True
            }
        }
        self.risk_manager = Mock(spec=RiskManager)
    
    def test_strategy_registry(self):
        """Test strategy registry functionality"""
        from strategies import STRATEGY_REGISTRY, get_strategy, list_strategies
        
        # Check that all strategies are registered
        expected_strategies = ['momentum', 'mean_reversion', 'breakout', 'microstructure', 'pairs_trading']
        registered_strategies = list_strategies()
        
        for strategy in expected_strategies:
            self.assertIn(strategy, registered_strategies)
        
        # Test getting strategies by name
        for strategy_name in expected_strategies:
            strategy_class = get_strategy(strategy_name)
            self.assertIsNotNone(strategy_class)
    
    def test_common_strategy_interface(self):
        """Test that all strategies implement the common interface"""
        strategy_classes = [
            MomentumStrategy,
            MeanReversionStrategy,
            BreakoutStrategy,
            MicrostructureStrategy,
            PairsTradingStrategy
        ]
        
        for strategy_class in strategy_classes:
            strategy = strategy_class(self.base_config, self.risk_manager)
            
            # Check required methods
            self.assertTrue(hasattr(strategy, 'calculate_indicators'))
            self.assertTrue(hasattr(strategy, 'generate_signals'))
            self.assertTrue(hasattr(strategy, 'calculate_position_size'))
            self.assertTrue(hasattr(strategy, 'get_strategy_name'))
            self.assertTrue(hasattr(strategy, 'get_required_indicators'))
            
            # Check that methods return expected types
            self.assertIsInstance(strategy.get_strategy_name(), str)
            self.assertIsInstance(strategy.get_required_indicators(), list)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
