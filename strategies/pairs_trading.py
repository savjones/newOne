"""
Pairs Trading Strategy Implementation

This strategy identifies cointegrated pairs and trades the spread when it
deviates from its historical mean, using z-score based entry/exit signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from statsmodels.tsa.stattools import coint

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, Trade
from utils.risk import RiskManager


@dataclass
class PairsTradingConfig:
    """Configuration for Pairs Trading Strategy"""
    # Cointegration Parameters
    cointegration_lookback: int = 252  # Days to test cointegration
    cointegration_pvalue: float = 0.05  # Maximum p-value for cointegration
    cointegration_retest_frequency: int = 21  # Retest every 21 days
    
    # Spread Analysis
    spread_lookback: int = 60  # Days to calculate spread statistics
    z_score_entry: float = 2.0  # Z-score threshold for entry
    z_score_exit: float = 0.5   # Z-score threshold for exit
    z_score_stop: float = 4.0   # Z-score threshold for stop loss
    
    # Position Sizing
    base_position_size: float = 0.1
    volatility_scaling: bool = True
    max_position_size: float = 0.3
    correlation_scaling: bool = True
    
    # Risk Management
    stop_loss_atr_multiplier: float = 2.0
    take_profit_ratio: float = 2.0
    max_hold_period: int = 120  # Maximum days to hold position
    correlation_threshold: float = 0.7
    
    # Performance Filters
    min_correlation: float = 0.6
    max_spread_volatility: float = 0.1
    min_spread_mean_reversion: float = 0.3
    
    # Pairs Selection
    min_pairs: int = 3
    max_pairs: int = 10
    pairs_update_frequency: int = 7  # Update pairs every 7 days
    
    # Spread Calculation
    spread_method: str = "linear"  # linear, log, ratio
    hedge_ratio_method: str = "ols"  # ols, kalman, rolling
    
    # Machine Learning Integration
    ml_enabled: bool = True
    feature_engineering: bool = True


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy using Cointegration and Z-Score Analysis
    
    This strategy identifies cointegrated pairs and trades the spread when
    it deviates from its historical mean.
    """
    
    def __init__(self, config: Dict, risk_manager: RiskManager):
        super().__init__(config, risk_manager)
        self.strategy_config = PairsTradingConfig(**config.get('strategy', {}))
        self.pairs_data = {}
        self.cointegration_results = {}
        self.spread_statistics = {}
        self.hedge_ratios = {}
        self.pairs_correlation = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for pairs trading strategy"""
        df = data.copy()
        
        # This strategy requires multiple symbols, so we'll handle it differently
        # The actual implementation would be in a separate method for pairs analysis
        
        # For now, add basic indicators that might be useful
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df
    
    def analyze_pairs(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Analyze potential trading pairs for cointegration"""
        pairs_analysis = {}
        
        symbols = list(market_data.keys())
        if len(symbols) < 2:
            return pairs_analysis
        
        # Generate all possible pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                pair_key = f"{symbol1}_{symbol2}"
                
                # Get price data for both symbols
                data1 = market_data[symbol1]['close'].dropna()
                data2 = market_data[symbol2]['close'].dropna()
                
                # Align data
                aligned_data = pd.concat([data1, data2], axis=1).dropna()
                if len(aligned_data) < self.strategy_config.cointegration_lookback:
                    continue
                
                # Test cointegration
                cointegration_result = self._test_cointegration(
                    aligned_data.iloc[:, 0], 
                    aligned_data.iloc[:, 1]
                )
                
                if cointegration_result['is_cointegrated']:
                    # Calculate spread and statistics
                    spread_data = self._calculate_spread(
                        aligned_data.iloc[:, 0], 
                        aligned_data.iloc[:, 1],
                        cointegration_result['hedge_ratio']
                    )
                    
                    spread_stats = self._calculate_spread_statistics(spread_data)
                    
                    # Calculate correlation
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    
                    pairs_analysis[pair_key] = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'cointegration': cointegration_result,
                        'spread_statistics': spread_stats,
                        'correlation': correlation,
                        'last_update': pd.Timestamp.now(),
                        'data_length': len(aligned_data)
                    }
        
        return pairs_analysis
    
    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """Test for cointegration between two time series"""
        try:
            # Perform Engle-Granger cointegration test
            score, pvalue, _ = coint(series1, series2)
            
            is_cointegrated = pvalue < self.strategy_config.cointegration_pvalue
            
            # Calculate hedge ratio using OLS
            hedge_ratio = self._calculate_hedge_ratio(series1, series2)
            
            return {
                'is_cointegrated': is_cointegrated,
                'score': score,
                'pvalue': pvalue,
                'hedge_ratio': hedge_ratio,
                'test_method': 'engle_granger'
            }
        except Exception as e:
            return {
                'is_cointegrated': False,
                'score': np.nan,
                'pvalue': np.nan,
                'hedge_ratio': 1.0,
                'test_method': 'engle_granger',
                'error': str(e)
            }
    
    def _calculate_hedge_ratio(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate hedge ratio between two series"""
        try:
            if self.strategy_config.hedge_ratio_method == 'ols':
                # Simple OLS regression
                X = series1.values.reshape(-1, 1)
                y = series2.values
                
                # Add constant
                X = np.column_stack([np.ones(len(X)), X])
                
                # OLS solution
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                return beta[1]  # Return slope coefficient
            else:
                # Default to 1.0
                return 1.0
        except:
            return 1.0
    
    def _calculate_spread(self, series1: pd.Series, series2: pd.Series, 
                         hedge_ratio: float) -> pd.Series:
        """Calculate the spread between two series"""
        if self.strategy_config.spread_method == 'linear':
            spread = series2 - hedge_ratio * series1
        elif self.strategy_config.spread_method == 'log':
            spread = np.log(series2) - hedge_ratio * np.log(series1)
        elif self.strategy_config.spread_method == 'ratio':
            spread = series2 / (hedge_ratio * series1) - 1
        else:
            spread = series2 - hedge_ratio * series1
        
        return spread
    
    def _calculate_spread_statistics(self, spread: pd.Series) -> Dict:
        """Calculate statistics for the spread series"""
        if len(spread) < self.strategy_config.spread_lookback:
            return {}
        
        # Use rolling window for recent statistics
        recent_spread = spread.tail(self.strategy_config.spread_lookback)
        
        mean = recent_spread.mean()
        std = recent_spread.std()
        
        # Calculate mean reversion strength
        mean_reversion = self._calculate_mean_reversion_strength(recent_spread)
        
        # Calculate current z-score
        current_zscore = (spread.iloc[-1] - mean) / std if std > 0 else 0
        
        return {
            'mean': mean,
            'std': std,
            'current_zscore': current_zscore,
            'mean_reversion_strength': mean_reversion,
            'min_zscore': (recent_spread.min() - mean) / std if std > 0 else 0,
            'max_zscore': (recent_spread.max() - mean) / std if std > 0 else 0
        }
    
    def _calculate_mean_reversion_strength(self, spread: pd.Series) -> float:
        """Calculate the strength of mean reversion"""
        if len(spread) < 2:
            return 0.0
        
        # Calculate autocorrelation at lag 1
        autocorr = spread.autocorr(lag=1)
        
        # Convert to mean reversion strength (negative autocorr = mean reversion)
        if autocorr < 0:
            return abs(autocorr)
        else:
            return 0.0
    
    def generate_signals(self, pairs_analysis: Dict[str, Dict]) -> List[Signal]:
        """Generate pairs trading signals based on spread analysis"""
        signals = []
        
        for pair_key, pair_data in pairs_analysis.items():
            if not pair_data['cointegration']['is_cointegrated']:
                continue
            
            spread_stats = pair_data['spread_statistics']
            if not spread_stats:
                continue
            
            current_zscore = spread_stats['current_zscore']
            
            # Generate entry signals
            entry_signal = self._generate_entry_signal(pair_key, pair_data, current_zscore)
            if entry_signal:
                signals.append(entry_signal)
            
            # Generate exit signals
            exit_signal = self._generate_exit_signal(pair_key, pair_data, current_zscore)
            if exit_signal:
                signals.append(exit_signal)
        
        return signals
    
    def _generate_entry_signal(self, pair_key: str, pair_data: Dict, 
                              current_zscore: float) -> Optional[Signal]:
        """Generate entry signal based on spread z-score"""
        spread_stats = pair_data['spread_statistics']
        
        # Check if spread is significantly deviated
        if abs(current_zscore) < self.strategy_config.z_score_entry:
            return None
        
        # Check mean reversion strength
        if spread_stats['mean_reversion_strength'] < self.strategy_config.min_spread_mean_reversion:
            return None
        
        # Check correlation
        if pair_data['correlation'] < self.strategy_config.min_correlation:
            return None
        
        # Generate long-short signal
        if current_zscore > self.strategy_config.z_score_entry:
            # Spread is wide, short the spread (long symbol1, short symbol2)
            return Signal(
                timestamp=pd.Timestamp.now(),
                symbol=pair_key,
                signal_type=SignalType.PAIRS_LONG_SHORT,
                strength=SignalStrength.STRONG,
                price=0.0,  # Will be calculated separately for each leg
                confidence=self._calculate_signal_confidence(pair_data, 'long_short'),
                metadata={
                    'strategy': 'pairs_trading',
                    'entry_reason': 'spread_widening',
                    'zscore': current_zscore,
                    'hedge_ratio': pair_data['cointegration']['hedge_ratio'],
                    'symbol1': pair_data['symbol1'],
                    'symbol2': pair_data['symbol2'],
                    'position_type': 'long_short'
                }
            )
        
        elif current_zscore < -self.strategy_config.z_score_entry:
            # Spread is narrow, long the spread (short symbol1, long symbol2)
            return Signal(
                timestamp=pd.Timestamp.now(),
                symbol=pair_key,
                signal_type=SignalType.PAIRS_SHORT_LONG,
                strength=SignalStrength.STRONG,
                price=0.0,  # Will be calculated separately for each leg
                confidence=self._calculate_signal_confidence(pair_data, 'short_long'),
                metadata={
                    'strategy': 'pairs_trading',
                    'entry_reason': 'spread_narrowing',
                    'zscore': current_zscore,
                    'hedge_ratio': pair_data['cointegration']['hedge_ratio'],
                    'symbol1': pair_data['symbol1'],
                    'symbol2': pair_data['symbol2'],
                    'position_type': 'short_long'
                }
            )
        
        return None
    
    def _generate_exit_signal(self, pair_key: str, pair_data: Dict, 
                             current_zscore: float) -> Optional[Signal]:
        """Generate exit signal based on spread z-score"""
        spread_stats = pair_data['spread_statistics']
        
        # Exit when spread returns to mean
        if abs(current_zscore) < self.strategy_config.z_score_exit:
            return Signal(
                timestamp=pd.Timestamp.now(),
                symbol=pair_key,
                signal_type=SignalType.EXIT_PAIRS,
                strength=SignalStrength.MEDIUM,
                price=0.0,
                confidence=0.8,
                metadata={
                    'strategy': 'pairs_trading',
                    'exit_reason': 'spread_convergence',
                    'zscore': current_zscore
                }
            )
        
        # Stop loss when spread becomes too extreme
        if abs(current_zscore) > self.strategy_config.z_score_stop:
            return Signal(
                timestamp=pd.Timestamp.now(),
                symbol=pair_key,
                signal_type=SignalType.EXIT_PAIRS,
                strength=SignalStrength.STRONG,
                price=0.0,
                confidence=0.9,
                metadata={
                    'strategy': 'pairs_trading',
                    'exit_reason': 'stop_loss',
                    'zscore': current_zscore
                }
            )
        
        return None
    
    def _calculate_signal_confidence(self, pair_data: Dict, signal_type: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Cointegration strength
        cointegration_pvalue = pair_data['cointegration']['pvalue']
        cointegration_confidence = (0.05 - cointegration_pvalue) / 0.05 * 0.3
        confidence += max(0, cointegration_confidence)
        
        # Mean reversion strength
        mean_reversion = pair_data['spread_statistics']['mean_reversion_strength']
        mean_reversion_confidence = min(mean_reversion, 1.0) * 0.3
        confidence += mean_reversion_confidence
        
        # Correlation strength
        correlation = pair_data['correlation']
        correlation_confidence = (correlation - 0.6) / 0.4 * 0.2
        confidence += max(0, correlation_confidence)
        
        # Data quality
        data_length = pair_data['data_length']
        data_confidence = min(data_length / 252, 1.0) * 0.2
        confidence += data_confidence
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                               portfolio_value: float) -> float:
        """Calculate position size for pairs trading"""
        base_size = self.strategy_config.base_position_size
        
        # Scale by signal confidence
        confidence_multiplier = signal.confidence
        
        # Scale by correlation if enabled
        if self.strategy_config.correlation_scaling:
            pair_key = signal.symbol
            if pair_key in self.pairs_correlation:
                correlation = self.pairs_correlation[pair_key]
                correlation_multiplier = 0.5 + correlation * 0.5
            else:
                correlation_multiplier = 1.0
        else:
            correlation_multiplier = 1.0
        
        # Scale by volatility if enabled
        if self.strategy_config.volatility_scaling:
            # Use spread volatility for scaling
            volatility_multiplier = 1.0  # Placeholder for actual volatility calculation
        else:
            volatility_multiplier = 1.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * correlation_multiplier * volatility_multiplier
        
        # Apply limits
        position_size = min(position_size, self.strategy_config.max_position_size)
        position_size = max(position_size, 0.01)  # Minimum position size
        
        return position_size
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "Pairs Trading (Cointegration + Z-Score)"
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['spread', 'zscore', 'hedge_ratio', 'correlation', 'mean_reversion_strength']
    
    def update_pairs_analysis(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update pairs analysis with new market data"""
        # This would be called periodically to update cointegration tests
        # and spread statistics
        pass
    
    def get_trading_pairs(self) -> List[str]:
        """Get list of currently tradeable pairs"""
        return [pair_key for pair_key, pair_data in self.pairs_data.items() 
                if pair_data['cointegration']['is_cointegrated']]
    
    def get_pair_statistics(self, pair_key: str) -> Optional[Dict]:
        """Get statistics for a specific pair"""
        return self.pairs_data.get(pair_key, {}).get('spread_statistics')
