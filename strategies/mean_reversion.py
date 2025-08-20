"""
Mean Reversion Strategy Implementation

This strategy identifies overbought/oversold conditions using VWAP and Bollinger Bands
to generate mean reversion signals. It includes volume confirmation and dynamic
position sizing based on deviation from the mean.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, Trade
from utils.risk import RiskManager
from utils.data_utils import calculate_vwap, calculate_bollinger_bands


@dataclass
class MeanReversionConfig:
    """Configuration for Mean Reversion Strategy"""
    # VWAP Parameters
    vwap_period: int = 20
    vwap_deviation_threshold: float = 2.0
    
    # Bollinger Bands Parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_squeeze_threshold: float = 0.5
    
    # Volume Confirmation
    volume_ma_period: int = 20
    volume_threshold: float = 1.5
    
    # Mean Reversion Parameters
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # Entry/Exit Rules
    entry_threshold: float = 2.5  # Standard deviations from mean
    exit_threshold: float = 0.5   # Return to mean
    max_hold_period: int = 48     # Maximum bars to hold position
    
    # Position Sizing
    base_position_size: float = 0.1
    volatility_scaling: bool = True
    max_position_size: float = 0.3
    
    # Risk Management
    stop_loss_atr_multiplier: float = 2.0
    take_profit_ratio: float = 2.0
    trailing_stop: bool = True
    
    # Performance Filters
    min_volatility: float = 0.01
    max_volatility: float = 0.10
    trend_filter: bool = True
    trend_ma_period: int = 50


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using VWAP and Bollinger Bands
    
    This strategy identifies overbought/oversold conditions and generates
    mean reversion signals when price deviates significantly from the mean.
    """
    
    def __init__(self, config: Dict, risk_manager: RiskManager):
        super().__init__(config, risk_manager)
        self.strategy_config = MeanReversionConfig(**config.get('strategy', {}))
        self.vwap_values = []
        self.bb_upper = []
        self.bb_lower = []
        self.bb_middle = []
        self.rsi_values = []
        self.volume_ma = []
        self.trend_ma = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for mean reversion strategy"""
        df = data.copy()
        
        # Calculate VWAP
        df['vwap'] = calculate_vwap(df, period=self.strategy_config.vwap_period)
        
        # Calculate Bollinger Bands
        bb_data = calculate_bollinger_bands(
            df['close'], 
            period=self.strategy_config.bb_period,
            std_dev=self.strategy_config.bb_std_dev
        )
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.strategy_config.rsi_period)
        
        # Calculate Volume Moving Average
        df['volume_ma'] = df['volume'].rolling(self.strategy_config.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate Trend Filter
        if self.strategy_config.trend_filter:
            df['trend_ma'] = df['close'].rolling(self.strategy_config.trend_ma_period).mean()
            df['trend_direction'] = np.where(df['close'] > df['trend_ma'], 1, -1)
        
        # Calculate Volatility
        df['atr'] = self._calculate_atr(df, period=14)
        df['volatility'] = df['atr'] / df['close']
        
        # Calculate Deviations from Mean
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        df['bb_deviation'] = (df['close'] - df['bb_middle']) / df['bb_middle']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion trading signals"""
        signals = []
        
        if len(data) < max(self.strategy_config.vwap_period, self.strategy_config.bb_period):
            return signals
        
        current_bar = data.iloc[-1]
        prev_bar = data.iloc[-2] if len(data) > 1 else None
        
        # Check if market conditions are suitable
        if not self._check_market_conditions(current_bar):
            return signals
        
        # Generate entry signals
        entry_signal = self._generate_entry_signal(current_bar, prev_bar)
        if entry_signal:
            signals.append(entry_signal)
        
        # Generate exit signals for existing positions
        exit_signal = self._generate_exit_signal(current_bar, prev_bar)
        if exit_signal:
            signals.append(exit_signal)
        
        return signals
    
    def _check_market_conditions(self, bar: pd.Series) -> bool:
        """Check if market conditions are suitable for mean reversion"""
        # Check volatility bounds
        if (bar['volatility'] < self.strategy_config.min_volatility or 
            bar['volatility'] > self.strategy_config.max_volatility):
            return False
        
        # Check volume confirmation
        if bar['volume_ratio'] < self.strategy_config.volume_threshold:
            return False
        
        # Check Bollinger Band squeeze (low volatility period)
        if bar['bb_width'] < self.strategy_config.bb_squeeze_threshold:
            return False
        
        return True
    
    def _generate_entry_signal(self, current_bar: pd.Series, prev_bar: pd.Series) -> Optional[Signal]:
        """Generate entry signal based on mean reversion logic"""
        # Check for oversold condition (price below lower Bollinger Band)
        if (current_bar['close'] < current_bar['bb_lower'] and 
            current_bar['rsi'] < self.strategy_config.rsi_oversold and
            current_bar['vwap_deviation'] < -self.strategy_config.entry_threshold):
            
            # Additional confirmation: price starting to move up
            if prev_bar and current_bar['close'] > prev_bar['close']:
                return Signal(
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.LONG,
                    strength=SignalStrength.STRONG,
                    price=current_bar['close'],
                    confidence=self._calculate_signal_confidence(current_bar, 'long'),
                    metadata={
                        'strategy': 'mean_reversion',
                        'entry_reason': 'oversold_bounce',
                        'vwap_deviation': current_bar['vwap_deviation'],
                        'bb_deviation': current_bar['bb_deviation'],
                        'rsi': current_bar['rsi']
                    }
                )
        
        # Check for overbought condition (price above upper Bollinger Band)
        elif (current_bar['close'] > current_bar['bb_upper'] and 
              current_bar['rsi'] > self.strategy_config.rsi_overbought and
              current_bar['vwap_deviation'] > self.strategy_config.entry_threshold):
            
            # Additional confirmation: price starting to move down
            if prev_bar and current_bar['close'] < prev_bar['close']:
                return Signal(
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SHORT,
                    strength=SignalStrength.STRONG,
                    price=current_bar['close'],
                    confidence=self._calculate_signal_confidence(current_bar, 'short'),
                    metadata={
                        'strategy': 'mean_reversion',
                        'entry_reason': 'overbought_rejection',
                        'vwap_deviation': current_bar['vwap_deviation'],
                        'bb_deviation': current_bar['bb_deviation'],
                        'rsi': current_bar['rsi']
                    }
                )
        
        return None
    
    def _generate_exit_signal(self, current_bar: pd.Series, prev_bar: pd.Series) -> Optional[Signal]:
        """Generate exit signal based on mean reversion logic"""
        # Exit long position when price returns to VWAP
        if (current_bar['vwap_deviation'] > -self.strategy_config.exit_threshold and
            current_bar['vwap_deviation'] < self.strategy_config.exit_threshold):
            
            return Signal(
                timestamp=current_bar.name,
                symbol=current_bar.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_LONG,
                strength=SignalStrength.MEDIUM,
                price=current_bar['close'],
                confidence=0.8,
                metadata={
                    'strategy': 'mean_reversion',
                    'exit_reason': 'return_to_mean',
                    'vwap_deviation': current_bar['vwap_deviation']
                }
            )
        
        # Exit short position when price returns to VWAP
        if (current_bar['vwap_deviation'] > -self.strategy_config.exit_threshold and
            current_bar['vwap_deviation'] < self.strategy_config.exit_threshold):
            
            return Signal(
                timestamp=current_bar.name,
                symbol=current_bar.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_SHORT,
                strength=SignalStrength.MEDIUM,
                price=current_bar['close'],
                confidence=0.8,
                metadata={
                    'strategy': 'mean_reversion',
                    'exit_reason': 'return_to_mean',
                    'vwap_deviation': current_bar['vwap_deviation']
                }
            )
        
        return None
    
    def _calculate_signal_confidence(self, bar: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # RSI contribution
        if signal_type == 'long':
            rsi_confidence = max(0, (30 - bar['rsi']) / 30) * 0.3
        else:
            rsi_confidence = max(0, (bar['rsi'] - 70) / 30) * 0.3
        confidence += rsi_confidence
        
        # Volume confirmation
        volume_confidence = min(1.0, bar['volume_ratio'] / 2.0) * 0.2
        confidence += volume_confidence
        
        # Deviation strength
        deviation_confidence = min(1.0, abs(bar['vwap_deviation']) / 0.05) * 0.3
        confidence += deviation_confidence
        
        # Volatility contribution
        vol_confidence = (bar['volatility'] - self.strategy_config.min_volatility) / \
                        (self.strategy_config.max_volatility - self.strategy_config.min_volatility)
        confidence += vol_confidence * 0.2
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                               portfolio_value: float) -> float:
        """Calculate position size based on volatility and confidence"""
        base_size = self.strategy_config.base_position_size
        
        # Scale by signal confidence
        confidence_multiplier = signal.confidence
        
        # Scale by volatility if enabled
        if self.strategy_config.volatility_scaling:
            # Use ATR-based volatility scaling
            current_data = self.get_latest_data()
            if current_data is not None and len(current_data) > 0:
                atr = current_data.iloc[-1].get('atr', 0)
                if atr > 0:
                    volatility_multiplier = 1.0 / (atr / current_price)
                    volatility_multiplier = np.clip(volatility_multiplier, 0.5, 2.0)
                else:
                    volatility_multiplier = 1.0
            else:
                volatility_multiplier = 1.0
        else:
            volatility_multiplier = 1.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * volatility_multiplier
        
        # Apply limits
        position_size = min(position_size, self.strategy_config.max_position_size)
        position_size = max(position_size, 0.01)  # Minimum position size
        
        return position_size
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "Mean Reversion (VWAP + Bollinger Bands)"
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['vwap', 'bb_upper', 'bb_lower', 'bb_middle', 'rsi', 'volume_ma', 'atr']
