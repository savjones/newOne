"""
Breakout Strategy Implementation

This strategy identifies breakouts from established ranges using Donchian Channels
and Keltner Channels, with volume and volatility confirmation to filter signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, Trade
from utils.risk import RiskManager
from utils.data_utils import calculate_atr


@dataclass
class BreakoutConfig:
    """Configuration for Breakout Strategy"""
    # Channel Parameters
    donchian_period: int = 20
    keltner_period: int = 20
    keltner_multiplier: float = 2.0
    
    # Volume Confirmation
    volume_ma_period: int = 20
    volume_threshold: float = 1.5
    volume_surge_threshold: float = 2.0
    
    # Volatility Confirmation
    atr_period: int = 14
    volatility_threshold: float = 1.2
    volatility_expansion_threshold: float = 1.5
    
    # Breakout Parameters
    breakout_confirmation_bars: int = 2
    false_breakout_filter: bool = True
    consolidation_threshold: float = 0.02
    
    # Entry/Exit Rules
    entry_delay: int = 1  # Bars to wait after breakout
    stop_loss_atr_multiplier: float = 2.0
    take_profit_ratio: float = 3.0
    trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    
    # Position Sizing
    base_position_size: float = 0.1
    volatility_scaling: bool = True
    max_position_size: float = 0.3
    breakout_strength_scaling: bool = True
    
    # Performance Filters
    min_range_size: float = 0.05  # Minimum range as % of price
    max_range_size: float = 0.30  # Maximum range as % of price
    trend_filter: bool = True
    trend_ma_period: int = 50
    
    # Risk Management
    max_hold_period: int = 72  # Maximum bars to hold position
    correlation_threshold: float = 0.7


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy using Donchian and Keltner Channels
    
    This strategy identifies breakouts from established ranges and generates
    signals with volume and volatility confirmation.
    """
    
    def __init__(self, config: Dict, risk_manager: RiskManager):
        super().__init__(config, risk_manager)
        self.strategy_config = BreakoutConfig(**config.get('strategy', {}))
        self.donchian_upper = []
        self.donchian_lower = []
        self.keltner_upper = []
        self.keltner_lower = []
        self.keltner_middle = []
        self.volume_ma = []
        self.atr_values = []
        self.trend_ma = []
        self.consolidation_ranges = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for breakout strategy"""
        df = data.copy()
        
        # Calculate Donchian Channels
        df['donchian_upper'] = df['high'].rolling(self.strategy_config.donchian_period).max()
        df['donchian_lower'] = df['low'].rolling(self.strategy_config.donchian_period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        df['donchian_range'] = df['donchian_upper'] - df['donchian_lower']
        df['donchian_range_pct'] = df['donchian_range'] / df['close']
        
        # Calculate Keltner Channels
        df['atr'] = calculate_atr(df, period=self.strategy_config.atr_period)
        df['keltner_middle'] = df['close'].rolling(self.strategy_config.keltner_period).mean()
        df['keltner_upper'] = df['keltner_middle'] + (self.strategy_config.keltner_multiplier * df['atr'])
        df['keltner_lower'] = df['keltner_middle'] - (self.strategy_config.keltner_multiplier * df['atr'])
        
        # Calculate Volume Indicators
        df['volume_ma'] = df['volume'].rolling(self.strategy_config.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_surge'] = df['volume_ratio'] > self.strategy_config.volume_surge_threshold
        
        # Calculate Volatility Indicators
        df['volatility'] = df['atr'] / df['close']
        df['volatility_ma'] = df['volatility'].rolling(self.strategy_config.atr_period).mean()
        df['volatility_ratio'] = df['volatility'] / df['volatility_ma']
        df['volatility_expansion'] = df['volatility_ratio'] > self.strategy_config.volatility_expansion_threshold
        
        # Calculate Trend Filter
        if self.strategy_config.trend_filter:
            df['trend_ma'] = df['close'].rolling(self.strategy_config.trend_ma_period).mean()
            df['trend_direction'] = np.where(df['close'] > df['trend_ma'], 1, -1)
        
        # Calculate Breakout Indicators
        df['above_donchian'] = df['close'] > df['donchian_upper']
        df['below_donchian'] = df['close'] < df['donchian_lower']
        df['above_keltner'] = df['close'] > df['keltner_upper']
        df['below_keltner'] = df['close'] < df['keltner_lower']
        
        # Calculate Consolidation Detection
        df['consolidation'] = df['donchian_range_pct'] < self.strategy_config.consolidation_threshold
        
        # Calculate Breakout Strength
        df['breakout_strength'] = self._calculate_breakout_strength(df)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate breakout trading signals"""
        signals = []
        
        if len(data) < max(self.strategy_config.donchian_period, self.strategy_config.keltner_period):
            return signals
        
        current_bar = data.iloc[-1]
        prev_bars = data.iloc[-self.strategy_config.breakout_confirmation_bars-1:-1] if len(data) > self.strategy_config.breakout_confirmation_bars else None
        
        # Check if market conditions are suitable
        if not self._check_market_conditions(current_bar):
            return signals
        
        # Generate entry signals
        entry_signal = self._generate_entry_signal(current_bar, prev_bars)
        if entry_signal:
            signals.append(entry_signal)
        
        # Generate exit signals for existing positions
        exit_signal = self._generate_exit_signal(current_bar, prev_bars)
        if exit_signal:
            signals.append(exit_signal)
        
        return signals
    
    def _check_market_conditions(self, bar: pd.Series) -> bool:
        """Check if market conditions are suitable for breakout trading"""
        # Check range size bounds
        if (bar['donchian_range_pct'] < self.strategy_config.min_range_size or 
            bar['donchian_range_pct'] > self.strategy_config.max_range_size):
            return False
        
        # Check if not in consolidation
        if bar['consolidation']:
            return False
        
        # Check volume confirmation
        if bar['volume_ratio'] < self.strategy_config.volume_threshold:
            return False
        
        # Check volatility confirmation
        if bar['volatility_ratio'] < self.strategy_config.volatility_threshold:
            return False
        
        return True
    
    def _generate_entry_signal(self, current_bar: pd.Series, prev_bars: pd.DataFrame) -> Optional[Signal]:
        """Generate entry signal based on breakout logic"""
        if prev_bars is None or len(prev_bars) < self.strategy_config.breakout_confirmation_bars:
            return None
        
        # Check for bullish breakout
        if (current_bar['above_donchian'] and 
            current_bar['above_keltner'] and
            current_bar['volume_surge'] and
            current_bar['volatility_expansion']):
            
            # Confirm breakout over multiple bars
            if self._confirm_breakout(prev_bars, 'bullish'):
                return Signal(
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.LONG,
                    strength=SignalStrength.STRONG,
                    price=current_bar['close'],
                    confidence=self._calculate_signal_confidence(current_bar, 'bullish'),
                    metadata={
                        'strategy': 'breakout',
                        'entry_reason': 'bullish_breakout',
                        'breakout_strength': current_bar['breakout_strength'],
                        'volume_ratio': current_bar['volume_ratio'],
                        'volatility_ratio': current_bar['volatility_ratio'],
                        'range_size': current_bar['donchian_range_pct']
                    }
                )
        
        # Check for bearish breakout
        elif (current_bar['below_donchian'] and 
              current_bar['below_keltner'] and
              current_bar['volume_surge'] and
              current_bar['volatility_expansion']):
            
            # Confirm breakout over multiple bars
            if self._confirm_breakout(prev_bars, 'bearish'):
                return Signal(
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SHORT,
                    strength=SignalStrength.STRONG,
                    price=current_bar['close'],
                    confidence=self._calculate_signal_confidence(current_bar, 'bearish'),
                    metadata={
                        'strategy': 'breakout',
                        'entry_reason': 'bearish_breakout',
                        'breakout_strength': current_bar['breakout_strength'],
                        'volume_ratio': current_bar['volume_ratio'],
                        'volatility_ratio': current_bar['volatility_ratio'],
                        'range_size': current_bar['donchian_range_pct']
                    }
                )
        
        return None
    
    def _generate_exit_signal(self, current_bar: pd.Series, prev_bars: pd.DataFrame) -> Optional[Signal]:
        """Generate exit signal based on breakout logic"""
        # Exit long position on bearish reversal
        if (current_bar['below_keltner_middle'] and 
            current_bar['volatility_ratio'] > self.strategy_config.volatility_threshold):
            
            return Signal(
                timestamp=current_bar.name,
                symbol=current_bar.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_LONG,
                strength=SignalStrength.MEDIUM,
                price=current_bar['close'],
                confidence=0.8,
                metadata={
                    'strategy': 'breakout',
                    'exit_reason': 'bearish_reversal',
                    'volatility_ratio': current_bar['volatility_ratio']
                }
            )
        
        # Exit short position on bullish reversal
        if (current_bar['above_keltner_middle'] and 
            current_bar['volatility_ratio'] > self.strategy_config.volatility_threshold):
            
            return Signal(
                timestamp=current_bar.name,
                symbol=current_bar.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_SHORT,
                strength=SignalStrength.MEDIUM,
                price=current_bar['close'],
                confidence=0.8,
                metadata={
                    'strategy': 'breakout',
                    'exit_reason': 'bullish_reversal',
                    'volatility_ratio': current_bar['volatility_ratio']
                }
            )
        
        return None
    
    def _confirm_breakout(self, prev_bars: pd.DataFrame, direction: str) -> bool:
        """Confirm breakout over multiple bars to avoid false breakouts"""
        if direction == 'bullish':
            # Check if price stayed above breakout level
            return all(bar['close'] > bar['donchian_upper'] for _, bar in prev_bars.iterrows())
        else:
            # Check if price stayed below breakout level
            return all(bar['close'] < bar['donchian_lower'] for _, bar in prev_bars.iterrows())
    
    def _calculate_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the strength of breakout signals"""
        strength = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < 1:
                strength.iloc[i] = 0.0
                continue
            
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Calculate breakout strength based on multiple factors
            price_momentum = (current_bar['close'] - prev_bar['close']) / prev_bar['close']
            volume_factor = min(current_bar['volume_ratio'] / 2.0, 1.0)
            volatility_factor = min(current_bar['volatility_ratio'] / 2.0, 1.0)
            range_factor = min(current_bar['donchian_range_pct'] / 0.15, 1.0)
            
            # Combine factors
            strength.iloc[i] = (
                abs(price_momentum) * 0.4 +
                volume_factor * 0.3 +
                volatility_factor * 0.2 +
                range_factor * 0.1
            )
        
        return strength
    
    def _calculate_signal_confidence(self, bar: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Breakout strength contribution
        breakout_confidence = min(1.0, bar['breakout_strength']) * 0.3
        confidence += breakout_confidence
        
        # Volume confirmation
        volume_confidence = min(1.0, bar['volume_ratio'] / 2.0) * 0.25
        confidence += volume_confidence
        
        # Volatility confirmation
        volatility_confidence = min(1.0, bar['volatility_ratio'] / 2.0) * 0.25
        confidence += volatility_confidence
        
        # Range size contribution
        range_confidence = min(1.0, bar['donchian_range_pct'] / 0.15) * 0.2
        confidence += range_confidence
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                               portfolio_value: float) -> float:
        """Calculate position size based on volatility and breakout strength"""
        base_size = self.strategy_config.base_position_size
        
        # Scale by signal confidence
        confidence_multiplier = signal.confidence
        
        # Scale by volatility if enabled
        if self.strategy_config.volatility_scaling:
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
        
        # Scale by breakout strength if enabled
        if self.strategy_config.breakout_strength_scaling:
            breakout_strength = signal.metadata.get('breakout_strength', 0.5)
            strength_multiplier = 0.5 + breakout_strength * 0.5
        else:
            strength_multiplier = 1.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * volatility_multiplier * strength_multiplier
        
        # Apply limits
        position_size = min(position_size, self.strategy_config.max_position_size)
        position_size = max(position_size, 0.01)  # Minimum position size
        
        return position_size
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "Breakout (Donchian + Keltner + Volume/Volatility)"
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['donchian_upper', 'donchian_lower', 'keltner_upper', 'keltner_lower', 
                'volume_ma', 'atr', 'volatility', 'breakout_strength']
