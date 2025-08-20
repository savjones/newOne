"""
Ultimate Oscillator + ADX + MFI Strategy Implementation

This strategy combines three powerful momentum and trend indicators:
- Ultimate Oscillator: Multi-timeframe momentum oscillator
- ADX (Average Directional Index): Trend strength indicator  
- MFI (Money Flow Index): Volume-weighted momentum oscillator

The strategy generates signals when all three indicators align to confirm
strong trending moves with good momentum and volume support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, Trade
from utils.risk import RiskManager


@dataclass
class UltimateOscillatorConfig:
    """Configuration for Ultimate Oscillator + ADX + MFI Strategy"""
    
    # Ultimate Oscillator Parameters
    uo_short_period: int = 7
    uo_medium_period: int = 14
    uo_long_period: int = 28
    uo_short_weight: float = 4.0
    uo_medium_weight: float = 2.0
    uo_long_weight: float = 1.0
    
    # Ultimate Oscillator Thresholds
    uo_oversold: float = 30.0
    uo_overbought: float = 70.0
    uo_extreme_oversold: float = 20.0
    uo_extreme_overbought: float = 80.0
    
    # ADX Parameters
    adx_period: int = 14
    adx_smoothing: int = 14
    adx_trend_threshold: float = 25.0
    adx_strong_trend: float = 40.0
    
    # MFI Parameters
    mfi_period: int = 14
    mfi_oversold: float = 20.0
    mfi_overbought: float = 80.0
    mfi_extreme_oversold: float = 10.0
    mfi_extreme_overbought: float = 90.0
    
    # Signal Generation Rules
    require_all_confirm: bool = True
    use_divergence_signals: bool = True
    min_signal_separation: int = 5  # Minimum bars between signals
    
    # Entry/Exit Rules
    entry_confirmation_bars: int = 2
    exit_on_opposite_extreme: bool = True
    use_trailing_stops: bool = True
    
    # Position Sizing
    base_position_size: float = 0.1
    volatility_scaling: bool = True
    max_position_size: float = 0.25
    confidence_scaling: bool = True
    
    # Risk Management
    stop_loss_atr_multiplier: float = 2.0
    take_profit_ratio: float = 2.5
    trailing_stop_atr_multiplier: float = 1.5
    max_hold_period: int = 50
    
    # Performance Filters
    min_adx_for_entry: float = 20.0
    max_adx_for_reversal: float = 60.0
    volume_confirmation: bool = True
    trend_filter: bool = True


class UltimateOscillatorStrategy(BaseStrategy):
    """
    Ultimate Oscillator + ADX + MFI Strategy
    
    This strategy combines three complementary indicators to identify high-probability
    trading opportunities:
    
    1. Ultimate Oscillator: Identifies momentum extremes across multiple timeframes
    2. ADX: Confirms trend strength and direction
    3. MFI: Validates moves with volume-weighted momentum
    
    Entry Conditions:
    - Ultimate Oscillator in oversold/overbought territory
    - ADX above threshold indicating strong trend
    - MFI confirming momentum direction
    - Optional volume and trend confirmation
    
    Exit Conditions:
    - Opposite extreme readings
    - ADX weakening
    - Stop loss or take profit levels
    """
    
    def __init__(self, config: Dict, risk_manager: RiskManager):
        # Set up instance variables first
        self.strategy_config = UltimateOscillatorConfig(**config.get('strategy', {}))
        
        # State tracking
        self.last_signal_bar = -1
        self.position_entry_bar = -1
        
        # Indicator history for divergence analysis
        self.uo_history = []
        self.adx_history = []
        self.mfi_history = []
        self.price_history = []
        
        # Call parent constructor (which will call _initialize_strategy)
        super().__init__("Ultimate Oscillator + ADX + MFI", config, risk_manager)
        
    def _initialize_strategy(self) -> None:
        """Initialize strategy-specific components."""
        # Reset state tracking
        self.last_signal_bar = -1
        self.position_entry_bar = -1
        self._last_data_length = 0
        
        # Clear indicator history
        self.uo_history.clear()
        self.adx_history.clear()
        self.mfi_history.clear()
        self.price_history.clear()
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the strategy"""
        df = data.copy()
        
        # Calculate Ultimate Oscillator
        df = self._calculate_ultimate_oscillator(df)
        
        # Calculate ADX and Directional Movement
        df = self._calculate_adx(df)
        
        # Calculate Money Flow Index
        df = self._calculate_mfi(df)
        
        # Calculate additional indicators
        df['atr'] = self._calculate_atr(df, period=14)
        df['volatility'] = df['atr'] / df['close']
        
        # Calculate trend filter
        if self.strategy_config.trend_filter:
            df['trend_ma'] = df['close'].rolling(50).mean()
            df['trend_direction'] = np.where(df['close'] > df['trend_ma'], 1, -1)
        
        # Calculate volume confirmation
        if self.strategy_config.volume_confirmation:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate composite signals
        df['composite_signal'] = self._calculate_composite_signal(df)
        df['signal_strength'] = self._calculate_signal_strength(df)
        
        return df
    
    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ultimate Oscillator"""
        # Calculate True Low (TL) and Buying Pressure (BP)
        prev_close = df['close'].shift(1)
        
        # True Low = min(Low, Previous Close)
        true_low = np.minimum(df['low'], prev_close)
        
        # Buying Pressure = Close - True Low
        buying_pressure = df['close'] - true_low
        
        # True Range = max(High, Previous Close) - True Low
        true_high = np.maximum(df['high'], prev_close)
        true_range = true_high - true_low
        
        # Calculate averages for different periods
        bp_short = buying_pressure.rolling(self.strategy_config.uo_short_period).sum()
        tr_short = true_range.rolling(self.strategy_config.uo_short_period).sum()
        raw_uo_short = (bp_short / tr_short) * 100
        
        bp_medium = buying_pressure.rolling(self.strategy_config.uo_medium_period).sum()
        tr_medium = true_range.rolling(self.strategy_config.uo_medium_period).sum()
        raw_uo_medium = (bp_medium / tr_medium) * 100
        
        bp_long = buying_pressure.rolling(self.strategy_config.uo_long_period).sum()
        tr_long = true_range.rolling(self.strategy_config.uo_long_period).sum()
        raw_uo_long = (bp_long / tr_long) * 100
        
        # Calculate weighted Ultimate Oscillator
        total_weight = (self.strategy_config.uo_short_weight + 
                       self.strategy_config.uo_medium_weight + 
                       self.strategy_config.uo_long_weight)
        
        df['ultimate_oscillator'] = (
            (raw_uo_short * self.strategy_config.uo_short_weight +
             raw_uo_medium * self.strategy_config.uo_medium_weight +
             raw_uo_long * self.strategy_config.uo_long_weight) / total_weight
        )
        
        # Calculate UO signals
        df['uo_oversold'] = df['ultimate_oscillator'] < self.strategy_config.uo_oversold
        df['uo_overbought'] = df['ultimate_oscillator'] > self.strategy_config.uo_overbought
        df['uo_extreme_oversold'] = df['ultimate_oscillator'] < self.strategy_config.uo_extreme_oversold
        df['uo_extreme_overbought'] = df['ultimate_oscillator'] > self.strategy_config.uo_extreme_overbought
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX and Directional Movement Indicators using proper Wilder's smoothing"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        n = len(df)
        period = self.strategy_config.adx_period
        smoothing = self.strategy_config.adx_smoothing
        
        # Initialize arrays
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        adx = np.zeros(n)
        
        # Calculate True Range and Directional Movement
        for i in range(1, n):
            # True Range
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
            
            # Directional Movement
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Wilder's smoothing for TR, +DM, -DM
        tr_smooth = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)
        
        # Initial smoothed values (simple average for first period)
        if n > period:
            tr_smooth[period-1] = np.mean(tr[1:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
            
            # Wilder's smoothing for subsequent values
            for i in range(period, n):
                tr_smooth[i] = (tr_smooth[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
                
                # Calculate DI
                if tr_smooth[i] != 0:
                    plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                    minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
                
                # Calculate DX
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        
        # Calculate ADX using Wilder's smoothing on DX
        if n > period + smoothing:
            # Initial ADX (simple average of first smoothing period of DX values)
            adx[period + smoothing - 1] = np.mean(dx[period:period + smoothing])
            
            # Wilder's smoothing for subsequent ADX values
            for i in range(period + smoothing, n):
                adx[i] = (adx[i-1] * (smoothing-1) + dx[i]) / smoothing
        
        # Store results in dataframe
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = adx
        
        # ADX signals
        df['adx_trending'] = df['adx'] > self.strategy_config.adx_trend_threshold
        df['adx_strong_trend'] = df['adx'] > self.strategy_config.adx_strong_trend
        df['bullish_di'] = df['plus_di'] > df['minus_di']
        df['bearish_di'] = df['minus_di'] > df['plus_di']
        
        return df
    
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Money Flow Index"""
        # Typical Price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Raw Money Flow
        raw_money_flow = typical_price * df['volume']
        
        # Positive and Negative Money Flow
        price_change = typical_price.diff()
        positive_flow = np.where(price_change > 0, raw_money_flow, 0)
        negative_flow = np.where(price_change < 0, raw_money_flow, 0)
        
        # Sum over period
        period = self.strategy_config.mfi_period
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        # Money Flow Index (avoid division by zero)
        money_ratio = np.where(negative_mf != 0, positive_mf / negative_mf, 0)
        mfi = np.where(money_ratio != 0, 100 - (100 / (1 + money_ratio)), 50)
        
        df['mfi'] = mfi
        
        # MFI signals
        df['mfi_oversold'] = df['mfi'] < self.strategy_config.mfi_oversold
        df['mfi_overbought'] = df['mfi'] > self.strategy_config.mfi_overbought
        df['mfi_extreme_oversold'] = df['mfi'] < self.strategy_config.mfi_extreme_oversold
        df['mfi_extreme_overbought'] = df['mfi'] > self.strategy_config.mfi_extreme_overbought
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return atr
    
    def _calculate_composite_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite signal from all indicators"""
        signal = pd.Series(0, index=df.index)
        
        # Bullish signals
        bullish_uo = df['uo_oversold'] | df['uo_extreme_oversold']
        bullish_adx = df['adx_trending'] & df['bullish_di']
        bullish_mfi = df['mfi_oversold'] | df['mfi_extreme_oversold']
        
        # Bearish signals
        bearish_uo = df['uo_overbought'] | df['uo_extreme_overbought']
        bearish_adx = df['adx_trending'] & df['bearish_di']
        bearish_mfi = df['mfi_overbought'] | df['mfi_extreme_overbought']
        
        if self.strategy_config.require_all_confirm:
            # All indicators must agree
            signal = np.where(bullish_uo & bullish_adx & bullish_mfi, 1,
                            np.where(bearish_uo & bearish_adx & bearish_mfi, -1, 0))
        else:
            # Majority rules
            bullish_votes = bullish_uo.astype(int) + bullish_adx.astype(int) + bullish_mfi.astype(int)
            bearish_votes = bearish_uo.astype(int) + bearish_adx.astype(int) + bearish_mfi.astype(int)
            
            signal = np.where(bullish_votes >= 2, 1,
                            np.where(bearish_votes >= 2, -1, 0))
        
        return pd.Series(signal, index=df.index)
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on indicator alignment"""
        strength = pd.Series(0.0, index=df.index)
        
        for i in range(len(df)):
            if i < max(self.strategy_config.uo_long_period, 
                      self.strategy_config.adx_period, 
                      self.strategy_config.mfi_period):
                continue
            
            # UO strength component
            uo_val = df.iloc[i]['ultimate_oscillator']
            if uo_val < self.strategy_config.uo_extreme_oversold:
                uo_strength = 1.0
            elif uo_val < self.strategy_config.uo_oversold:
                uo_strength = 0.7
            elif uo_val > self.strategy_config.uo_extreme_overbought:
                uo_strength = 1.0
            elif uo_val > self.strategy_config.uo_overbought:
                uo_strength = 0.7
            else:
                uo_strength = 0.0
            
            # ADX strength component
            adx_val = df.iloc[i]['adx']
            if adx_val > self.strategy_config.adx_strong_trend:
                adx_strength = 1.0
            elif adx_val > self.strategy_config.adx_trend_threshold:
                adx_strength = 0.6
            else:
                adx_strength = 0.0
            
            # MFI strength component
            mfi_val = df.iloc[i]['mfi']
            if mfi_val < self.strategy_config.mfi_extreme_oversold:
                mfi_strength = 1.0
            elif mfi_val < self.strategy_config.mfi_oversold:
                mfi_strength = 0.7
            elif mfi_val > self.strategy_config.mfi_extreme_overbought:
                mfi_strength = 1.0
            elif mfi_val > self.strategy_config.mfi_overbought:
                mfi_strength = 0.7
            else:
                mfi_strength = 0.0
            
            # Combine strengths
            strength.iloc[i] = (uo_strength * 0.4 + adx_strength * 0.3 + mfi_strength * 0.3)
        
        return strength
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on indicator alignment"""
        signals = []
        
        min_bars = max(self.strategy_config.uo_long_period, 
                      self.strategy_config.adx_period, 
                      self.strategy_config.mfi_period)
        
        if len(data) < min_bars:
            return signals
        
        # Calculate indicators first
        data_with_indicators = self.calculate_indicators(data)
        
        # Check if this is a single-timestamp call (incremental) or full dataset (batch)
        # If data has grown since last call, only process new bars
        if hasattr(self, '_last_data_length'):
            start_bar = max(min_bars, self._last_data_length)
        else:
            start_bar = min_bars
        
        self._last_data_length = len(data_with_indicators)
        
        # Process bars from start_bar to end
        for current_bar in range(start_bar, len(data_with_indicators)):
            current_data = data_with_indicators.iloc[current_bar]
            
            # Check minimum signal separation
            if (self.last_signal_bar >= 0 and 
                current_bar - self.last_signal_bar < self.strategy_config.min_signal_separation):
                continue
            
            # Check market conditions
            if not self._check_market_conditions(current_data):
                continue
            
            # Generate entry signals
            entry_signal = self._generate_entry_signal(data_with_indicators, current_bar)
            if entry_signal:
                signals.append(entry_signal)
                self.last_signal_bar = current_bar
            
            # Generate exit signals
            exit_signal = self._generate_exit_signal(data_with_indicators, current_bar)
            if exit_signal:
                signals.append(exit_signal)
        
        return signals
    
    def _check_market_conditions(self, bar: pd.Series) -> bool:
        """Check if market conditions are suitable for trading"""
        # ADX must be above minimum threshold
        if bar['adx'] < self.strategy_config.min_adx_for_entry:
            return False
        
        # Volume confirmation if enabled
        if (self.strategy_config.volume_confirmation and 
            'volume_ratio' in bar and bar['volume_ratio'] < 0.8):
            return False
        
        return True
    
    def _generate_entry_signal(self, data: pd.DataFrame, current_bar: int) -> Optional[Signal]:
        """Generate entry signal based on indicator alignment"""
        current_data = data.iloc[current_bar]
        
        # Check for confirmation over multiple bars if required
        if self.strategy_config.entry_confirmation_bars > 1:
            confirmation_start = max(0, current_bar - self.strategy_config.entry_confirmation_bars + 1)
            confirmation_data = data.iloc[confirmation_start:current_bar + 1]
            
            # All confirmation bars must have the same signal
            signals = confirmation_data['composite_signal']
            if not all(signals == signals.iloc[-1]) or signals.iloc[-1] == 0:
                return None
        
        signal_direction = current_data['composite_signal']
        if signal_direction == 0:
            return None
        
        # Generate appropriate signal
        if signal_direction > 0:
            return Signal(
                timestamp=current_data.name,
                symbol=current_data.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.LONG,
                strength=SignalStrength.STRONG if current_data['signal_strength'] > 0.7 else SignalStrength.MEDIUM,
                price=current_data['close'],
                confidence=self._calculate_signal_confidence(current_data, 'long'),
                metadata={
                    'strategy': 'ultimate_oscillator',
                    'entry_reason': 'bullish_alignment',
                    'ultimate_oscillator': current_data['ultimate_oscillator'],
                    'adx': current_data['adx'],
                    'mfi': current_data['mfi'],
                    'signal_strength': current_data['signal_strength'],
                    'plus_di': current_data['plus_di'],
                    'minus_di': current_data['minus_di']
                }
            )
        else:
            return Signal(
                timestamp=current_data.name,
                symbol=current_data.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.SHORT,
                strength=SignalStrength.STRONG if current_data['signal_strength'] > 0.7 else SignalStrength.MEDIUM,
                price=current_data['close'],
                confidence=self._calculate_signal_confidence(current_data, 'short'),
                metadata={
                    'strategy': 'ultimate_oscillator',
                    'entry_reason': 'bearish_alignment',
                    'ultimate_oscillator': current_data['ultimate_oscillator'],
                    'adx': current_data['adx'],
                    'mfi': current_data['mfi'],
                    'signal_strength': current_data['signal_strength'],
                    'plus_di': current_data['plus_di'],
                    'minus_di': current_data['minus_di']
                }
            )
    
    def _generate_exit_signal(self, data: pd.DataFrame, current_bar: int) -> Optional[Signal]:
        """Generate exit signal based on opposite extremes or weakening trend"""
        if self.position_entry_bar < 0:
            return None
        
        current_data = data.iloc[current_bar]
        
        # Exit on opposite extreme if enabled
        if self.strategy_config.exit_on_opposite_extreme:
            # For long positions, exit on overbought
            if (current_data['uo_extreme_overbought'] and 
                current_data['mfi_extreme_overbought']):
                return Signal(
                    timestamp=current_data.name,
                    symbol=current_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.EXIT_LONG,
                    strength=SignalStrength.MEDIUM,
                    price=current_data['close'],
                    confidence=0.8,
                    metadata={
                        'strategy': 'ultimate_oscillator',
                        'exit_reason': 'opposite_extreme',
                        'ultimate_oscillator': current_data['ultimate_oscillator'],
                        'mfi': current_data['mfi']
                    }
                )
            
            # For short positions, exit on oversold
            elif (current_data['uo_extreme_oversold'] and 
                  current_data['mfi_extreme_oversold']):
                return Signal(
                    timestamp=current_data.name,
                    symbol=current_data.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.EXIT_SHORT,
                    strength=SignalStrength.MEDIUM,
                    price=current_data['close'],
                    confidence=0.8,
                    metadata={
                        'strategy': 'ultimate_oscillator',
                        'exit_reason': 'opposite_extreme',
                        'ultimate_oscillator': current_data['ultimate_oscillator'],
                        'mfi': current_data['mfi']
                    }
                )
        
        # Exit on ADX weakening below threshold
        if current_data['adx'] < self.strategy_config.adx_trend_threshold:
            return Signal(
                timestamp=current_data.name,
                symbol=current_data.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_LONG,  # Will be adjusted based on actual position
                strength=SignalStrength.MEDIUM,
                price=current_data['close'],
                confidence=0.7,
                metadata={
                    'strategy': 'ultimate_oscillator',
                    'exit_reason': 'trend_weakening',
                    'adx': current_data['adx']
                }
            )
        
        return None
    
    def _calculate_signal_confidence(self, bar: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on indicator strength and alignment"""
        confidence = 0.5  # Base confidence
        
        # Ultimate Oscillator contribution
        uo_val = bar['ultimate_oscillator']
        if signal_type == 'long':
            if uo_val < self.strategy_config.uo_extreme_oversold:
                uo_confidence = 0.3
            elif uo_val < self.strategy_config.uo_oversold:
                uo_confidence = 0.2
            else:
                uo_confidence = 0.0
        else:
            if uo_val > self.strategy_config.uo_extreme_overbought:
                uo_confidence = 0.3
            elif uo_val > self.strategy_config.uo_overbought:
                uo_confidence = 0.2
            else:
                uo_confidence = 0.0
        
        confidence += uo_confidence
        
        # ADX contribution
        adx_val = bar['adx']
        if adx_val > self.strategy_config.adx_strong_trend:
            adx_confidence = 0.25
        elif adx_val > self.strategy_config.adx_trend_threshold:
            adx_confidence = 0.15
        else:
            adx_confidence = 0.0
        
        confidence += adx_confidence
        
        # MFI contribution
        mfi_val = bar['mfi']
        if signal_type == 'long':
            if mfi_val < self.strategy_config.mfi_extreme_oversold:
                mfi_confidence = 0.25
            elif mfi_val < self.strategy_config.mfi_oversold:
                mfi_confidence = 0.15
            else:
                mfi_confidence = 0.0
        else:
            if mfi_val > self.strategy_config.mfi_extreme_overbought:
                mfi_confidence = 0.25
            elif mfi_val > self.strategy_config.mfi_overbought:
                mfi_confidence = 0.15
            else:
                mfi_confidence = 0.0
        
        confidence += mfi_confidence
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                               portfolio_value: float) -> float:
        """Calculate position size based on confidence and volatility"""
        base_size = self.strategy_config.base_position_size
        
        # Scale by signal confidence
        if self.strategy_config.confidence_scaling:
            confidence_multiplier = signal.confidence
        else:
            confidence_multiplier = 1.0
        
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
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * volatility_multiplier
        
        # Apply limits
        position_size = min(position_size, self.strategy_config.max_position_size)
        position_size = max(position_size, 0.01)  # Minimum position size
        
        return position_size
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "Ultimate Oscillator + ADX + MFI"
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['ultimate_oscillator', 'adx', 'plus_di', 'minus_di', 'mfi', 
                'composite_signal', 'signal_strength', 'atr']
