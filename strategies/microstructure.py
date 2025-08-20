"""
Microstructure Order Flow Imbalance Strategy Implementation

This strategy analyzes order flow imbalances, book imbalance, and VWAP drift
to identify short-term trading opportunities based on market microstructure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, Trade
from utils.risk import RiskManager
from utils.data_utils import calculate_vwap


@dataclass
class MicrostructureConfig:
    """Configuration for Microstructure Strategy"""
    # Order Flow Parameters
    order_flow_window: int = 10
    imbalance_threshold: float = 0.6
    flow_strength_threshold: float = 0.7
    
    # Book Imbalance Parameters
    book_levels: int = 5
    book_imbalance_threshold: float = 0.3
    book_depth_threshold: float = 1000000  # Minimum depth in base currency
    
    # VWAP Drift Parameters
    vwap_period: int = 20
    vwap_drift_threshold: float = 0.001
    drift_confirmation_bars: int = 3
    
    # Volume Analysis
    volume_profile_period: int = 20
    volume_imbalance_threshold: float = 0.4
    large_trade_threshold: float = 0.1  # % of average volume
    
    # Market Impact
    impact_threshold: float = 0.0005
    impact_decay_period: int = 5
    
    # Entry/Exit Rules
    entry_delay: int = 0  # Immediate entry
    stop_loss_atr_multiplier: float = 1.5
    take_profit_ratio: float = 2.0
    max_hold_period: int = 24  # Short-term positions
    
    # Position Sizing
    base_position_size: float = 0.05  # Smaller positions for microstructure
    volatility_scaling: bool = True
    max_position_size: float = 0.15
    imbalance_scaling: bool = True
    
    # Performance Filters
    min_volatility: float = 0.005
    max_volatility: float = 0.05
    spread_threshold: float = 0.0002  # Maximum bid-ask spread
    
    # Risk Management
    correlation_threshold: float = 0.8
    max_correlation_exposure: float = 0.3


class MicrostructureStrategy(BaseStrategy):
    """
    Microstructure Strategy using Order Flow Imbalance and Book Analysis
    
    This strategy analyzes market microstructure to identify short-term
    trading opportunities based on order flow patterns.
    """
    
    def __init__(self, config: Dict, risk_manager: RiskManager):
        super().__init__(config, risk_manager)
        self.strategy_config = MicrostructureConfig(**config.get('strategy', {}))
        self.order_flow_history = []
        self.book_imbalance_history = []
        self.vwap_drift_history = []
        self.volume_profile = []
        self.market_impact_history = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for microstructure strategy"""
        df = data.copy()
        
        # Calculate VWAP
        df['vwap'] = calculate_vwap(df, period=self.strategy_config.vwap_period)
        
        # Calculate Order Flow Imbalance
        df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        df['flow_strength'] = self._calculate_flow_strength(df)
        
        # Calculate Book Imbalance (simulated from OHLCV data)
        df['book_imbalance'] = self._calculate_book_imbalance(df)
        df['book_depth'] = self._calculate_book_depth(df)
        
        # Calculate VWAP Drift
        df['vwap_drift'] = self._calculate_vwap_drift(df)
        df['drift_direction'] = np.where(df['vwap_drift'] > 0, 1, -1)
        
        # Calculate Volume Profile
        df['volume_profile'] = self._calculate_volume_profile(df)
        df['volume_imbalance'] = self._calculate_volume_imbalance(df)
        df['large_trades'] = self._detect_large_trades(df)
        
        # Calculate Market Impact
        df['market_impact'] = self._calculate_market_impact(df)
        df['impact_decay'] = self._calculate_impact_decay(df)
        
        # Calculate Microstructure Signals
        df['microstructure_signal'] = self._calculate_microstructure_signal(df)
        df['signal_strength'] = self._calculate_signal_strength(df)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate microstructure trading signals"""
        signals = []
        
        if len(data) < max(self.strategy_config.order_flow_window, self.strategy_config.vwap_period):
            return signals
        
        current_bar = data.iloc[-1]
        prev_bars = data.iloc[-self.strategy_config.drift_confirmation_bars-1:-1] if len(data) > self.strategy_config.drift_confirmation_bars else None
        
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
        """Check if market conditions are suitable for microstructure trading"""
        # Check volatility bounds
        if (bar.get('volatility', 0) < self.strategy_config.min_volatility or 
            bar.get('volatility', 0) > self.strategy_config.max_volatility):
            return False
        
        # Check book depth
        if bar.get('book_depth', 0) < self.strategy_config.book_depth_threshold:
            return False
        
        # Check spread (if available)
        if 'spread' in bar and bar['spread'] > self.strategy_config.spread_threshold:
            return False
        
        return True
    
    def _generate_entry_signal(self, current_bar: pd.Series, prev_bars: pd.DataFrame) -> Optional[Signal]:
        """Generate entry signal based on microstructure logic"""
        if prev_bars is None or len(prev_bars) < self.strategy_config.drift_confirmation_bars:
            return None
        
        # Check for bullish microstructure signal
        if (current_bar['microstructure_signal'] > 0 and
            current_bar['signal_strength'] > self.strategy_config.flow_strength_threshold and
            current_bar['order_flow_imbalance'] > self.strategy_config.imbalance_threshold and
            current_bar['vwap_drift'] > self.strategy_config.vwap_drift_threshold):
            
            # Confirm drift over multiple bars
            if self._confirm_drift(prev_bars, 'bullish'):
                return Signal(
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.LONG,
                    strength=SignalStrength.STRONG,
                    price=current_bar['close'],
                    confidence=self._calculate_signal_confidence(current_bar, 'bullish'),
                    metadata={
                        'strategy': 'microstructure',
                        'entry_reason': 'bullish_microstructure',
                        'order_flow_imbalance': current_bar['order_flow_imbalance'],
                        'vwap_drift': current_bar['vwap_drift'],
                        'signal_strength': current_bar['signal_strength'],
                        'book_imbalance': current_bar.get('book_imbalance', 0)
                    }
                )
        
        # Check for bearish microstructure signal
        elif (current_bar['microstructure_signal'] < 0 and
              current_bar['signal_strength'] > self.strategy_config.flow_strength_threshold and
              current_bar['order_flow_imbalance'] < -self.strategy_config.imbalance_threshold and
              current_bar['vwap_drift'] < -self.strategy_config.vwap_drift_threshold):
            
            # Confirm drift over multiple bars
            if self._confirm_drift(prev_bars, 'bearish'):
                return Signal(
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SHORT,
                    strength=SignalStrength.STRONG,
                    price=current_bar['close'],
                    confidence=self._calculate_signal_confidence(current_bar, 'bearish'),
                    metadata={
                        'strategy': 'microstructure',
                        'entry_reason': 'bearish_microstructure',
                        'order_flow_imbalance': current_bar['order_flow_imbalance'],
                        'vwap_drift': current_bar['vwap_drift'],
                        'signal_strength': current_bar['signal_strength'],
                        'book_imbalance': current_bar.get('book_imbalance', 0)
                    }
                )
        
        return None
    
    def _generate_exit_signal(self, current_bar: pd.Series, prev_bars: pd.DataFrame) -> Optional[Signal]:
        """Generate exit signal based on microstructure logic"""
        # Exit long position on bearish microstructure reversal
        if (current_bar['microstructure_signal'] < 0 and
            current_bar['vwap_drift'] < -self.strategy_config.vwap_drift_threshold):
            
            return Signal(
                timestamp=current_bar.name,
                symbol=current_bar.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_LONG,
                strength=SignalStrength.MEDIUM,
                price=current_bar['close'],
                confidence=0.8,
                metadata={
                    'strategy': 'microstructure',
                    'exit_reason': 'bearish_reversal',
                    'vwap_drift': current_bar['vwap_drift']
                }
            )
        
        # Exit short position on bullish microstructure reversal
        if (current_bar['microstructure_signal'] > 0 and
            current_bar['vwap_drift'] > self.strategy_config.vwap_drift_threshold):
            
            return Signal(
                timestamp=current_bar.name,
                symbol=current_bar.get('symbol', 'UNKNOWN'),
                signal_type=SignalType.EXIT_SHORT,
                strength=SignalStrength.MEDIUM,
                price=current_bar['close'],
                confidence=0.8,
                metadata={
                    'strategy': 'microstructure',
                    'exit_reason': 'bullish_reversal',
                    'vwap_drift': current_bar['vwap_drift']
                }
            )
        
        return None
    
    def _confirm_drift(self, prev_bars: pd.DataFrame, direction: str) -> bool:
        """Confirm VWAP drift over multiple bars"""
        if direction == 'bullish':
            return all(bar['vwap_drift'] > 0 for _, bar in prev_bars.iterrows())
        else:
            return all(bar['vwap_drift'] < 0 for _, bar in prev_bars.iterrows())
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance based on volume and price action"""
        imbalance = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.order_flow_window:
                imbalance.iloc[i] = 0.0
                continue
            
            # Calculate imbalance based on recent price and volume action
            recent_data = df.iloc[i-self.strategy_config.order_flow_window:i+1]
            
            # Price momentum component
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Volume-weighted component
            volume_weighted_price = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
            volume_component = (recent_data['close'].iloc[-1] - volume_weighted_price) / volume_weighted_price
            
            # Combine components
            imbalance.iloc[i] = (price_change * 0.6 + volume_component * 0.4)
        
        return imbalance
    
    def _calculate_flow_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the strength of order flow signals"""
        strength = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.order_flow_window:
                strength.iloc[i] = 0.0
                continue
            
            recent_data = df.iloc[i-self.strategy_config.order_flow_window:i+1]
            
            # Volume consistency
            volume_std = recent_data['volume'].std() / recent_data['volume'].mean()
            volume_strength = 1.0 / (1.0 + volume_std)
            
            # Price consistency
            price_std = recent_data['close'].std() / recent_data['close'].mean()
            price_strength = 1.0 / (1.0 + price_std)
            
            # Combined strength
            strength.iloc[i] = (volume_strength * 0.6 + price_strength * 0.4)
        
        return strength
    
    def _calculate_book_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate book imbalance (simulated from OHLCV data)"""
        # In a real implementation, this would use actual order book data
        # Here we simulate it using price action and volume
        imbalance = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < 1:
                imbalance.iloc[i] = 0.0
                continue
            
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Simulate book imbalance based on price action and volume
            price_momentum = (current_bar['close'] - prev_bar['close']) / prev_bar['close']
            volume_factor = current_bar['volume'] / prev_bar['volume'] if prev_bar['volume'] > 0 else 1.0
            
            # Combine factors to simulate book imbalance
            imbalance.iloc[i] = price_momentum * volume_factor
        
        return imbalance
    
    def _calculate_book_depth(self, df: pd.DataFrame) -> pd.Series:
        """Calculate book depth (simulated)"""
        # In a real implementation, this would use actual order book data
        # Here we simulate it using volume and volatility
        depth = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.order_flow_window:
                depth.iloc[i] = 1000000.0  # Default depth
                continue
            
            recent_data = df.iloc[i-self.strategy_config.order_flow_window:i+1]
            
            # Simulate depth based on average volume and volatility
            avg_volume = recent_data['volume'].mean()
            volatility = recent_data['close'].std() / recent_data['close'].mean()
            
            # Higher volume and lower volatility suggest deeper books
            depth.iloc[i] = avg_volume * (1.0 - volatility) * 1000
        
        return depth
    
    def _calculate_vwap_drift(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP drift from the moving average"""
        drift = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.vwap_period:
                drift.iloc[i] = 0.0
                continue
            
            current_price = df.iloc[i]['close']
            current_vwap = df.iloc[i]['vwap']
            
            if current_vwap > 0:
                drift.iloc[i] = (current_price - current_vwap) / current_vwap
            else:
                drift.iloc[i] = 0.0
        
        return drift
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile indicator"""
        profile = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.volume_profile_period:
                profile.iloc[i] = 0.0
                continue
            
            recent_data = df.iloc[i-self.strategy_config.volume_profile_period:i+1]
            
            # Calculate volume-weighted average price
            vwap = (recent_data['close'] * recent_data['volume']).sum() / recent_data['volume'].sum()
            current_price = df.iloc[i]['close']
            
            if vwap > 0:
                profile.iloc[i] = (current_price - vwap) / vwap
            else:
                profile.iloc[i] = 0.0
        
        return profile
    
    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume imbalance indicator"""
        imbalance = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.volume_profile_period:
                imbalance.iloc[i] = 0.0
                continue
            
            recent_data = df.iloc[i-self.strategy_config.volume_profile_period:i+1]
            
            # Calculate volume imbalance based on price action
            up_volume = recent_data[recent_data['close'] > recent_data['close'].shift(1)]['volume'].sum()
            down_volume = recent_data[recent_data['close'] < recent_data['close'].shift(1)]['volume'].sum()
            total_volume = recent_data['volume'].sum()
            
            if total_volume > 0:
                imbalance.iloc[i] = (up_volume - down_volume) / total_volume
            else:
                imbalance.iloc[i] = 0.0
        
        return imbalance
    
    def _detect_large_trades(self, df: pd.DataFrame) -> pd.Series:
        """Detect large trades based on volume threshold"""
        large_trades = pd.Series(index=df.index, dtype=bool)
        
        for i in range(len(df)):
            if i < self.strategy_config.volume_profile_period:
                large_trades.iloc[i] = False
                continue
            
            recent_data = df.iloc[i-self.strategy_config.volume_profile_period:i+1]
            avg_volume = recent_data['volume'].mean()
            current_volume = df.iloc[i]['volume']
            
            large_trades.iloc[i] = current_volume > (avg_volume * self.strategy_config.large_trade_threshold)
        
        return large_trades
    
    def _calculate_market_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market impact of trades"""
        impact = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < 1:
                impact.iloc[i] = 0.0
                continue
            
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Calculate impact based on volume and price change
            volume_factor = current_bar['volume'] / prev_bar['volume'] if prev_bar['volume'] > 0 else 1.0
            price_change = abs(current_bar['close'] - prev_bar['close']) / prev_bar['close']
            
            impact.iloc[i] = price_change * volume_factor
        
        return impact
    
    def _calculate_impact_decay(self, df: pd.DataFrame) -> pd.Series:
        """Calculate impact decay over time"""
        decay = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.impact_decay_period:
                decay.iloc[i] = 1.0
                continue
            
            # Calculate decay based on recent impact
            recent_impact = df.iloc[i-self.strategy_config.impact_decay_period:i+1]['market_impact']
            decay.iloc[i] = recent_impact.iloc[-1] / recent_impact.iloc[0] if recent_impact.iloc[0] > 0 else 1.0
        
        return decay
    
    def _calculate_microstructure_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite microstructure signal"""
        signal = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.order_flow_window:
                signal.iloc[i] = 0.0
                continue
            
            current_bar = df.iloc[i]
            
            # Combine multiple factors
            flow_component = current_bar['order_flow_imbalance'] * 0.4
            book_component = current_bar.get('book_imbalance', 0) * 0.3
            drift_component = current_bar['vwap_drift'] * 0.3
            
            signal.iloc[i] = flow_component + book_component + drift_component
        
        return signal
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the strength of microstructure signals"""
        strength = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < self.strategy_config.order_flow_window:
                strength.iloc[i] = 0.0
                continue
            
            current_bar = df.iloc[i]
            
            # Calculate strength based on multiple factors
            flow_strength = current_bar['flow_strength']
            volume_strength = min(current_bar.get('volume_imbalance', 0) / 0.5, 1.0)
            impact_strength = min(current_bar.get('market_impact', 0) / self.strategy_config.impact_threshold, 1.0)
            
            # Combined strength
            strength.iloc[i] = (flow_strength * 0.5 + volume_strength * 0.3 + impact_strength * 0.2)
        
        return strength
    
    def _calculate_signal_confidence(self, bar: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Signal strength contribution
        strength_confidence = bar['signal_strength'] * 0.3
        confidence += strength_confidence
        
        # Order flow imbalance contribution
        flow_confidence = min(abs(bar['order_flow_imbalance']) / 0.8, 1.0) * 0.25
        confidence += flow_confidence
        
        # VWAP drift contribution
        drift_confidence = min(abs(bar['vwap_drift']) / 0.002, 1.0) * 0.25
        confidence += drift_confidence
        
        # Book imbalance contribution
        book_confidence = min(abs(bar.get('book_imbalance', 0)) / 0.5, 1.0) * 0.2
        confidence += book_confidence
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                               portfolio_value: float) -> float:
        """Calculate position size based on volatility and imbalance strength"""
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
        
        # Scale by imbalance strength if enabled
        if self.strategy_config.imbalance_scaling:
            imbalance_strength = abs(signal.metadata.get('order_flow_imbalance', 0.5))
            imbalance_multiplier = 0.5 + imbalance_strength * 0.5
        else:
            imbalance_multiplier = 1.0
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * volatility_multiplier * imbalance_multiplier
        
        # Apply limits
        position_size = min(position_size, self.strategy_config.max_position_size)
        position_size = max(position_size, 0.01)  # Minimum position size
        
        return position_size
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return "Microstructure (Order Flow + Book Imbalance + VWAP Drift)"
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        return ['order_flow_imbalance', 'flow_strength', 'book_imbalance', 'vwap_drift',
                'volume_profile', 'volume_imbalance', 'market_impact', 'microstructure_signal']
