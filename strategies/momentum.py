"""
Momentum trading strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength
from utils.risk import PositionSide


class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy using moving average crossovers and volatility scaling."""
    
    def _initialize_strategy(self) -> None:
        """Initialize momentum strategy components."""
        self.logger.info("Initializing Momentum Strategy")
        
        # Strategy-specific state
        self.last_signals: Dict[str, Signal] = {}
        self.signal_history: List[Signal] = []
        
        # Performance tracking
        self.entry_prices: Dict[str, float] = {}
        self.position_timestamps: Dict[str, datetime] = {}
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate momentum trading signals.
        
        Args:
            data: Market data with indicators
            
        Returns:
            List of momentum trading signals
        """
        if data.empty:
            return []
        
        signals = []
        signals_config = self.config.get('signals', {})
        
        # Get latest data point
        latest_data = data.tail(1).iloc[0]
        
        # Check if we have required indicators
        required_indicators = ['sma_20', 'sma_50', 'rsi', 'volume_ratio', 'volatility']
        if not all(indicator in latest_data.index for indicator in required_indicators):
            self.logger.warning("Missing required indicators for signal generation")
            return []
        
        # Generate signals for each symbol
        symbols = data['symbol'].unique() if 'symbol' in data.columns else ['UNKNOWN']
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data
            
            if symbol_data.empty:
                continue
            
            # Generate signal for this symbol
            signal = self._generate_symbol_signal(symbol_data, symbol, signals_config)
            if signal:
                signals.append(signal)
        
        # Store signals
        self.signals.extend(signals)
        
        return signals
    
    def _generate_symbol_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        signals_config: Dict
    ) -> Optional[Signal]:
        """Generate signal for a specific symbol.
        
        Args:
            data: Symbol-specific market data
            symbol: Trading symbol
            signals_config: Signals configuration
            
        Returns:
            Trading signal or None
        """
        if len(data) < 50:  # Need enough data for indicators
            return None
        
        # Get latest data
        latest = data.tail(1).iloc[0]
        previous = data.tail(2).iloc[0] if len(data) >= 2 else latest
        
        # Check for MA crossover
        ma_signal = self._check_ma_crossover(data, signals_config)
        
        # Check for RSI signals
        rsi_signal = self._check_rsi_signal(latest, signals_config)
        
        # Check for MACD signals
        macd_signal = self._check_macd_signal(data, signals_config)
        
        # Check for volume confirmation
        volume_confirmed = self._check_volume_confirmation(latest, signals_config)
        
        # Check for volatility filter
        volatility_ok = self._check_volatility_filter(latest, signals_config)
        
        # Combine signals
        signal_type = self._combine_signals(ma_signal, rsi_signal, macd_signal, volume_confirmed, volatility_ok)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate signal strength and confidence
        strength, confidence = self._calculate_signal_strength(
            data, ma_signal, rsi_signal, macd_signal, volume_confirmed, volatility_ok
        )
        
        # Create signal
        signal = Signal(
            timestamp=latest['timestamp'],
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=latest['close'],
            confidence=confidence,
            metadata={
                'ma_signal': ma_signal.value if ma_signal else None,
                'rsi_signal': rsi_signal.value if rsi_signal else None,
                'macd_signal': macd_signal.value if macd_signal else None,
                'volume_confirmed': volume_confirmed,
                'volatility_ok': volatility_ok,
                'rsi': latest.get('rsi', None),
                'volume_ratio': latest.get('volume_ratio', None),
                'volatility': latest.get('volatility', None)
            }
        )
        
        # Check if this is a new signal (avoid duplicate signals)
        if self._is_new_signal(signal):
            self.last_signals[symbol] = signal
            return signal
        
        return None
    
    def _check_ma_crossover(self, data: pd.DataFrame, signals_config: Dict) -> Optional[SignalType]:
        """Check for moving average crossover signals.
        
        Args:
            data: Market data
            signals_config: Signals configuration
            
        Returns:
            Signal type or None
        """
        if len(data) < 2:
            return None
        
        short_ma = signals_config.get('short_ma', 20)
        long_ma = signals_config.get('long_ma', 50)
        
        short_col = f'sma_{short_ma}'
        long_col = f'sma_{long_ma}'
        
        if short_col not in data.columns or long_col not in data.columns:
            return None
        
        # Get current and previous values
        current = data.tail(1).iloc[0]
        previous = data.tail(2).iloc[0]
        
        # Check for crossover
        current_short = current[short_col]
        current_long = current[long_col]
        previous_short = previous[short_col]
        previous_long = previous[long_col]
        
        # Bullish crossover: short MA crosses above long MA
        if (previous_short <= previous_long) and (current_short > current_long):
            return SignalType.BUY
        
        # Bearish crossover: short MA crosses below long MA
        elif (previous_short >= previous_long) and (current_short < current_long):
            return SignalType.SELL
        
        return None
    
    def _check_rsi_signal(self, latest: pd.Series, signals_config: Dict) -> Optional[SignalType]:
        """Check for RSI-based signals.
        
        Args:
            latest: Latest market data
            signals_config: Signals configuration
            
        Returns:
            Signal type or None
        """
        if 'rsi' not in latest.index:
            return None
        
        rsi = latest['rsi']
        rsi_overbought = signals_config.get('rsi_overbought', 70)
        rsi_oversold = signals_config.get('rsi_oversold', 30)
        
        if pd.isna(rsi):
            return None
        
        # Oversold condition (potential buy)
        if rsi < rsi_oversold:
            return SignalType.BUY
        
        # Overbought condition (potential sell)
        elif rsi > rsi_overbought:
            return SignalType.SELL
        
        return None
    
    def _check_macd_signal(self, data: pd.DataFrame, signals_config: Dict) -> Optional[SignalType]:
        """Check for MACD-based signals.
        
        Args:
            data: Market data
            signals_config: Signals configuration
            
        Returns:
            Signal type or None
        """
        if len(data) < 2:
            return None
        
        # Check if MACD columns exist
        macd_cols = ['macd', 'macd_signal', 'macd_histogram']
        if not all(col in data.columns for col in macd_cols):
            return None
        
        current = data.tail(1).iloc[0]
        previous = data.tail(2).iloc[0]
        
        # Check for MACD crossover
        current_macd = current['macd']
        current_signal = current['macd_signal']
        previous_macd = previous['macd']
        previous_signal = previous['macd_signal']
        
        # Bullish crossover: MACD crosses above signal line
        if (previous_macd <= previous_signal) and (current_macd > current_signal):
            return SignalType.BUY
        
        # Bearish crossover: MACD crosses below signal line
        elif (previous_macd >= previous_signal) and (current_macd < current_signal):
            return SignalType.SELL
        
        return None
    
    def _check_volume_confirmation(self, latest: pd.Series, signals_config: Dict) -> bool:
        """Check for volume confirmation.
        
        Args:
            latest: Latest market data
            signals_config: Signals configuration
            
        Returns:
            True if volume confirms the signal
        """
        if 'volume_ratio' not in latest.index:
            return True  # Assume confirmed if no volume data
        
        volume_ratio = latest['volume_ratio']
        volume_threshold = signals_config.get('volume_threshold', 1.5)
        
        if pd.isna(volume_ratio):
            return True
        
        return volume_ratio >= volume_threshold
    
    def _check_volatility_filter(self, latest: pd.Series, signals_config: Dict) -> bool:
        """Check volatility filter.
        
        Args:
            latest: Latest market data
            signals_config: Signals configuration
            
        Returns:
            True if volatility is within acceptable range
        """
        if 'volatility' not in latest.index:
            return True  # Assume OK if no volatility data
        
        volatility = latest['volatility']
        min_vol = signals_config.get('min_volatility', 0.01)
        max_vol = signals_config.get('max_volatility', 0.05)
        
        if pd.isna(volatility):
            return True
        
        return min_vol <= volatility <= max_vol
    
    def _combine_signals(
        self,
        ma_signal: Optional[SignalType],
        rsi_signal: Optional[SignalType],
        macd_signal: Optional[SignalType],
        volume_confirmed: bool,
        volatility_ok: bool
    ) -> SignalType:
        """Combine multiple signals into a final signal.
        
        Args:
            ma_signal: Moving average signal
            rsi_signal: RSI signal
            macd_signal: MACD signal
            volume_confirmed: Volume confirmation
            volatility_ok: Volatility filter result
            
        Returns:
            Combined signal type
        """
        # If volatility is not OK, no signal
        if not volatility_ok:
            return SignalType.HOLD
        
        # If volume is not confirmed, no signal
        if not volume_confirmed:
            return SignalType.HOLD
        
        # Count bullish and bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        for signal in [ma_signal, rsi_signal, macd_signal]:
            if signal == SignalType.BUY:
                bullish_signals += 1
            elif signal == SignalType.SELL:
                bearish_signals += 1
        
        # Generate final signal based on majority
        if bullish_signals > bearish_signals and bullish_signals >= 1:
            return SignalType.BUY
        elif bearish_signals > bullish_signals and bearish_signals >= 1:
            return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(
        self,
        data: pd.DataFrame,
        ma_signal: Optional[SignalType],
        rsi_signal: Optional[SignalType],
        macd_signal: Optional[SignalType],
        volume_confirmed: bool,
        volatility_ok: bool
    ) -> Tuple[SignalStrength, float]:
        """Calculate signal strength and confidence.
        
        Args:
            data: Market data
            ma_signal: Moving average signal
            rsi_signal: RSI signal
            macd_signal: MACD signal
            volume_confirmed: Volume confirmation
            volatility_ok: Volatility filter result
            
        Returns:
            Tuple of (signal_strength, confidence)
        """
        # Count confirming signals
        confirming_signals = 0
        total_signals = 0
        
        for signal in [ma_signal, rsi_signal, macd_signal]:
            if signal is not None:
                total_signals += 1
                if signal != SignalType.HOLD:
                    confirming_signals += 1
        
        # Add volume and volatility confirmation
        if volume_confirmed:
            confirming_signals += 1
        total_signals += 1
        
        if volatility_ok:
            confirming_signals += 1
        total_signals += 1
        
        # Calculate confidence
        confidence = confirming_signals / total_signals if total_signals > 0 else 0
        
        # Determine signal strength
        if confidence >= 0.8:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK
        
        return strength, confidence
    
    def _is_new_signal(self, signal: Signal) -> bool:
        """Check if this is a new signal (not a duplicate).
        
        Args:
            signal: Trading signal
            
        Returns:
            True if this is a new signal
        """
        if signal.symbol not in self.last_signals:
            return True
        
        last_signal = self.last_signals[signal.symbol]
        
        # Check if signal type changed
        if signal.signal_type != last_signal.signal_type:
            return True
        
        # Check if enough time has passed (avoid rapid signal changes)
        time_diff = signal.timestamp - last_signal.timestamp
        min_interval = pd.Timedelta(hours=1)  # Minimum 1 hour between signals
        
        if time_diff >= min_interval:
            return True
        
        return False
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float
    ) -> float:
        """Calculate position size for momentum strategy.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price
            
        Returns:
            Position size in base currency
        """
        position_config = self.config.get('position_sizing', {})
        method = position_config.get('method', 'volatility_scaled')
        
        if method == 'volatility_scaled':
            return self._calculate_volatility_scaled_size(
                signal, portfolio_value, current_price, position_config
            )
        elif method == 'fixed':
            return self._calculate_fixed_size(signal, portfolio_value, current_price, position_config)
        elif method == 'kelly':
            return self._calculate_kelly_size(signal, portfolio_value, current_price, position_config)
        else:
            # Default to fixed size
            return self._calculate_fixed_size(signal, portfolio_value, current_price, position_config)
    
    def _calculate_volatility_scaled_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        position_config: Dict
    ) -> float:
        """Calculate volatility-scaled position size.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price
            position_config: Position sizing configuration
            
        Returns:
            Position size in base currency
        """
        base_size = position_config.get('base_size', 0.02)
        vol_config = position_config.get('volatility_scaling', {})
        
        # Get volatility from signal metadata
        volatility = signal.metadata.get('volatility', 0.02)
        target_volatility = vol_config.get('target_volatility', 0.02)
        
        # Calculate volatility adjustment
        vol_adjustment = target_volatility / (volatility + 1e-8)  # Avoid division by zero
        
        # Apply bounds
        max_multiplier = vol_config.get('max_size_multiplier', 3.0)
        min_multiplier = vol_config.get('min_size_multiplier', 0.5)
        
        vol_adjustment = np.clip(vol_adjustment, min_multiplier, max_multiplier)
        
        # Apply confidence adjustment
        confidence_adjustment = signal.confidence * 0.5 + 0.5  # 0.5 to 1.0
        
        # Calculate final position size
        position_value = portfolio_value * base_size * vol_adjustment * confidence_adjustment
        
        # Convert to quantity
        quantity = position_value / current_price
        
        return quantity
    
    def _calculate_fixed_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        position_config: Dict
    ) -> float:
        """Calculate fixed position size.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price
            position_config: Position sizing configuration
            
        Returns:
            Position size in base currency
        """
        base_size = position_config.get('base_size', 0.02)
        position_value = portfolio_value * base_size
        
        # Apply confidence adjustment
        confidence_adjustment = signal.confidence * 0.5 + 0.5
        position_value *= confidence_adjustment
        
        # Convert to quantity
        quantity = position_value / current_price
        
        return quantity
    
    def _calculate_kelly_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        position_config: Dict
    ) -> float:
        """Calculate Kelly criterion position size.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price
            position_config: Position sizing configuration
            
        Returns:
            Position size in base currency
        """
        kelly_config = position_config.get('kelly', {})
        
        if not kelly_config.get('enabled', False):
            return self._calculate_fixed_size(signal, portfolio_value, current_price, position_config)
        
        win_rate = kelly_config.get('win_rate', 0.55)
        avg_win = kelly_config.get('avg_win', 0.03)
        avg_loss = kelly_config.get('avg_loss', 0.02)
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss == 0:
            return self._calculate_fixed_size(signal, portfolio_value, current_price, position_config)
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply bounds (0 to 0.25 for safety)
        kelly_fraction = np.clip(kelly_fraction, 0, 0.25)
        
        # Calculate position size
        position_value = portfolio_value * kelly_fraction
        
        # Apply confidence adjustment
        confidence_adjustment = signal.confidence * 0.5 + 0.5
        position_value *= confidence_adjustment
        
        # Convert to quantity
        quantity = position_value / current_price
        
        return quantity
    
    def should_exit_position(
        self,
        symbol: str,
        current_data: pd.DataFrame,
        entry_price: float,
        current_price: float
    ) -> Tuple[bool, str]:
        """Check if position should be exited.
        
        Args:
            symbol: Trading symbol
            current_data: Current market data
            entry_price: Position entry price
            current_price: Current market price
            
        Returns:
            Tuple of (should_exit, reason)
        """
        rules_config = self.config.get('rules', {}).get('exit', {})
        
        # Check stop loss
        if rules_config.get('stop_loss', False):
            stop_loss_type = rules_config.get('stop_loss_type', 'percentage')
            
            if stop_loss_type == 'percentage':
                stop_loss_pct = rules_config.get('stop_loss_multiplier', 0.05)
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                
                if current_price <= stop_loss_price:
                    return True, f"Stop loss triggered at {stop_loss_price:.4f}"
            
            elif stop_loss_type == 'atr':
                if 'atr' in current_data.columns:
                    atr = current_data['atr'].iloc[-1]
                    multiplier = rules_config.get('stop_loss_multiplier', 2.0)
                    stop_loss_price = entry_price - (atr * multiplier)
                    
                    if current_price <= stop_loss_price:
                        return True, f"ATR stop loss triggered at {stop_loss_price:.4f}"
        
        # Check take profit
        if rules_config.get('take_profit', False):
            take_profit_type = rules_config.get('take_profit_type', 'risk_reward')
            
            if take_profit_type == 'risk_reward':
                risk_reward_ratio = rules_config.get('risk_reward_ratio', 3.0)
                # Calculate based on entry price and stop loss
                stop_loss_pct = rules_config.get('stop_loss_multiplier', 0.05)
                risk_amount = entry_price * stop_loss_pct
                reward_amount = risk_amount * risk_reward_ratio
                take_profit_price = entry_price + reward_amount
                
                if current_price >= take_profit_price:
                    return True, f"Take profit triggered at {take_profit_price:.4f}"
        
        # Check time-based exit
        if rules_config.get('time_exit', False):
            max_hold_periods = rules_config.get('max_hold_periods', 50)
            if symbol in self.position_timestamps:
                entry_time = self.position_timestamps[symbol]
                current_time = current_data['timestamp'].iloc[-1]
                time_diff = current_time - entry_time
                
                # Convert to periods (assuming hourly data)
                periods_held = time_diff.total_seconds() / 3600
                
                if periods_held >= max_hold_periods:
                    return True, f"Time-based exit after {periods_held:.1f} periods"
        
        return False, ""
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get momentum strategy specific metrics.
        
        Returns:
            Strategy metrics dictionary
        """
        metrics = super().get_strategy_summary()
        
        # Add momentum-specific metrics
        if self.signals:
            recent_signals = self.signals[-100:]  # Last 100 signals
            
            # Signal distribution
            signal_types = [s.signal_type.value for s in recent_signals]
            signal_distribution = pd.Series(signal_types).value_counts().to_dict()
            
            # Signal strength distribution
            signal_strengths = [s.strength.value for s in recent_signals]
            strength_distribution = pd.Series(signal_strengths).value_counts().to_dict()
            
            # Average confidence
            avg_confidence = np.mean([s.confidence for s in recent_signals])
            
            metrics.update({
                'signal_distribution': signal_distribution,
                'strength_distribution': strength_distribution,
                'avg_confidence': avg_confidence,
                'total_signals_generated': len(self.signals)
            })
        
        return metrics
