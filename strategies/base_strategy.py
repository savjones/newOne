"""
Base strategy class for all trading strategies.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.logging import TradingLogger
from utils.risk import RiskManager, PositionSide
from utils.config import ConfigManager


class SignalType(Enum):
    """Signal types enumeration."""
    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    PAIRS_LONG_SHORT = "pairs_long_short"
    PAIRS_SHORT_LONG = "pairs_short_long"
    EXIT_PAIRS = "exit_pairs"
    # Legacy support
    BUY = "long"
    SELL = "short"
    CLOSE = "exit_long"


class SignalStrength(Enum):
    """Signal strength enumeration."""
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"


@dataclass
class Signal:
    """Trading signal information."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    price: float
    confidence: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate signal data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class Trade:
    """Trade information."""
    timestamp: datetime
    symbol: str
    side: PositionSide
    quantity: float
    price: float
    signal: Signal
    metadata: Dict[str, Any]


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(
        self,
        strategy_name: str,
        config: Optional[Dict] = None,
        risk_manager: Optional[RiskManager] = None,
        logger: Optional[TradingLogger] = None
    ):
        """Initialize base strategy.
        
        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration
            risk_manager: Risk management instance
            logger: Trading logger instance
        """
        self.strategy_name = strategy_name
        self.config = config or {}
        self.risk_manager = risk_manager or RiskManager()
        
        # Initialize logger
        if logger:
            self.logger = logger
        else:
            self.logger = TradingLogger(strategy_name)
        
        # Strategy state
        self.positions: Dict[str, Dict] = {}
        self.signals: List[Signal] = []
        self.trades: List[Trade] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Configuration validation
        self._validate_config()
        
        # Initialize strategy-specific components
        self._initialize_strategy()
    
    @abstractmethod
    def _initialize_strategy(self) -> None:
        """Initialize strategy-specific components."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals from market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float
    ) -> float:
        """Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current market price
            
        Returns:
            Position size in base currency
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate strategy configuration."""
        required_keys = ['strategy', 'signals', 'rules', 'risk']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate signal parameters
        signals_config = self.config.get('signals', {})
        if not signals_config:
            raise ValueError("Signals configuration is required")
        
        # Validate risk parameters
        risk_config = self.config.get('risk', {})
        if not risk_config:
            raise ValueError("Risk configuration is required")
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and prepare data for signal generation.
        
        Args:
            data: Raw market data
            
        Returns:
            Processed data with indicators
        """
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate technical indicators
        data = self._calculate_indicators(data)
        
        # Add market regime information
        data = self._add_market_regime(data)
        
        return data
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data.
        
        Args:
            data: Market data
            
        Returns:
            Data with calculated indicators
        """
        df = data.copy()
        
        # Calculate basic indicators based on configuration
        signals_config = self.config.get('signals', {})
        
        # Moving averages
        if 'short_ma' in signals_config:
            short_ma = signals_config['short_ma']
            df[f'sma_{short_ma}'] = df['close'].rolling(window=short_ma).mean()
        
        if 'long_ma' in signals_config:
            long_ma = signals_config['long_ma']
            df[f'sma_{long_ma}'] = df['close'].rolling(window=long_ma).mean()
        
        # EMAs
        if 'ema_short' in signals_config:
            ema_short = signals_config['ema_short']
            df[f'ema_{ema_short}'] = df['close'].ewm(span=ema_short).mean()
        
        if 'ema_long' in signals_config:
            ema_long = signals_config['ema_long']
            df[f'ema_{ema_long}'] = df['close'].ewm(span=ema_long).mean()
        
        # RSI
        if 'rsi_period' in signals_config:
            rsi_period = signals_config['rsi_period']
            df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # MACD
        if all(key in signals_config for key in ['macd_fast', 'macd_slow', 'macd_signal']):
            macd_fast = signals_config['macd_fast']
            macd_slow = signals_config['macd_slow']
            macd_signal = signals_config['macd_signal']
            
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(
                df['close'], macd_fast, macd_slow, macd_signal
            )
        
        # ATR
        if 'atr_period' in signals_config:
            atr_period = signals_config['atr_period']
            df['atr'] = self._calculate_atr(df, atr_period)
        
        # Volume indicators
        if 'volume_ma_period' in signals_config:
            volume_ma_period = signals_config['volume_ma_period']
            df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        if 'volatility_window' in signals_config:
            volatility_window = signals_config['volatility_window']
            df['volatility'] = df['close'].pct_change().rolling(window=volatility_window).std()
        
        return df
    
    def _add_market_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification.
        
        Args:
            data: Market data with indicators
            
        Returns:
            Data with market regime information
        """
        df = data.copy()
        
        # Simple market regime classification
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            # Trend strength
            df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
            
            # Market regime classification
            def classify_regime(row):
                if pd.isna(row['trend_strength']) or pd.isna(row['volatility']):
                    return 'UNKNOWN'
                
                trend_threshold = 0.01
                vol_threshold = 0.03
                
                if abs(row['trend_strength']) > trend_threshold:
                    return 'TRENDING'
                elif row['volatility'] > vol_threshold:
                    return 'HIGH_VOL'
                else:
                    return 'RANGING'
            
            df['market_regime'] = df.apply(classify_regime, axis=1)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator.
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range.
        
        Args:
            data: OHLC data
            period: ATR period
            
        Returns:
            ATR values
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def apply_filters(self, signals: List[Signal], data: pd.DataFrame) -> List[Signal]:
        """Apply filters to trading signals.
        
        Args:
            signals: List of trading signals
            data: Market data
            
        Returns:
            Filtered signals
        """
        if not signals:
            return signals
        
        filtered_signals = []
        filters_config = self.config.get('filters', {})
        
        for signal in signals:
            if self._passes_filters(signal, data, filters_config):
                filtered_signals.append(signal)
            else:
                self.logger.log_trade_signal(
                    signal_type=signal.signal_type.value,
                    symbol=signal.symbol,
                    price=signal.price,
                    timestamp=signal.timestamp,
                    reason="Filtered out"
                )
        
        self.logger.info(f"Applied filters: {len(signals)} -> {len(filtered_signals)} signals")
        return filtered_signals
    
    def _passes_filters(self, signal: Signal, data: pd.DataFrame, filters_config: Dict) -> bool:
        """Check if signal passes all filters.
        
        Args:
            signal: Trading signal
            data: Market data
            filters_config: Filters configuration
            
        Returns:
            True if signal passes all filters
        """
        # Market regime filter
        if filters_config.get('market_regime', {}).get('enabled', False):
            if not self._passes_market_regime_filter(signal, data, filters_config):
                return False
        
        # Volatility filter
        if filters_config.get('volatility', {}).get('enabled', False):
            if not self._passes_volatility_filter(signal, data, filters_config):
                return False
        
        # Volume filter
        if filters_config.get('volume', {}).get('enabled', False):
            if not self._passes_volume_filter(signal, data, filters_config):
                return False
        
        # Time filter
        if filters_config.get('time', {}).get('enabled', False):
            if not self._passes_time_filter(signal, data, filters_config):
                return False
        
        return True
    
    def _passes_market_regime_filter(self, signal: Signal, data: pd.DataFrame, filters_config: Dict) -> bool:
        """Check market regime filter.
        
        Args:
            signal: Trading signal
            data: Market data
            filters_config: Filters configuration
            
        Returns:
            True if passes market regime filter
        """
        regime_config = filters_config.get('market_regime', {})
        
        if not regime_config.get('enabled', False):
            return True
        
        # Get current market regime
        current_data = data[data['timestamp'] <= signal.timestamp].tail(1)
        if current_data.empty or 'market_regime' not in current_data.columns:
            return True
        
        current_regime = current_data['market_regime'].iloc[0]
        
        # Check if trending only is required
        if regime_config.get('trending_only', False):
            if current_regime != 'TRENDING':
                return False
        
        # Check trend strength
        min_trend_strength = regime_config.get('min_trend_strength', 0)
        if 'trend_strength' in current_data.columns:
            trend_strength = abs(current_data['trend_strength'].iloc[0])
            if trend_strength < min_trend_strength:
                return False
        
        return True
    
    def _passes_volatility_filter(self, signal: Signal, data: pd.DataFrame, filters_config: Dict) -> bool:
        """Check volatility filter.
        
        Args:
            signal: Trading signal
            data: Market data
            filters_config: Filters configuration
            
        Returns:
            True if passes volatility filter
        """
        vol_config = filters_config.get('volatility', {})
        
        if not vol_config.get('enabled', False):
            return True
        
        # Get current volatility
        current_data = data[data['timestamp'] <= signal.timestamp].tail(1)
        if current_data.empty or 'volatility' not in current_data.columns:
            return True
        
        current_vol = current_data['volatility'].iloc[0]
        
        # Check volatility bounds
        min_vol = vol_config.get('min_volatility', 0)
        max_vol = vol_config.get('max_volatility', float('inf'))
        
        if current_vol < min_vol or current_vol > max_vol:
            return False
        
        return True
    
    def _passes_volume_filter(self, signal: Signal, data: pd.DataFrame, filters_config: Dict) -> bool:
        """Check volume filter.
        
        Args:
            signal: Trading signal
            data: Market data
            filters_config: Filters configuration
            
        Returns:
            True if passes volume filter
        """
        vol_config = filters_config.get('volume', {})
        
        if not vol_config.get('enabled', False):
            return True
        
        # Get current volume data
        current_data = data[data['timestamp'] <= signal.timestamp].tail(1)
        if current_data.empty:
            return True
        
        # Check minimum volume
        min_volume = vol_config.get('min_volume', 0)
        if 'volume' in current_data.columns:
            current_volume = current_data['volume'].iloc[0]
            if current_volume < min_volume:
                return False
        
        # Check volume ratio
        min_volume_ratio = vol_config.get('min_volume_ratio', 0)
        if 'volume_ratio' in current_data.columns:
            current_volume_ratio = current_data['volume_ratio'].iloc[0]
            if current_volume_ratio < min_volume_ratio:
                return False
        
        return True
    
    def _passes_time_filter(self, signal: Signal, data: pd.DataFrame, filters_config: Dict) -> bool:
        """Check time filter.
        
        Args:
            signal: Trading signal
            data: Market data
            filters_config: Filters configuration
            
        Returns:
            True if passes time filter
        """
        time_config = filters_config.get('time', {})
        
        if not time_config.get('enabled', False):
            return True
        
        # Check trading hours
        trading_hours = time_config.get('trading_hours', [0, 23])
        current_hour = signal.timestamp.hour
        
        if current_hour < trading_hours[0] or current_hour > trading_hours[1]:
            return False
        
        # Check weekends (for crypto, this might not apply)
        if time_config.get('avoid_weekends', False):
            if signal.timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
        
        return True
    
    def execute_signals(
        self,
        signals: List[Signal],
        portfolio_value: float,
        current_prices: Dict[str, float]
    ) -> List[Trade]:
        """Execute trading signals.
        
        Args:
            signals: List of trading signals
            portfolio_value: Current portfolio value
            current_prices: Current prices for symbols
            
        Returns:
            List of executed trades
        """
        trades = []
        
        for signal in signals:
            try:
                # Calculate position size
                current_price = current_prices.get(signal.symbol, signal.price)
                position_size = self.calculate_position_size(signal, portfolio_value, current_price)
                
                # Check risk limits
                is_valid, reason = self.risk_manager.check_position_limits(
                    signal.symbol, position_size, current_price, portfolio_value
                )
                
                if not is_valid:
                    self.logger.log_risk_event(
                        event_type="POSITION_LIMIT_EXCEEDED",
                        symbol=signal.symbol,
                        message=reason
                    )
                    continue
                
                # Create trade
                trade = Trade(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    side=PositionSide.LONG if signal.signal_type == SignalType.BUY else PositionSide.SHORT,
                    quantity=position_size,
                    price=current_price,
                    signal=signal,
                    metadata={
                        'strategy': self.strategy_name,
                        'confidence': signal.confidence,
                        'strength': signal.strength.value
                    }
                )
                
                trades.append(trade)
                
                # Log trade execution
                self.logger.log_trade_execution(
                    order_id=f"{self.strategy_name}_{len(trades)}",
                    symbol=trade.symbol,
                    side=trade.side.value,
                    quantity=trade.quantity,
                    price=trade.price,
                    timestamp=trade.timestamp
                )
                
            except Exception as e:
                self.logger.log_risk_event(
                    event_type="TRADE_EXECUTION_ERROR",
                    symbol=signal.symbol,
                    message=str(e)
                )
                continue
        
        self.trades.extend(trades)
        return trades
    
    def update_performance(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Update strategy performance metrics.
        
        Args:
            current_prices: Current prices for all symbols
            
        Returns:
            Updated performance metrics
        """
        if not self.trades:
            return {}
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if self._calculate_trade_pnl(t, current_prices) > 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate P&L metrics
        total_pnl = sum(self._calculate_trade_pnl(t, current_prices) for t in self.trades)
        
        # Calculate returns
        if total_trades > 0:
            avg_return = total_pnl / total_trades
        else:
            avg_return = 0
        
        # Update performance metrics
        self.performance_metrics.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return': avg_return,
            'last_updated': datetime.now().isoformat()
        })
        
        return self.performance_metrics
    
    def _calculate_trade_pnl(self, trade: Trade, current_prices: Dict[str, float]) -> float:
        """Calculate P&L for a trade.
        
        Args:
            trade: Trade object
            current_prices: Current prices
            
        Returns:
            Trade P&L
        """
        current_price = current_prices.get(trade.symbol, trade.price)
        
        if trade.side == PositionSide.LONG:
            return (current_price - trade.price) * trade.quantity
        elif trade.side == PositionSide.SHORT:
            return (trade.price - current_price) * trade.quantity
        else:
            return 0
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary information.
        
        Returns:
            Strategy summary dictionary
        """
        return {
            'strategy_name': self.strategy_name,
            'config': self.config,
            'total_signals': len(self.signals),
            'total_trades': len(self.trades),
            'performance_metrics': self.performance_metrics,
            'positions': self.positions,
            'last_updated': datetime.now().isoformat()
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.positions.clear()
        self.signals.clear()
        self.trades.clear()
        self.performance_metrics.clear()
        self.logger.info(f"Reset strategy {self.strategy_name}")
