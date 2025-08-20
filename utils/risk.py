"""
Risk management utilities for the crypto trading bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class Position:
    """Position information."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - self.current_price) * self.quantity
        return 0.0
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    drawdown: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float


class RiskManager:
    """Risk management system for the trading bot."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_drawdown: float = 0.2,
        max_leverage: float = 1.0,
        stop_loss_default: float = 0.05,
        take_profit_default: float = 0.15,
        max_correlation: float = 0.7,
        var_confidence: float = 0.95
    ):
        """Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as % of portfolio
            max_drawdown: Maximum allowed drawdown
            max_leverage: Maximum allowed leverage
            stop_loss_default: Default stop loss percentage
            take_profit_default: Default take profit percentage
            max_correlation: Maximum correlation between positions
            var_confidence: VaR confidence level
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.stop_loss_default = stop_loss_default
        self.take_profit_default = take_profit_default
        self.max_correlation = max_correlation
        self.var_confidence = var_confidence
        
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[Dict] = []
        self.risk_events: List[Dict] = []
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        symbol: str,
        price: float,
        volatility: float,
        confidence: float = 1.0
    ) -> float:
        """Calculate position size based on risk parameters.
        
        Args:
            portfolio_value: Total portfolio value
            symbol: Trading symbol
            price: Current price
            volatility: Asset volatility
            confidence: ML model confidence (0-1)
            
        Returns:
            Position size in base currency
        """
        # Base position size
        max_position_value = portfolio_value * self.max_position_size
        
        # Volatility adjustment
        vol_adjustment = 1.0 / (1.0 + volatility)
        
        # Confidence adjustment
        confidence_adjustment = confidence * 0.5 + 0.5
        
        # Final position size
        position_value = max_position_value * vol_adjustment * confidence_adjustment
        
        # Convert to quantity
        quantity = position_value / price
        
        return quantity
    
    def check_position_limits(
        self,
        symbol: str,
        quantity: float,
        price: float,
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Check if position meets risk limits.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Position price
            portfolio_value: Total portfolio value
            
        Returns:
            Tuple of (is_valid, reason)
        """
        position_value = abs(quantity * price)
        position_size_pct = position_value / portfolio_value
        
        # Check position size limit
        if position_size_pct > self.max_position_size:
            return False, f"Position size {position_size_pct:.2%} exceeds limit {self.max_position_size:.2%}"
        
        # Check leverage limit
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in self.positions.values())
        leverage = total_exposure / portfolio_value
        
        if leverage > self.max_leverage:
            return False, f"Leverage {leverage:.2f} exceeds limit {self.max_leverage:.2f}"
        
        return True, "Position within limits"
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: PositionSide,
        atr: float,
        multiplier: float = 2.0
    ) -> float:
        """Calculate dynamic stop loss based on ATR.
        
        Args:
            entry_price: Entry price
            side: Position side
            atr: Average True Range
            multiplier: ATR multiplier
            
        Returns:
            Stop loss price
        """
        atr_stop = atr * multiplier
        
        if side == PositionSide.LONG:
            return entry_price - atr_stop
        elif side == PositionSide.SHORT:
            return entry_price + atr_stop
        
        return entry_price
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: PositionSide,
        risk_reward_ratio: float = 3.0,
        stop_loss_pct: float = None
    ) -> float:
        """Calculate take profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            side: Position side
            risk_reward_ratio: Risk-reward ratio
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Take profit price
        """
        if stop_loss_pct is None:
            stop_loss_pct = self.stop_loss_default
        
        risk_amount = entry_price * stop_loss_pct
        reward_amount = risk_amount * risk_reward_ratio
        
        if side == PositionSide.LONG:
            return entry_price + reward_amount
        elif side == PositionSide.SHORT:
            return entry_price - reward_amount
        
        return entry_price
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss has been triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if stop loss triggered
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        if position.stop_loss is None:
            return False
        
        if position.side == PositionSide.LONG and current_price <= position.stop_loss:
            return True
        elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
            return True
        
        return False
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit has been triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if take profit triggered
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        if position.take_profit is None:
            return False
        
        if position.side == PositionSide.LONG and current_price >= position.take_profit:
            return True
        elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
            return True
        
        return False
    
    def calculate_portfolio_metrics(
        self,
        portfolio_value: float,
        cash: float,
        returns: List[float]
    ) -> PortfolioMetrics:
        """Calculate portfolio performance metrics.
        
        Args:
            portfolio_value: Current portfolio value
            cash: Available cash
            returns: List of historical returns
            
        Returns:
            Portfolio metrics object
        """
        positions_value = portfolio_value - cash
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Calculate realized P&L (simplified)
        realized_pnl = 0.0  # This would come from trade history
        
        total_pnl = unrealized_pnl + realized_pnl
        
        # Calculate drawdown
        if returns:
            cumulative_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        else:
            current_drawdown = 0.0
            max_drawdown = 0.0
        
        # Calculate Sharpe ratio
        if returns and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252) if returns else 0.0
        
        return PortfolioMetrics(
            total_value=portfolio_value,
            cash=cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility
        )
    
    def check_portfolio_risk(self, metrics: PortfolioMetrics) -> List[str]:
        """Check portfolio-level risk metrics.
        
        Args:
            metrics: Portfolio metrics
            
        Returns:
            List of risk warnings
        """
        warnings = []
        
        # Check drawdown limit
        if abs(metrics.drawdown) > self.max_drawdown:
            warnings.append(f"Drawdown {metrics.drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
        
        # Check leverage
        if metrics.positions_value > 0:
            leverage = metrics.positions_value / metrics.total_value
            if leverage > self.max_leverage:
                warnings.append(f"Leverage {leverage:.2f} exceeds limit {self.max_leverage:.2f}")
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < 0.5:
            warnings.append(f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        return warnings
    
    def add_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Add a new position.
        
        Args:
            symbol: Trading symbol
            side: Position side
            quantity: Position quantity
            entry_price: Entry price
            entry_time: Entry time
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        if stop_loss is None:
            stop_loss = entry_price * (1 - self.stop_loss_default) if side == PositionSide.LONG else entry_price * (1 + self.stop_loss_default)
        
        if take_profit is None:
            take_profit = entry_price * (1 + self.take_profit_default) if side == PositionSide.LONG else entry_price * (1 - self.take_profit_default)
        
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update position current price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
    
    def close_position(self, symbol: str) -> Optional[Position]:
        """Close a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Closed position or None if not found
        """
        return self.positions.pop(symbol, None)
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions.
        
        Returns:
            Dictionary of position summaries
        """
        summary = {}
        for symbol, position in self.positions.items():
            summary[symbol] = {
                "side": position.side.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "unrealized_pnl_pct": position.unrealized_pnl_pct,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit
            }
        return summary
