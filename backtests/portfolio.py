"""
Portfolio management for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass

from utils.risk import PositionSide


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update position with current price."""
        self.current_price = price
        self._calculate_pnl()
    
    def _calculate_pnl(self) -> None:
        """Calculate P&L for the position."""
        if self.side == 'BUY':
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:  # SELL (short position)
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
    
    def close(self, exit_price: float, exit_time: datetime) -> float:
        """Close the position and return realized P&L."""
        if self.side == 'BUY':
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SELL (short position)
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        return self.realized_pnl


@dataclass
class Trade:
    """Trade record."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    strategy: str


class Portfolio:
    """Portfolio management system for backtesting."""
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize portfolio.
        
        Args:
            initial_capital: Initial portfolio capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Positions and trades
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        
        # Portfolio history
        self.history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        
        # Initialize history
        self._record_snapshot()
    
    def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float
    ) -> bool:
        """Execute a trade.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('BUY' or 'SELL')
            quantity: Trade quantity
            price: Execution price
            commission: Commission amount
            
        Returns:
            True if trade executed successfully
        """
        # Calculate trade value
        trade_value = quantity * price
        total_cost = trade_value + commission
        
        # Check if we have enough cash for buy orders
        if side == 'BUY':
            if self.cash < total_cost:
                return False
            
            # Deduct cash
            self.cash -= total_cost
            
            # Update or create position
            if symbol in self.positions:
                # Add to existing position
                position = self.positions[symbol]
                if position.side == 'BUY':
                    # Same side, average the price
                    total_quantity = position.quantity + quantity
                    total_cost_basis = (position.quantity * position.entry_price) + trade_value
                    position.entry_price = total_cost_basis / total_quantity
                    position.quantity = total_quantity
                else:
                    # Opposite side, reduce position
                    remaining_quantity = position.quantity - quantity
                    if remaining_quantity <= 0:
                        # Position closed
                        realized_pnl = position.close(price, datetime.now())
                        self.realized_pnl += realized_pnl
                        self.cash += abs(remaining_quantity) * price
                        del self.positions[symbol]
                    else:
                        # Reduce position
                        position.quantity = remaining_quantity
                        realized_pnl = (price - position.entry_price) * quantity
                        self.realized_pnl += realized_pnl
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                    unrealized_pnl=0.0
                )
        
        elif side == 'SELL':
            # For sell orders, we need to have a long position or create a short position
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.side == 'BUY':
                    # Close long position
                    remaining_quantity = position.quantity - quantity
                    if remaining_quantity <= 0:
                        # Position closed
                        realized_pnl = position.close(price, datetime.now())
                        self.realized_pnl += realized_pnl
                        self.cash += abs(remaining_quantity) * price
                        del self.positions[symbol]
                    else:
                        # Reduce position
                        position.quantity = remaining_quantity
                        realized_pnl = (price - position.entry_price) * quantity
                        self.realized_pnl += realized_pnl
                else:
                    # Add to short position
                    total_quantity = position.quantity + quantity
                    total_cost_basis = (position.quantity * position.entry_price) + trade_value
                    position.entry_price = total_cost_basis / total_quantity
                    position.quantity = total_quantity
            else:
                # Create new short position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                    unrealized_pnl=0.0
                )
            
            # Add cash from sale
            self.cash += trade_value - commission
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            strategy='backtest'
        )
        
        self.trades.append(trade)
        
        # Update portfolio metrics
        self._update_metrics()
        
        return True
    
    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
            self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update portfolio metrics."""
        # Calculate unrealized P&L
        self.unrealized_pnl = sum(
            position.unrealized_pnl for position in self.positions.values()
        )
        
        # Total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Update peak value and drawdown
        current_value = self.get_total_value()
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_total_value(self) -> float:
        """Get total portfolio value.
        
        Returns:
            Total portfolio value
        """
        positions_value = sum(
            position.quantity * position.current_price 
            for position in self.positions.values()
        )
        
        return self.cash + positions_value
    
    def get_positions_value(self) -> float:
        """Get total value of all positions.
        
        Returns:
            Total positions value
        """
        return sum(
            position.quantity * position.current_price 
            for position in self.positions.values()
        )
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has a position in a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position exists
        """
        return symbol in self.positions
    
    def get_positions_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all positions.
        
        Returns:
            Dictionary of position summaries
        """
        summary = {}
        
        for symbol, position in self.positions.items():
            summary[symbol] = {
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'entry_time': position.entry_time,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'market_value': position.quantity * position.current_price
            }
        
        return summary
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        
        # Calculate returns
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate allocation
        cash_allocation = self.cash / total_value if total_value > 0 else 0
        positions_allocation = positions_value / total_value if total_value > 0 else 0
        
        # Calculate turnover
        total_trades_value = sum(
            trade.quantity * trade.price for trade in self.trades
        )
        turnover = total_trades_value / self.initial_capital if self.initial_capital > 0 else 0
        
        metrics = {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return': total_return,
            'cash_allocation': cash_allocation,
            'positions_allocation': positions_allocation,
            'max_drawdown': self.max_drawdown,
            'turnover': turnover,
            'total_trades': len(self.trades)
        }
        
        return metrics
    
    def _record_snapshot(self) -> None:
        """Record portfolio snapshot."""
        snapshot = {
            'timestamp': datetime.now(),
            'metrics': self.calculate_metrics()
        }
        
        self.history.append(snapshot)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get portfolio history.
        
        Returns:
            List of portfolio snapshots
        """
        return self.history
    
    def get_returns_series(self) -> pd.Series:
        """Get portfolio returns time series.
        
        Returns:
            Series of portfolio returns
        """
        if not self.history:
            return pd.Series()
        
        # Extract total values and calculate returns
        values = [snapshot['metrics']['total_value'] for snapshot in self.history]
        timestamps = [snapshot['timestamp'] for snapshot in self.history]
        
        # Calculate returns
        returns = pd.Series(values, index=timestamps).pct_change().dropna()
        
        return returns
    
    def get_drawdown_series(self) -> pd.Series:
        """Get portfolio drawdown time series.
        
        Returns:
            Series of portfolio drawdowns
        """
        if not self.history:
            return pd.Series()
        
        # Extract total values
        values = [snapshot['metrics']['total_value'] for snapshot in self.history]
        timestamps = [snapshot['timestamp'] for snapshot in self.history]
        
        # Calculate drawdown
        values_series = pd.Series(values, index=timestamps)
        running_max = values_series.expanding().max()
        drawdown = (values_series - running_max) / running_max
        
        return drawdown
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get portfolio equity curve.
        
        Returns:
            DataFrame with portfolio metrics over time
        """
        if not self.history:
            return pd.DataFrame()
        
        # Convert history to DataFrame
        equity_data = []
        
        for snapshot in self.history:
            row = {
                'timestamp': snapshot['timestamp'],
                **snapshot['metrics']
            }
            equity_data.append(row)
        
        equity_df = pd.DataFrame(equity_data)
        equity_df = equity_df.set_index('timestamp').sort_index()
        
        return equity_df
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.closed_trades.clear()
        self.history.clear()
        
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        
        # Record initial snapshot
        self._record_snapshot()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary.
        
        Returns:
            Portfolio summary dictionary
        """
        metrics = self.calculate_metrics()
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_metrics': metrics,
            'positions_count': len(self.positions),
            'total_trades': len(self.trades),
            'positions_summary': self.get_positions_summary()
        }
        
        return summary
