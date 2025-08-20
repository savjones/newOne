"""
Main backtesting engine for the crypto trading bot.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from strategies import BaseStrategy
from utils.logging import get_logger
from utils.risk import RiskManager, PositionSide
from backtests.portfolio import Portfolio
from backtests.performance import PerformanceAnalyzer


class EventType(Enum):
    """Event types for the backtesting engine."""
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    TRADE = "TRADE"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"


@dataclass
class Event:
    """Event in the backtesting engine."""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any]
    source: str = ""


class BacktestEngine:
    """Main backtesting engine with event-driven architecture."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_manager: Optional[RiskManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize backtesting engine.
        
        Args:
            initial_capital: Initial portfolio capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            risk_manager: Risk management instance
            logger: Logger instance
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = risk_manager or RiskManager()
        
        # Initialize logger
        self.logger = logger or get_logger("backtest_engine")
        
        # Initialize components
        self.portfolio = Portfolio(initial_capital)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Event handling
        self.events: List[Event] = []
        self.event_handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_results: Dict[str, Dict] = {}
        
        # Data management
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        
        # Backtest state
        self.is_running = False
        self.current_timestamp = None
        self.start_date = None
        self.end_date = None
        
        # Register default event handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self.register_handler(EventType.MARKET_DATA, self._handle_market_data)
        self.register_handler(EventType.SIGNAL, self._handle_signal)
        self.register_handler(EventType.ORDER, self._handle_order)
        self.register_handler(EventType.TRADE, self._handle_trade)
        self.register_handler(EventType.PORTFOLIO_UPDATE, self._handle_portfolio_update)
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for {event_type.value}")
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a trading strategy to the backtest.
        
        Args:
            strategy: Trading strategy instance
        """
        strategy_name = strategy.strategy_name
        self.strategies[strategy_name] = strategy
        self.strategy_results[strategy_name] = {
            'signals': [],
            'trades': [],
            'positions': {},
            'performance': {}
        }
        
        self.logger.info(f"Added strategy: {strategy_name}")
    
    def add_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Add market data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Market data DataFrame
        """
        # Ensure data has required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Convert timestamp to timezone-naive if needed
        if 'timestamp' in data.columns:
            if hasattr(data['timestamp'].dtype, 'tz') and data['timestamp'].dt.tz is not None:
                data['timestamp'] = data['timestamp'].dt.tz_convert(None)
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Store data
        self.market_data[symbol] = data
        
        # Update current price
        if not data.empty:
            self.current_prices[symbol] = data['close'].iloc[-1]
        
        self.logger.info(f"Added market data for {symbol}: {len(data)} rows")
    
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1H'
    ) -> Dict[str, Any]:
        """Run the backtest.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Data timeframe
            
        Returns:
            Backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Validate setup
        if not self.strategies:
            raise ValueError("No strategies added to backtest")
        
        if not self.market_data:
            raise ValueError("No market data added to backtest")
        
        # Initialize backtest state
        self.is_running = True
        self.start_date = start_date
        self.end_date = end_date
        self.current_timestamp = start_date
        
        # Reset portfolio and strategies
        self.portfolio.reset()
        for strategy in self.strategies.values():
            strategy.reset()
        
        # Generate timeline
        timeline = self._generate_timeline(start_date, end_date, timeframe)
        
        # Main backtest loop
        for timestamp in timeline:
            self.current_timestamp = timestamp
            
            # Process market data for this timestamp
            self._process_timestamp(timestamp)
            
            # Update portfolio
            self._update_portfolio(timestamp)
            
            # Check for exit signals
            self._check_exit_signals(timestamp)
        
        # Finalize backtest
        self._finalize_backtest()
        
        # Generate results
        results = self._generate_results()
        
        self.logger.info("Backtest completed")
        self.is_running = False
        
        return results
    
    def _generate_timeline(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> List[datetime]:
        """Generate timeline for the backtest.
        
        Args:
            start_date: Start date
            end_date: End date
            timeframe: Timeframe string
            
        Returns:
            List of timestamps
        """
        # Convert timeframe to pandas frequency
        freq_map = {
            '1M': '1min',
            '5M': '5min',
            '15M': '15min',
            '1H': '1h',
            '4H': '4h',
            '6h': '6h',
            '1D': 'D'
        }
        
        pandas_freq = freq_map.get(timeframe, '1h')
        
        # Generate timeline (ensure timezone-naive)
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_convert(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_convert(None)
            
        timeline = pd.date_range(start=start_date, end=end_date, freq=pandas_freq)
        
        return timeline.tolist()
    
    def _process_timestamp(self, timestamp: datetime) -> None:
        """Process events for a specific timestamp.
        
        Args:
            timestamp: Current timestamp
        """
        # Get market data for this timestamp
        for symbol, data in self.market_data.items():
            # Find data for current timestamp
            current_data = data[data['timestamp'] <= timestamp]
            
            if current_data.empty:
                continue
            
            # Update current price
            latest_price = current_data['close'].iloc[-1]
            self.current_prices[symbol] = latest_price
            
            # Create market data event
            market_event = Event(
                timestamp=timestamp,
                event_type=EventType.MARKET_DATA,
                data={
                    'symbol': symbol,
                    'price': latest_price,
                    'data': current_data.tail(1).iloc[0].to_dict()
                },
                source=symbol
            )
            
            self._emit_event(market_event)
        
        # Generate signals from strategies
        for strategy_name, strategy in self.strategies.items():
            try:
                # Get data for this strategy
                strategy_data = self._get_strategy_data(strategy_name, timestamp)
                
                if not strategy_data.empty:
                    # Generate signals
                    signals = strategy.generate_signals(strategy_data)
                    
                    # Process each signal
                    for signal in signals:
                        signal_event = Event(
                            timestamp=timestamp,
                            event_type=EventType.SIGNAL,
                            data={
                                'strategy': strategy_name,
                                'signal': signal
                            },
                            source=strategy_name
                        )
                        
                        self._emit_event(signal_event)
                        
                        # Store signal in results
                        self.strategy_results[strategy_name]['signals'].append(signal)
                
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {e}")
    
    def _get_strategy_data(self, strategy_name: str, timestamp: datetime) -> pd.DataFrame:
        """Get data for a specific strategy at a given timestamp.
        
        Args:
            strategy_name: Strategy name
            timestamp: Current timestamp
            
        Returns:
            Strategy data
        """
        # For now, return all available data up to timestamp
        # In a more sophisticated implementation, you might want to limit the data
        all_data = []
        
        for symbol, data in self.market_data.items():
            symbol_data = data[data['timestamp'] <= timestamp].copy()
            if not symbol_data.empty:
                symbol_data['symbol'] = symbol
                all_data.append(symbol_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        
        return pd.DataFrame()
    
    def _emit_event(self, event: Event) -> None:
        """Emit an event to all registered handlers.
        
        Args:
            event: Event to emit
        """
        self.events.append(event)
        
        # Get handlers for this event type
        handlers = self.event_handlers.get(event.event_type, [])
        
        # Call each handler
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler {handler.__name__}: {e}")
    
    def _handle_market_data(self, event: Event) -> None:
        """Handle market data events.
        
        Args:
            event: Market data event
        """
        # Update portfolio with current prices
        symbol = event.data['symbol']
        price = event.data['price']
        
        self.portfolio.update_price(symbol, price)
    
    def _handle_signal(self, event: Event) -> None:
        """Handle trading signal events.
        
        Args:
            event: Signal event
        """
        strategy_name = event.data['strategy']
        signal = event.data['signal']
        
        # Check if we should execute this signal
        if self._should_execute_signal(signal):
            # Create order
            order = self._create_order_from_signal(signal, strategy_name)
            
            # Emit order event
            order_event = Event(
                timestamp=event.timestamp,
                event_type=EventType.ORDER,
                data={'order': order},
                source=strategy_name
            )
            
            self._emit_event(order_event)
    
    def _should_execute_signal(self, signal) -> bool:
        """Check if a signal should be executed.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if signal should be executed
        """
        # Basic checks
        if signal.confidence < 0.5:
            return False
        
        # Check if we already have a position in this symbol
        if self.portfolio.has_position(signal.symbol):
            return False
        
        # Check risk limits
        # This would be more sophisticated in production
        
        return True
    
    def _create_order_from_signal(self, signal, strategy_name: str) -> Dict[str, Any]:
        """Create an order from a trading signal.
        
        Args:
            signal: Trading signal
            strategy_name: Strategy name
            
        Returns:
            Order dictionary
        """
        # Calculate position size
        portfolio_value = self.portfolio.get_total_value()
        current_price = self.current_prices.get(signal.symbol, signal.price)
        
        strategy = self.strategies[strategy_name]
        position_size = strategy.calculate_position_size(signal, portfolio_value, current_price)
        
        # Create order
        order = {
            'id': f"{strategy_name}_{len(self.events)}",
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'side': 'BUY' if signal.signal_type.value in ['long', 'BUY'] else 'SELL',
            'quantity': position_size,
            'price': current_price,
            'strategy': strategy_name,
            'signal': signal
        }
        
        return order
    
    def _handle_order(self, event: Event) -> None:
        """Handle order events.
        
        Args:
            event: Order event
        """
        order = event.data['order']
        
        # Execute the order
        trade = self._execute_order(order)
        
        if trade:
            # Emit trade event
            trade_event = Event(
                timestamp=event.timestamp,
                event_type=EventType.TRADE,
                data={'trade': trade},
                source=order['strategy']
            )
            
            self._emit_event(trade_event)
    
    def _execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute an order.
        
        Args:
            order: Order to execute
            
        Returns:
            Executed trade or None
        """
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        price = order['price']
        
        # Apply slippage
        if side == 'BUY':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate commission
        commission_amount = execution_price * quantity * self.commission
        
        # Execute the trade
        success = self.portfolio.execute_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            commission=commission_amount
        )
        
        if success:
            # Create trade record
            trade = {
                'id': order['id'],
                'timestamp': order['timestamp'],
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'commission': commission_amount,
                'strategy': order['strategy']
            }
            
            # Store in strategy results
            strategy_name = order['strategy']
            if strategy_name in self.strategy_results:
                self.strategy_results[strategy_name]['trades'].append(trade)
            
            return trade
        
        return None
    
    def _handle_trade(self, event: Event) -> None:
        """Handle trade events.
        
        Args:
            event: Trade event
        """
        trade = event.data['trade']
        
        # Update strategy positions
        strategy_name = trade['strategy']
        if strategy_name in self.strategy_results:
            symbol = trade['symbol']
            side = trade['side']
            quantity = trade['quantity']
            price = trade['price']
            
            # Update positions
            if symbol not in self.strategy_results[strategy_name]['positions']:
                self.strategy_results[strategy_name]['positions'][symbol] = {
                    'side': side,
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': trade['timestamp']
                }
            else:
                # Update existing position
                position = self.strategy_results[strategy_name]['positions'][symbol]
                if position['side'] == side:
                    # Same side, add to position
                    total_quantity = position['quantity'] + quantity
                    avg_price = ((position['quantity'] * position['entry_price']) + 
                               (quantity * price)) / total_quantity
                    position['quantity'] = total_quantity
                    position['entry_price'] = avg_price
                else:
                    # Opposite side, reduce position
                    remaining_quantity = position['quantity'] - quantity
                    if remaining_quantity <= 0:
                        # Position closed
                        del self.strategy_results[strategy_name]['positions'][symbol]
                    else:
                        position['quantity'] = remaining_quantity
        
        # Emit portfolio update event
        portfolio_event = Event(
            timestamp=event.timestamp,
            event_type=EventType.PORTFOLIO_UPDATE,
            data={'trade': trade},
            source='portfolio'
        )
        
        self._emit_event(portfolio_event)
    
    def _handle_portfolio_update(self, event: Event) -> None:
        """Handle portfolio update events.
        
        Args:
            event: Portfolio update event
        """
        # Portfolio is updated automatically when trades are executed
        # This handler can be used for additional portfolio-related logic
        pass
    
    def _update_portfolio(self, timestamp: datetime) -> None:
        """Update portfolio for current timestamp.
        
        Args:
            timestamp: Current timestamp
        """
        # Update portfolio with current prices
        for symbol, price in self.current_prices.items():
            self.portfolio.update_price(symbol, price)
        
        # Calculate portfolio metrics
        self.portfolio.calculate_metrics()
    
    def _check_exit_signals(self, timestamp: datetime) -> None:
        """Check for exit signals from strategies.
        
        Args:
            timestamp: Current timestamp
        """
        for strategy_name, strategy in self.strategies.items():
            # Check each position for exit conditions
            positions = self.strategy_results[strategy_name]['positions']
            
            for symbol, position in list(positions.items()):
                if symbol not in self.current_prices:
                    continue
                
                current_price = self.current_prices[symbol]
                entry_price = position['entry_price']
                
                # Check if position should be exited
                should_exit, reason = strategy.should_exit_position(
                    symbol, 
                    self._get_strategy_data(strategy_name, timestamp),
                    entry_price,
                    current_price
                )
                
                if should_exit:
                    # Create exit order
                    exit_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                    
                    exit_order = {
                        'id': f"{strategy_name}_exit_{symbol}_{timestamp}",
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'side': exit_side,
                        'quantity': position['quantity'],
                        'price': current_price,
                        'strategy': strategy_name,
                        'reason': reason
                    }
                    
                    # Execute exit order
                    exit_trade = self._execute_order(exit_order)
                    
                    if exit_trade:
                        # Remove position
                        del positions[symbol]
                        
                        self.logger.info(f"Exited position in {symbol}: {reason}")
    
    def _finalize_backtest(self) -> None:
        """Finalize the backtest."""
        # Close all remaining positions
        for strategy_name, strategy in self.strategies.items():
            positions = self.strategy_results[strategy_name]['positions']
            
            for symbol, position in list(positions.items()):
                if symbol in self.current_prices:
                    current_price = self.current_prices[symbol]
                    exit_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                    
                    exit_order = {
                        'id': f"{strategy_name}_final_exit_{symbol}",
                        'timestamp': self.current_timestamp,
                        'symbol': symbol,
                        'side': exit_side,
                        'quantity': position['quantity'],
                        'price': current_price,
                        'strategy': strategy_name,
                        'reason': 'Backtest end'
                    }
                    
                    self._execute_order(exit_order)
                    del positions[symbol]
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results.
        
        Args:
            Backtest results dictionary
        """
        # Portfolio performance
        portfolio_history = self.portfolio.get_history()
        portfolio_performance = self.performance_analyzer.analyze_portfolio(portfolio_history)
        
        # Strategy performance
        strategy_performance = {}
        for strategy_name, results in self.strategy_results.items():
            strategy_performance[strategy_name] = self.performance_analyzer.analyze_strategy(
                results['trades'],
                results['signals']
            )
        
        # Overall results
        results = {
            'backtest_info': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'final_capital': self.portfolio.get_total_value(),
                'total_return': (self.portfolio.get_total_value() - self.initial_capital) / self.initial_capital
            },
            'portfolio_performance': portfolio_performance,
            'strategy_performance': strategy_performance,
            'events': len(self.events),
            'trades': sum(len(results['trades']) for results in self.strategy_results.values())
        }
        
        return results
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies.
        
        Returns:
            Strategy summary dictionary
        """
        summary = {}
        
        for strategy_name, strategy in self.strategies.items():
            summary[strategy_name] = {
                'config': strategy.config,
                'performance': strategy.get_strategy_summary(),
                'positions': self.strategy_results[strategy_name]['positions'],
                'total_signals': len(self.strategy_results[strategy_name]['signals']),
                'total_trades': len(self.strategy_results[strategy_name]['trades'])
            }
        
        return summary
