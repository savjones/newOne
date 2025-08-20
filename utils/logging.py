"""
Logging configuration for the crypto trading bot.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    name: str = "crypto_trading_bot",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    # Handle both string and integer log levels
    if isinstance(level, str):
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        # Handle both string and integer log levels for console handler
        if isinstance(level, str):
            console_handler.setLevel(getattr(logging, level.upper()))
        else:
            console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        main_log_file = log_path / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_path / f"{name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = "crypto_trading_bot") -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, strategy_name: str, log_dir: Optional[str] = None):
        """Initialize trading logger.
        
        Args:
            strategy_name: Name of the trading strategy
            log_dir: Directory for log files
        """
        self.strategy_name = strategy_name
        self.logger = setup_logging(
            name=f"trading.{strategy_name}",
            log_dir=log_dir,
            level="INFO"
        )
    
    def log_trade_signal(self, signal_type: str, symbol: str, price: float, 
                         timestamp: datetime, **kwargs) -> None:
        """Log a trade signal.
        
        Args:
            signal_type: Type of signal (BUY, SELL, HOLD)
            symbol: Trading symbol
            price: Signal price
            timestamp: Signal timestamp
            **kwargs: Additional signal parameters
        """
        extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"TRADE_SIGNAL: {signal_type} {symbol} @ {price:.8f} | {extra_info}"
        )
    
    def log_trade_execution(self, order_id: str, symbol: str, side: str,
                           quantity: float, price: float, timestamp: datetime) -> None:
        """Log trade execution.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            timestamp: Execution timestamp
        """
        self.logger.info(
            f"TRADE_EXECUTED: {order_id} {side} {quantity:.8f} {symbol} @ {price:.8f}"
        )
    
    def log_position_update(self, symbol: str, side: str, quantity: float,
                           avg_price: float, pnl: float) -> None:
        """Log position update.
        
        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            quantity: Position quantity
            avg_price: Average position price
            pnl: Unrealized P&L
        """
        self.logger.info(
            f"POSITION_UPDATE: {side} {quantity:.8f} {symbol} @ {avg_price:.8f} | PnL: {pnl:.2f}"
        )
    
    def log_risk_event(self, event_type: str, symbol: str, message: str,
                       **kwargs) -> None:
        """Log risk management events.
        
        Args:
            event_type: Type of risk event
            symbol: Trading symbol
            message: Risk event message
            **kwargs: Additional risk parameters
        """
        extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.warning(
            f"RISK_EVENT: {event_type} {symbol} | {message} | {extra_info}"
        )
    
    def log_performance(self, metric: str, value: float, **kwargs) -> None:
        """Log performance metrics.
        
        Args:
            metric: Performance metric name
            value: Metric value
            **kwargs: Additional metric parameters
        """
        extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"PERFORMANCE: {metric}={value:.4f} | {extra_info}"
        )
    
    def __getattr__(self, name):
        """Delegate missing methods to the underlying logger."""
        return getattr(self.logger, name)


# Global logger instance
trading_logger = get_logger("crypto_trading_bot")
