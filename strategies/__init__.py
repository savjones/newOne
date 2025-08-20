"""
Trading Strategies Package

This package contains all implemented trading strategies for the crypto trading bot.
Each strategy inherits from BaseStrategy and implements specific trading logic.
"""

from .base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    SignalStrength,
    Trade
)

from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .microstructure import MicrostructureStrategy
from .pairs_trading import PairsTradingStrategy
from .ultimate_oscillator import UltimateOscillatorStrategy

__all__ = [
    # Base classes
    'BaseStrategy',
    'Signal',
    'SignalType',
    'SignalStrength',
    'Trade',
    
    # Strategy implementations
    'MomentumStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'MicrostructureStrategy',
    'PairsTradingStrategy',
    'UltimateOscillatorStrategy'
]

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'breakout': BreakoutStrategy,
    'microstructure': MicrostructureStrategy,
    'pairs_trading': PairsTradingStrategy,
    'ultimate_oscillator': UltimateOscillatorStrategy
}

def get_strategy(strategy_name: str):
    """Get strategy class by name"""
    return STRATEGY_REGISTRY.get(strategy_name.lower())

def list_strategies():
    """List all available strategies"""
    return list(STRATEGY_REGISTRY.keys())

def get_strategy_description(strategy_name: str) -> str:
    """Get description of a strategy"""
    strategy_class = get_strategy(strategy_name)
    if strategy_class:
        return strategy_class.__doc__ or "No description available"
    return "Strategy not found"
