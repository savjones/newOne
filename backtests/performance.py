"""
Performance analysis for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging

from utils.logging import get_logger


class PerformanceAnalyzer:
    """Performance analysis for backtesting results."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance analyzer.
        
        Args:
            risk_free_rate: Risk-free rate for calculations (annualized)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = get_logger("performance_analyzer")
    
    def analyze_portfolio(self, portfolio_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze portfolio performance.
        
        Args:
            portfolio_history: Portfolio history from backtest
            
        Returns:
            Portfolio performance metrics
        """
        if not portfolio_history:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Extract key metrics
        total_values = df['metrics'].apply(lambda x: x['total_value'])
        returns = total_values.pct_change().dropna()
        
        # Calculate performance metrics
        metrics = {
            'total_return': self._calculate_total_return(total_values),
            'annualized_return': self._calculate_annualized_return(total_values),
            'volatility': self._calculate_volatility(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(total_values),
            'max_drawdown': self._calculate_max_drawdown(total_values),
            'var_95': self._calculate_var(returns, 0.95),
            'cvar_95': self._calculate_cvar(returns, 0.95),
            'hit_rate': self._calculate_hit_rate(returns),
            'profit_factor': self._calculate_profit_factor(returns),
            'turnover': self._calculate_turnover(df),
            'win_rate': self._calculate_win_rate(returns),
            'avg_win': self._calculate_avg_win(returns),
            'avg_loss': self._calculate_avg_loss(returns),
            'best_day': self._calculate_best_day(returns),
            'worst_day': self._calculate_worst_day(returns),
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns)
        }
        
        return metrics
    
    def analyze_strategy(
        self,
        trades: List[Dict[str, Any]],
        signals: List[Any]
    ) -> Dict[str, Any]:
        """Analyze strategy performance.
        
        Args:
            trades: List of trades
            signals: List of signals
            
        Returns:
            Strategy performance metrics
        """
        if not trades:
            return {}
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.set_index('timestamp').sort_index()
        
        # Calculate trade-based metrics
        metrics = {
            'total_trades': len(trades),
            'winning_trades': self._count_winning_trades(trades_df),
            'losing_trades': self._count_losing_trades(trades_df),
            'win_rate': self._calculate_trade_win_rate(trades_df),
            'avg_trade_return': self._calculate_avg_trade_return(trades_df),
            'avg_winning_trade': self._calculate_avg_winning_trade(trades_df),
            'avg_losing_trade': self._calculate_avg_losing_trade(trades_df),
            'largest_win': self._calculate_largest_win(trades_df),
            'largest_loss': self._calculate_largest_loss(trades_df),
            'profit_factor': self._calculate_trade_profit_factor(trades_df),
            'avg_trade_duration': self._calculate_avg_trade_duration(trades_df),
            'total_signals': len(signals) if signals else 0
        }
        
        return metrics
    
    def _calculate_total_return(self, values: pd.Series) -> float:
        """Calculate total return.
        
        Args:
            values: Portfolio values series
            
        Returns:
            Total return
        """
        if len(values) < 2:
            return 0.0
        
        return (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
    
    def _calculate_annualized_return(self, values: pd.Series) -> float:
        """Calculate annualized return.
        
        Args:
            values: Portfolio values series
            
        Returns:
            Annualized return
        """
        if len(values) < 2:
            return 0.0
        
        total_return = self._calculate_total_return(values)
        
        # Calculate time period in years
        time_period = (values.index[-1] - values.index[0]).total_seconds() / (365.25 * 24 * 3600)
        
        if time_period <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / time_period) - 1
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility.
        
        Args:
            returns: Returns series
            
        Returns:
            Annualized volatility
        """
        if returns.empty:
            return 0.0
        
        # Assuming daily data, multiply by sqrt(252) for annualization
        # For hourly data, multiply by sqrt(252 * 24)
        # For minute data, multiply by sqrt(252 * 24 * 60)
        
        # Detect frequency
        if len(returns) > 1:
            time_diff = returns.index[1] - returns.index[0]
            if time_diff.total_seconds() <= 60:  # Minute data
                annualization_factor = np.sqrt(252 * 24 * 60)
            elif time_diff.total_seconds() <= 3600:  # Hourly data
                annualization_factor = np.sqrt(252 * 24)
            else:  # Daily data
                annualization_factor = np.sqrt(252)
        else:
            annualization_factor = np.sqrt(252)
        
        return returns.std() * annualization_factor
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Returns series
            
        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        # Annualize
        annualized_return = excess_returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        
        if annualized_volatility == 0:
            return 0.0
        
        return annualized_return / annualized_volatility
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio.
        
        Args:
            returns: Returns series
            
        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / 252)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualize
        annualized_return = excess_returns.mean() * 252
        annualized_downside = downside_deviation * np.sqrt(252)
        
        return annualized_return / annualized_downside
    
    def _calculate_calmar_ratio(self, values: pd.Series) -> float:
        """Calculate Calmar ratio.
        
        Args:
            values: Portfolio values series
            
        Returns:
            Calmar ratio
        """
        if len(values) < 2:
            return 0.0
        
        annualized_return = self._calculate_annualized_return(values)
        max_drawdown = self._calculate_max_drawdown(values)
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / abs(max_drawdown)
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            values: Portfolio values series
            
        Returns:
            Maximum drawdown
        """
        if len(values) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = values.expanding().max()
        
        # Calculate drawdown
        drawdown = (values - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: Returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Value at Risk
        """
        if returns.empty:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Conditional Value at Risk
        """
        if returns.empty:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]
        
        if tail_returns.empty:
            return var
        
        return tail_returns.mean()
    
    def _calculate_hit_rate(self, returns: pd.Series) -> float:
        """Calculate hit rate (percentage of positive returns).
        
        Args:
            returns: Returns series
            
        Returns:
            Hit rate
        """
        if returns.empty:
            return 0.0
        
        return (returns > 0).mean()
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor.
        
        Args:
            returns: Returns series
            
        Returns:
            Profit factor
        """
        if returns.empty:
            return 0.0
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0.0
        
        return positive_returns / negative_returns
    
    def _calculate_turnover(self, df: pd.DataFrame) -> float:
        """Calculate portfolio turnover.
        
        Args:
            df: Portfolio DataFrame
            
        Returns:
            Turnover rate
        """
        if 'metrics' not in df.columns:
            return 0.0
        
        # Extract turnover from metrics
        turnover_values = df['metrics'].apply(lambda x: x.get('turnover', 0))
        
        if turnover_values.empty:
            return 0.0
        
        return turnover_values.mean()
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate.
        
        Args:
            returns: Returns series
            
        Returns:
            Win rate
        """
        return self._calculate_hit_rate(returns)
    
    def _calculate_avg_win(self, returns: pd.Series) -> float:
        """Calculate average winning return.
        
        Args:
            returns: Returns series
            
        Returns:
            Average winning return
        """
        winning_returns = returns[returns > 0]
        
        if winning_returns.empty:
            return 0.0
        
        return winning_returns.mean()
    
    def _calculate_avg_loss(self, returns: pd.Series) -> float:
        """Calculate average losing return.
        
        Args:
            returns: Returns series
            
        Returns:
            Average losing return
        """
        losing_returns = returns[returns < 0]
        
        if losing_returns.empty:
            return 0.0
        
        return losing_returns.mean()
    
    def _calculate_best_day(self, returns: pd.Series) -> float:
        """Calculate best day return.
        
        Args:
            returns: Returns series
            
        Returns:
            Best day return
        """
        if returns.empty:
            return 0.0
        
        return returns.max()
    
    def _calculate_worst_day(self, returns: pd.Series) -> float:
        """Calculate worst day return.
        
        Args:
            returns: Returns series
            
        Returns:
            Worst day return
        """
        if returns.empty:
            return 0.0
        
        return returns.min()
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate returns skewness.
        
        Args:
            returns: Returns series
            
        Returns:
            Skewness
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        return returns.skew()
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate returns kurtosis.
        
        Args:
            returns: Returns series
            
        Returns:
            Kurtosis
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        return returns.kurtosis()
    
    def _count_winning_trades(self, trades_df: pd.DataFrame) -> int:
        """Count winning trades.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Number of winning trades
        """
        if 'side' not in trades_df.columns or 'price' not in trades_df.columns:
            return 0
        
        # This is a simplified calculation
        # In a real implementation, you'd need to track entry/exit prices
        return len(trades_df) // 2  # Placeholder
    
    def _count_losing_trades(self, trades_df: pd.DataFrame) -> int:
        """Count losing trades.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Number of losing trades
        """
        if 'side' not in trades_df.columns or 'price' not in trades_df.columns:
            return 0
        
        # This is a simplified calculation
        return len(trades_df) // 2  # Placeholder
    
    def _calculate_trade_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate trade win rate.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Trade win rate
        """
        winning_trades = self._count_winning_trades(trades_df)
        total_trades = len(trades_df)
        
        if total_trades == 0:
            return 0.0
        
        return winning_trades / total_trades
    
    def _calculate_avg_trade_return(self, trades_df: pd.DataFrame) -> float:
        """Calculate average trade return.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Average trade return
        """
        # This would require tracking actual P&L for each trade
        # For now, return a placeholder
        return 0.0
    
    def _calculate_avg_winning_trade(self, trades_df: pd.DataFrame) -> float:
        """Calculate average winning trade.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Average winning trade
        """
        # Placeholder implementation
        return 0.0
    
    def _calculate_avg_losing_trade(self, trades_df: pd.DataFrame) -> float:
        """Calculate average losing trade.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Average losing trade
        """
        # Placeholder implementation
        return 0.0
    
    def _calculate_largest_win(self, trades_df: pd.DataFrame) -> float:
        """Calculate largest winning trade.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Largest winning trade
        """
        # Placeholder implementation
        return 0.0
    
    def _calculate_largest_loss(self, trades_df: pd.DataFrame) -> float:
        """Calculate largest losing trade.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Largest losing trade
        """
        # Placeholder implementation
        return 0.0
    
    def _calculate_trade_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate trade profit factor.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Trade profit factor
        """
        # Placeholder implementation
        return 0.0
    
    def _calculate_avg_trade_duration(self, trades_df: pd.DataFrame) -> float:
        """Calculate average trade duration.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Average trade duration in hours
        """
        # Placeholder implementation
        return 0.0
    
    def generate_performance_report(
        self,
        portfolio_metrics: Dict[str, Any],
        strategy_metrics: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive performance report.
        
        Args:
            portfolio_metrics: Portfolio performance metrics
            strategy_metrics: Strategy performance metrics
            
        Returns:
            Formatted performance report
        """
        report_lines = [
            "Performance Report",
            "=" * 50,
            "",
            "Portfolio Performance:",
            "-" * 25
        ]
        
        # Portfolio metrics
        if portfolio_metrics:
            key_metrics = [
                ('Total Return', 'total_return', '{:.2%}'),
                ('Annualized Return', 'annualized_return', '{:.2%}'),
                ('Volatility', 'volatility', '{:.2%}'),
                ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
                ('Sortino Ratio', 'sortino_ratio', '{:.2f}'),
                ('Calmar Ratio', 'calmar_ratio', '{:.2f}'),
                ('Max Drawdown', 'max_drawdown', '{:.2%}'),
                ('VaR (95%)', 'var_95', '{:.2%}'),
                ('CVaR (95%)', 'cvar_95', '{:.2%}'),
                ('Hit Rate', 'hit_rate', '{:.2%}'),
                ('Profit Factor', 'profit_factor', '{:.2f}'),
                ('Turnover', 'turnover', '{:.2f}')
            ]
            
            for label, key, fmt in key_metrics:
                if key in portfolio_metrics:
                    value = portfolio_metrics[key]
                    if pd.isna(value):
                        continue
                    report_lines.append(f"{label}: {fmt.format(value)}")
        
        # Strategy metrics
        if strategy_metrics:
            report_lines.extend([
                "",
                "Strategy Performance:",
                "-" * 25
            ])
            
            key_strategy_metrics = [
                ('Total Trades', 'total_trades', '{:d}'),
                ('Win Rate', 'win_rate', '{:.2%}'),
                ('Total Signals', 'total_signals', '{:d}')
            ]
            
            for label, key, fmt in key_strategy_metrics:
                if key in strategy_metrics:
                    value = strategy_metrics[key]
                    if pd.isna(value):
                        continue
                    report_lines.append(f"{label}: {fmt.format(value)}")
        
        return "\n".join(report_lines)
    
    def plot_performance(self, portfolio_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance plots data.
        
        Args:
            portfolio_history: Portfolio history
            
        Returns:
            Dictionary with plot data
        """
        if not portfolio_history:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Extract metrics
        total_values = df['metrics'].apply(lambda x: x['total_value'])
        returns = total_values.pct_change().dropna()
        
        # Calculate drawdown
        running_max = total_values.expanding().max()
        drawdown = (total_values - running_max) / running_max
        
        # Prepare plot data
        plot_data = {
            'equity_curve': {
                'x': total_values.index.tolist(),
                'y': total_values.values.tolist(),
                'name': 'Portfolio Value'
            },
            'returns': {
                'x': returns.index.tolist(),
                'y': returns.values.tolist(),
                'name': 'Returns'
            },
            'drawdown': {
                'x': drawdown.index.tolist(),
                'y': drawdown.values.tolist(),
                'name': 'Drawdown'
            }
        }
        
        return plot_data
