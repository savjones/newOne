"""Performance metric helpers."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def trade_expectancy(trades: pd.DataFrame) -> float:
    """Return average profit per trade."""
    if trades.empty:
        return 0.0
    return trades['PnL'].mean()


def extract_metrics(stats: pd.Series) -> Dict[str, float]:
    """Extract relevant metrics from a backtesting result ``stats``."""
    trades = getattr(stats, '_trades', pd.DataFrame())
    out = dict(
        trades=len(trades),
        expectancy=trade_expectancy(trades),
        profit_factor=float(stats.get('Profit Factor', float('nan'))),
        sortino=float(stats.get('Sortino Ratio', float('nan'))),
        sharpe=float(stats.get('Sharpe Ratio', float('nan'))),
        max_drawdown_pct=float(stats.get('Max Drawdown [%]', float('nan'))),
        exposure_pct=float(stats.get('Exposure Time [%]', float('nan'))),
        win_rate_pct=float(stats.get('Win Rate [%]', float('nan'))),
        cagr=float(stats.get('Return (Ann.) [%]', float('nan'))),
    )
    start_cash = float(stats.get('Start Cash', 1.0))
    eq_final = float(stats.get('Equity Final [$]', start_cash))
    out['equity_final_pct'] = 100 * eq_final / start_cash
    return out
