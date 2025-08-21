"""Technical analysis helpers used across strategies.

These implementations are intentionally lightweight to avoid pulling in
heavy dependencies beyond pandas/numpy. All functions return ``pandas.Series``
objects aligned with the input ``close`` series.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def donchian_high(series: pd.Series, period: int) -> pd.Series:
    """Rolling highest high used for Donchian channel."""
    return series.rolling(period).max()


def donchian_low(series: pd.Series, period: int) -> pd.Series:
    """Rolling lowest low used for Donchian channel."""
    return series.rolling(period).min()


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        fast: int = 7, mid: int = 14, slow: int = 28) -> pd.Series:
    """Compute the Ultimate Oscillator.

    Parameters mirror the classic UO implementation where ``fast``, ``mid`` and
    ``slow`` are lookback periods for the weighted BP/TR averages.
    """
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    avg_fast = bp.rolling(fast).sum() / tr.rolling(fast).sum()
    avg_mid = bp.rolling(mid).sum() / tr.rolling(mid).sum()
    avg_slow = bp.rolling(slow).sum() / tr.rolling(slow).sum()

    uo = 100 * (4 * avg_fast + 2 * avg_mid + avg_slow) / 7
    return uo


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index implementation."""
    typical_price = (high + low + close) / 3
    raw_money = typical_price * volume
    direction = np.where(typical_price > typical_price.shift(1), 1,
                         np.where(typical_price < typical_price.shift(1), -1, 0))
    pos_flow = (raw_money * (direction == 1)).rolling(period).sum()
    neg_flow = (raw_money * (direction == -1)).rolling(period).sum()
    mfi = 100 - 100 / (1 + pos_flow / neg_flow)
    return mfi


def rolling_vwap(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Rolling VWAP over ``period`` bars."""
    price_vol = close * volume
    pv_sum = price_vol.rolling(period).sum()
    vol_sum = volume.rolling(period).sum()
    return pv_sum / vol_sum


def rolling_median(series: pd.Series, period: int) -> pd.Series:
    """Convenience wrapper for rolling median used in some strategies."""
    return series.rolling(period).median()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average Directional Index."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(period).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / atr_val
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()
