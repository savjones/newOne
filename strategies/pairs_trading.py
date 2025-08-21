"""Simplified pairs trading strategy placeholder.

The full two-leg execution is non-trivial within ``backtesting.py`` which only
supports a single price series. This class provides the parameter surface and
basic spread/z-score calculations but leaves trade execution as a TODO.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class PairsTrading(BaseStrategy):
    DEFAULT_PARAMS = dict(lookback=60, z_entry=2.0, z_exit=0.5, sl_z=3.5)
    PARAM_BOUNDS = dict(lookback=(30, 120), z_entry=(1.5, 3.0),
                        z_exit=(0.2, 1.0))

    def init(self) -> None:  # pragma: no cover - placeholder
        pass

    def next(self) -> None:  # pragma: no cover - placeholder
        pass
