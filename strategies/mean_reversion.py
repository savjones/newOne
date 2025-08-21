"""RSI2 + Bollinger Band mean reversion strategy."""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from utils.ta import rsi, sma


class MeanReversion(BaseStrategy):
    DEFAULT_PARAMS = dict(rsi_len=2, rsi_buy=10, bb_len=20, bb_std=2.0,
                          sl_pct=0.02, tp_mid=True)
    PARAM_BOUNDS = dict(rsi_len=(2, 5), rsi_buy=(5, 20), bb_len=(10, 40),
                        bb_std=(1.5, 3.0), sl_pct=(0.01, 0.03))

    def init(self) -> None:
        self.rsi = self.I(rsi, self.data.Close, self.rsi_len)
        self.bb_mid = self.I(sma, self.data.Close, self.bb_len)
        self.bb_stddev = self.I(lambda s, n: pd.Series(s).rolling(n).std(),
                                 self.data.Close, self.bb_len)
        self.bb_upper = self.bb_mid + self.bb_stddev * self.bb_std
        self.bb_lower = self.bb_mid - self.bb_stddev * self.bb_std

    def next(self) -> None:
        if len(self.data.Close) < max(self.bb_len, self.rsi_len) + 1:
            return
        close_prev = self.data.Close[-2]
        rsi_prev = self.rsi[-2]
        lower_prev = self.bb_lower[-2]
        mid_prev = self.bb_mid[-2]
        if not self.position:
            if rsi_prev <= self.rsi_buy and close_prev <= lower_prev:
                if self.tp_mid:
                    self.buy(sl=self.data.Close[-1] * (1 - self.sl_pct))
                else:
                    sl = self.data.Close[-1] * (1 - self.sl_pct)
                    tp = self.data.Close[-1] * 1.015
                    self.buy(sl=sl, tp=tp)
        else:
            if self.tp_mid:
                if close_prev >= mid_prev:
                    self.position.close()
