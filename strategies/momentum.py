"""Time-series momentum using dual moving averages with ADX confirmation."""
from __future__ import annotations

from .base import BaseStrategy
from utils.ta import adx, atr, sma


class Momentum(BaseStrategy):
    DEFAULT_PARAMS = dict(fast=20, slow=50, adx_period=14, adx_min=20,
                          sl_atr=2.0, tp_atr=4.0)
    PARAM_BOUNDS = dict(fast=(10, 40), slow=(40, 100), adx_period=(10, 20),
                        adx_min=(10, 40), sl_atr=(1.0, 3.0), tp_atr=(2.0, 6.0))

    def init(self) -> None:
        self.sma_fast = self.I(sma, self.data.Close, self.fast)
        self.sma_slow = self.I(sma, self.data.Close, self.slow)
        self.adx = self.I(adx, self.data.High, self.data.Low, self.data.Close,
                          self.adx_period)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close,
                          self.adx_period)

    def next(self) -> None:
        if len(self.data.Close) < max(self.slow, self.adx_period) + 1:
            return
        close_prev = self.data.Close[-2]
        sma_fast_prev = self.sma_fast[-2]
        sma_slow_prev = self.sma_slow[-2]
        adx_prev = self.adx[-2]
        atr_prev = self.atr[-2]
        if not self.position:
            if sma_fast_prev > sma_slow_prev and adx_prev >= self.adx_min:
                sl = self.data.Close[-1] - self.sl_atr * atr_prev
                tp = self.data.Close[-1] + self.tp_atr * atr_prev
                self.buy(sl=sl, tp=tp)
        else:
            if sma_fast_prev < sma_slow_prev:
                self.position.close()
