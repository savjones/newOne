"""Donchian breakout strategy with ATR filter."""
from __future__ import annotations

from .base import BaseStrategy
from utils.ta import atr, donchian_high, donchian_low, rolling_median


class Breakout(BaseStrategy):
    """Trade breakouts gated by volatility."""

    DEFAULT_PARAMS = dict(channel=20, atr_period=14, atr_mult=1.5,
                          sl_atr=2.0, tp_atr=3.0)
    PARAM_BOUNDS = dict(channel=(10, 40), atr_period=(10, 20),
                        atr_mult=(1.0, 2.5), sl_atr=(1.0, 3.0),
                        tp_atr=(2.0, 5.0))

    def init(self) -> None:
        self.hh = self.I(donchian_high, self.data.High, self.channel)
        self.ll = self.I(donchian_low, self.data.Low, self.channel)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close,
                          self.atr_period)
        self.atr_med = self.I(rolling_median, self.atr, 100)

    def next(self) -> None:
        if len(self.data.Close) < max(self.channel, self.atr_period, 100) + 1:
            return
        close_prev = self.data.Close[-2]
        hh_prev = self.hh[-2]
        ll_prev = self.ll[-2]
        atr_prev = self.atr[-2]
        atr_med_prev = self.atr_med[-2]
        if not self.position:
            if close_prev > hh_prev and atr_prev > atr_med_prev:
                sl = self.data.Close[-1] - self.sl_atr * atr_prev
                tp = self.data.Close[-1] + self.tp_atr * atr_prev
                self.buy(sl=sl, tp=tp)
        else:
            if close_prev < ll_prev:
                self.position.close()
