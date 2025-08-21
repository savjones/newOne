"""Microstructure proxy strategy using rolling VWAP and volume surge."""
from __future__ import annotations

from .base import BaseStrategy
from utils.ta import atr, rolling_vwap, sma


class Microstructure(BaseStrategy):
    DEFAULT_PARAMS = dict(vwap_len=20, vol_mult=1.2, sl_atr=1.8, tp_atr=2.5)
    PARAM_BOUNDS = dict(vwap_len=(10, 40), vol_mult=(1.0, 2.0),
                        sl_atr=(1.0, 3.0), tp_atr=(2.0, 4.0))

    def init(self) -> None:
        self.vwap = self.I(rolling_vwap, self.data.Close, self.data.Volume,
                            self.vwap_len)
        self.vol_ma = self.I(sma, self.data.Volume, self.vwap_len)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close,
                          self.vwap_len)

    def next(self) -> None:
        if len(self.data.Close) < self.vwap_len + 1:
            return
        close_prev = self.data.Close[-2]
        vwap_prev = self.vwap[-2]
        vol_prev = self.data.Volume[-2]
        vol_ma_prev = self.vol_ma[-2]
        atr_prev = self.atr[-2]
        if not self.position:
            if close_prev > vwap_prev and vol_prev > self.vol_mult * vol_ma_prev:
                sl = self.data.Close[-1] - self.sl_atr * atr_prev
                tp = self.data.Close[-1] + self.tp_atr * atr_prev
                self.buy(sl=sl, tp=tp)
        else:
            if close_prev < vwap_prev:
                self.position.close()
