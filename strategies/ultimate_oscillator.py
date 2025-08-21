"""Ultimate Oscillator + ADX + MFI strategy."""
from __future__ import annotations

from .base import BaseStrategy
from utils.ta import adx, atr, money_flow_index, ultimate_oscillator


class UltimateOscillator(BaseStrategy):
    DEFAULT_PARAMS = dict(uo_fast=7, uo_mid=14, uo_slow=28, uo_buy=50,
                          adx_period=14, adx_min=20, mfi_period=14,
                          mfi_min=50, sl_atr=2.0, tp_atr=3.0)
    PARAM_BOUNDS = dict(uo_fast=(5, 10), uo_mid=(10, 20), uo_slow=(20, 40),
                        uo_buy=(40, 60), adx_period=(10, 20), adx_min=(10, 40),
                        mfi_period=(10, 20), mfi_min=(40, 60),
                        sl_atr=(1.0, 3.0), tp_atr=(2.0, 5.0))

    def init(self) -> None:
        self.uo = self.I(ultimate_oscillator, self.data.High, self.data.Low,
                         self.data.Close, self.uo_fast, self.uo_mid,
                         self.uo_slow)
        self.adx = self.I(adx, self.data.High, self.data.Low, self.data.Close,
                          self.adx_period)
        self.mfi = self.I(money_flow_index, self.data.High, self.data.Low,
                          self.data.Close, self.data.Volume, self.mfi_period)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close,
                          self.adx_period)

    def next(self) -> None:
        if len(self.data.Close) < max(self.uo_slow, self.adx_period,
                                      self.mfi_period) + 1:
            return
        uo_prev = self.uo[-2]
        adx_prev = self.adx[-2]
        mfi_prev = self.mfi[-2]
        atr_prev = self.atr[-2]
        if not self.position:
            if uo_prev > self.uo_buy and adx_prev >= self.adx_min \
                    and mfi_prev >= self.mfi_min:
                sl = self.data.Close[-1] - self.sl_atr * atr_prev
                tp = self.data.Close[-1] + self.tp_atr * atr_prev
                self.buy(sl=sl, tp=tp)
        else:
            if uo_prev < 50:
                self.position.close()
