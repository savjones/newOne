"""Common helpers for strategy implementations."""
from __future__ import annotations

from typing import Any, Dict

from backtesting import Strategy


class BaseStrategy(Strategy):
    """Base class to expose default parameter containers.

    Individual strategies override :pyattr:`DEFAULT_PARAMS` and
    :pyattr:`PARAM_BOUNDS` for use by the runner and simple optimisation.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {}
    PARAM_BOUNDS: Dict[str, tuple] = {}
