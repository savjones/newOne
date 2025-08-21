"""CSV loader with basic schema detection and normalisation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _detect_schema(columns: list[str]) -> str:
    cols = {c.lower() for c in columns}
    if {'timestamp', 'open', 'high', 'low', 'close', 'volume'} <= cols:
        return 'binance'
    if {'unix', 'open', 'high', 'low', 'close', 'volume btc'} <= cols:
        return 'cryptodatadownload'
    if {'date', 'open', 'high', 'low', 'close', 'volume'} <= cols:
        return 'yahoo'
    return 'generic'


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file and normalise to standard OHLCV columns."""
    df = pd.read_csv(path)
    schema = _detect_schema(list(df.columns))

    if schema == 'binance':
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        out = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    elif schema == 'cryptodatadownload':
        df['datetime'] = pd.to_datetime(df['unix'], unit='s', utc=True)
        vol_col = 'Volume USDT' if 'Volume USDT' in df.columns else 'Volume BTC'
        out = df[['datetime', 'open', 'high', 'low', 'close', vol_col]].copy()
        out.rename(columns={vol_col: 'volume'}, inplace=True)
    elif schema == 'yahoo':
        df['datetime'] = pd.to_datetime(df['Date'], utc=True)
        out = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        out.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    else:  # generic
        # assume first six columns correspond to datetime/open/high/low/close/volume
        out = df.iloc[:, :6].copy()
        out.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        out['datetime'] = pd.to_datetime(out['datetime'], utc=True)

    out = out.dropna()
    out['volume'] = out['volume'].replace(0, np.finfo(float).eps)
    out = out.astype({'open': float, 'high': float, 'low': float,
                      'close': float, 'volume': float})
    out = out.drop_duplicates(subset='datetime').set_index('datetime')
    out.sort_index(inplace=True)
    return out


def parse_file_info(path: Path) -> Tuple[str, str]:
    """Return ``(symbol, timeframe)`` parsed from a filename."""
    name = path.stem
    token_re = re.compile(r'(\d+[smhdw])')
    tokens = re.split(r'[-_]', name)
    symbol = tokens[0].upper()
    timeframe = 'unknown'
    for t in tokens:
        m = token_re.fullmatch(t)
        if m:
            timeframe = m.group(1)
            break
    return symbol, timeframe
