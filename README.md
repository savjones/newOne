# RBI Backtesting System

Lightweight research → backtest → implement framework for crypto strategies
using local OHLCV CSV files and `backtesting.py`.

## Layout
```
configs/
  global.yaml         # fees and defaults
backtests/
  loader.py           # CSV schema detection + normalisation
  metrics.py          # expectancy, PF, Sortino, ...
  run.py              # CLI runner
strategies/
  base.py
  breakout.py
  momentum.py
  mean_reversion.py
  pairs_trading.py    # placeholder
  ultimate_oscillator.py
  microstructure.py
utils/
  ta.py               # lightweight indicators
```

## Usage

Backtest all strategies on BTC CSVs with light optimisation:
```bash
python backtests/run.py --strategies all --pattern "*BTC*" --timeframes 1h,1d --opt light
```

Run subset and print trades:
```bash
python backtests/run.py --strategies momentum,breakout --print-trades
```

By default results are only printed to the terminal. Pass `--save-artifacts`
to store `trades.csv` and `metrics.json` for each run.

## Requirements

- Python 3.11+
- pandas
- numpy
- backtesting
- pyyaml
