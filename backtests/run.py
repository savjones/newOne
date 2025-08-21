"""CLI runner for the RBI backtesting system."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from backtesting import Backtest

# Allow running as a script from repository root
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtests.loader import load_csv, parse_file_info
from backtests.metrics import extract_metrics
from strategies.breakout import Breakout
from strategies.momentum import Momentum
from strategies.mean_reversion import MeanReversion
from strategies.pairs_trading import PairsTrading
from strategies.ultimate_oscillator import UltimateOscillator
from strategies.microstructure import Microstructure

STRATEGIES = {
    'breakout': Breakout,
    'momentum': Momentum,
    'mean_reversion': MeanReversion,
    'pairs_trading': PairsTrading,
    'ultimate_oscillator': UltimateOscillator,
    'microstructure': Microstructure,
}


def _run_bt(df: pd.DataFrame, strat_cls, cash: float, commission: float,
            slippage: float, opt: str):
    bt = Backtest(df, strat_cls, cash=cash, commission=commission,
                  slippage=slippage, trade_on_close=False, hedging=False,
                  exclusive_orders=True)
    if opt == 'light' and strat_cls.PARAM_BOUNDS:
        best = None
        best_metric = -np.inf
        for _ in range(20):
            params = {}
            for k, (lo, hi) in strat_cls.PARAM_BOUNDS.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[k] = random.randint(lo, hi)
                else:
                    params[k] = random.uniform(lo, hi)
            stats = bt.run(**params)
            metric = stats.get('Sortino Ratio', 0)
            if metric > best_metric:
                best_metric = metric
                best = stats
        stats = best
    else:
        stats = bt.run(**strat_cls.DEFAULT_PARAMS)
    trades = getattr(stats, '_trades', pd.DataFrame())
    return stats, trades


def main() -> None:
    parser = argparse.ArgumentParser(description='RBI backtesting runner')
    parser.add_argument('--data-dir', default='data/csv', help='Directory with CSVs')
    parser.add_argument('--strategies', default='all',
                        help='Comma list or "all"')
    parser.add_argument('--pattern', default='*', help='Glob pattern for files')
    parser.add_argument('--timeframes', default='', help='Comma separated filter')
    parser.add_argument('--opt', choices=['none', 'light'], default='none')
    parser.add_argument('--print-trades', action='store_true')
    parser.add_argument('--save-artifacts', action='store_true')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path('configs/global.yaml').read_text())
    cash = float(cfg.get('start_cash', 100000))
    commission = cfg.get('commission_bps', 0) / 10000
    slippage = cfg.get('slippage_bps', 0) / 10000

    selected = list(STRATEGIES.keys()) if args.strategies == 'all' \
        else [s.strip() for s in args.strategies.split(',')]

    files = [f for f in Path(args.data_dir).glob(args.pattern)
             if f.suffix.lower() == '.csv']
    if args.timeframes:
        tfs = set(t.strip() for t in args.timeframes.split(','))
        files = [f for f in files if parse_file_info(f)[1] in tfs]

    rows = []
    for path in files:
        df = load_csv(path)
        symbol, tf = parse_file_info(path)
        for name in selected:
            strat_cls = STRATEGIES[name]
            stats, trades = _run_bt(df, strat_cls, cash, commission,
                                    slippage, args.opt)
            metrics = extract_metrics(stats)
            metrics.update(dict(symbol=symbol, timeframe=tf,
                                strategy=name))
            rows.append(metrics)
            if args.print_trades and not trades.empty:
                print(f"--- {symbol} {tf} {name} trades ---")
                print(trades.head(3).to_string(index=False))
                if len(trades) > 6:
                    print('...')
                print(trades.tail(3).to_string(index=False))
            if args.save_artifacts:
                out_dir = Path('backtest_outputs')
                out_dir.mkdir(exist_ok=True)
                base = f"{symbol}_{tf}_{name}"
                trades.to_csv(out_dir / f"{base}_trades.csv", index=False)
                with open(out_dir / f"{base}_metrics.json", 'w') as fh:
                    json.dump(metrics, fh, indent=2)

    if rows:
        dfm = pd.DataFrame(rows)
        dfm.sort_values(['expectancy', 'profit_factor', 'sortino'],
                         ascending=False, inplace=True)
        print('=' * 80)
        print(f"Backtest Summary ({pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC})  "
              f"data={args.data_dir}   strategies={args.strategies}   opt={args.opt}")
        print('-' * 80)
        cols = ['symbol', 'timeframe', 'strategy', 'trades', 'expectancy',
                'profit_factor', 'sortino', 'max_drawdown_pct', 'exposure_pct',
                'win_rate_pct', 'cagr', 'equity_final_pct']
        print(dfm[cols].to_string(index=False))
        print('-' * 80)
        picks = dfm[(dfm.trades >= 30) & (dfm.exposure_pct.between(5, 95))
                    & (dfm.max_drawdown_pct <= 50)].head(5)
        if not picks.empty:
            print('Top Picks (trades≥30, 5%≤expo≤95%, maxDD≤50%):')
            for i, row in enumerate(picks.itertuples(index=False), 1):
                print(f"{i}) {row.symbol} {row.timeframe} {row.strategy}  "
                      f"Expect={row.expectancy:.2f}  PF={row.profit_factor:.2f}  "
                      f"Sortino={row.sortino:.2f}  Trades={row.trades}  "
                      f"DD={row.max_drawdown_pct:.1f}%")
        print('=' * 80)


if __name__ == '__main__':
    main()
