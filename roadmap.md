# Crypto AI Trading Bot — 0 to Hero Roadmap

## Phase 1: Repo Setup & Foundations
- **Repo structure**
  - `/data/` → ingestion, cleaning, storage
  - `/strategies/` → momentum, mean reversion, breakout, microstructure, pairs
  - `/ml/` → feature engineering, labeling, models
  - `/backtests/` → backtest engines, configs, walk-forward harness
  - `/docs/` → roadmap.md, architecture.md, strategy_docs/
  - `/utils/` → logging, config management, risk utils
- **Initial stack**
  - Python 3.11+
  - pandas, numpy, ta-lib, scikit-learn, statsmodels, xgboost, pytorch (later)
  - backtrader or vectorbt (for prototyping), custom harness later
- **Core principles**
  - Clean code (PEP8, type hints, docstrings)
  - Modular, each strategy/test isolated
  - Config-driven (YAML/JSON configs, not hardcoded)

---

## Phase 2: Core “Tried & True” Strategies
Implement rule-based versions first, clean & tested:
1. **Time-series momentum (TSMOM)**
   - MA crossover / breakout logic
   - Add volatility-managed scaling
2. **Mean reversion to VWAP/Bollinger**
   - Z-score thresholds, fade to VWAP
   - Configurable windows
3. **Breakout + volume/volatility confirmation**
   - Donchian/Keltner channel
   - Volume spike or volatility filter
4. **Microstructure order flow**
   - Order Flow Imbalance (OFI), book imbalance, VWAP drift
   - Short-horizon signals
5. **Pairs trading (stat arb)**
   - Cointegration tests
   - Spread z-score entry/exit

Each strategy has:
- Config file with parameters
- Documentation page (`docs/strategy_docs/...`)
- Unit test + example backtest notebook

---

## Phase 3: Backtesting & Risk Framework
- **Backtester**: event-driven, supports fees, slippage, multiple timeframes
- **Risk module**:
  - Max % per position
  - Portfolio-level VaR/Drawdown caps
  - Stop/Take-profit logic
- **Data pipeline**:
  - Crypto OHLCV + L2/L3 order book (Binance, Coinbase, Bybit, etc.)
  - Cleaning: deduplication, uniform UTC timestamps, gap fills
  - Store as Parquet for speed
- **Evaluation**:
  - Walk-forward, not single backtest
  - Tearsheet metrics (Sharpe, Sortino, drawdown, turnover, hit-rate, capacity)

---

## Phase 4: AI/ML Layering
- **Meta-labeling** (Lopez de Prado triple-barrier)
  - Classify “signal success/failure” for each base strategy trade
- **Ensemble confidence**
  - Random Forest / XGBoost over combined features + base signals
  - Outputs calibrated p(win) → drives position sizing
- **Regime classifier**
  - Predict trending vs ranging vs high-vol regimes (filters which strategy to activate)
- **Advanced models**
  - LSTM for sequential order-flow patterns
  - Ensemble stacking of tree + NN models

---

## Phase 5: Live / Deployment
- **Execution layer**:
  - Exchange adapters (Binance/CCXT first)
  - Smart order routing: limit vs market vs TWAP
- **Risk live**: enforce per-trade + portfolio caps
- **Monitoring**:
  - PnL attribution per strategy
  - Live vs backtest drift analysis
- **Shadow trading**:
  - Run ML predictions in parallel but don’t execute until stable

---

# 🎯 Final Deliverable
- **A modular bot repo** with:
  - At least 5 core strategies
  - Unified backtest/risk engine
  - ML-based meta-labeling & ensemble logic
  - Documentation for every module
- **Robust, clean, extensible** → new strats & markets easy to plug in.
