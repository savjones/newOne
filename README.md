# Crypto AI Trading Bot

A comprehensive, production-ready crypto trading bot that evolves from rule-based strategies into AI/ML-driven trading.

## 🚀 Features

- **6 Core Trading Strategies**: Momentum, Mean Reversion, Breakout, Microstructure, Pairs Trading, Ultimate Oscillator + ADX + MFI
- **Unified Backtest Engine**: Event-driven with slippage, fees, and risk management
- **AI/ML Layer**: Meta-labeling, ensemble methods, regime classification, and LSTM models
- **Multi-Exchange Support**: Binance, Coinbase, Bybit with clean data ingestion
- **Production Ready**: Clean code, type hints, comprehensive testing, and documentation

## 📁 Project Structure

```
├── data/                   # Data ingestion, cleaning, storage, CSV loader
├── strategies/            # Trading strategy implementations (6 strategies)
├── ml/                   # Machine learning models and features
├── backtests/            # Backtesting engines, configs, and pipeline
├── docs/                 # Documentation and strategy guides
├── utils/                # Utilities and helpers
├── tests/                # Comprehensive test suite
├── notebooks/            # Example notebooks and tutorials
└── configs/              # Configuration files (YAML-based)
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd crypto-trading-bot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## 🚀 Quick Start

### New CSV-Based Backtesting Pipeline

1. **Test the Ultimate Oscillator strategy**:
```bash
python test_ultimate_oscillator.py
```

2. **Run comprehensive backtests on all your CSV data**:
```bash
python run_comprehensive_backtests.py
```

3. **Test specific strategies with parallel execution**:
```bash
python run_comprehensive_backtests.py --strategies ultimate_oscillator momentum --parallel
```

4. **Customize parameters**:
```bash
python run_comprehensive_backtests.py --initial-capital 50000 --commission 0.002 --parallel --max-workers 4
```

### Legacy Examples

5. **Run a simple backtest**:
```bash
python -m backtests.run_backtest --strategy momentum --symbol BTCUSDT
```

6. **Train ML models**:
```bash
python -m ml.train_models --strategy all --data_path data/processed/
```

7. **Live trading** (after configuration):
```bash
python -m live.trading_bot --config configs/live_config.yaml
```

## 📊 Strategies

### 1. Time-Series Momentum (TSMOM)
- Moving average crossovers
- Volatility-managed position sizing
- Trend following with momentum confirmation

### 2. Mean Reversion to VWAP/Bollinger
- Z-score based entry/exit signals
- VWAP and Bollinger Band mean reversion
- Configurable lookback windows

### 3. Breakout with Volume Confirmation
- Donchian/Keltner channel breakouts
- Volume spike validation
- Volatility filters

### 4. Microstructure Order Flow
- Order Flow Imbalance (OFI)
- Book imbalance signals
- VWAP drift detection

### 5. Pairs Trading
- Cointegration-based pair selection
- Spread z-score entry/exit
- Statistical arbitrage

### 6. Ultimate Oscillator + ADX + MFI ⭐ **NEW**
- Multi-timeframe momentum oscillator (7/14/28 periods)
- ADX trend strength confirmation
- Money Flow Index volume validation
- Triple indicator alignment for high-probability signals

## 🤖 AI/ML Features

- **Meta-labeling**: Triple-barrier labeling for strategy performance
- **Ensemble Methods**: Random Forest, XGBoost for trade filtering
- **Regime Classification**: Trend vs Range vs High-Volatility detection
- **LSTM Models**: Sequential order flow pattern recognition

## 📈 Backtesting

- Event-driven architecture
- Realistic slippage and fee modeling
- Risk management with drawdown caps
- Walk-forward testing framework
- Comprehensive performance metrics

## 🔧 Configuration

All strategies and components are config-driven using YAML files:

```yaml
# configs/strategies/momentum.yaml
strategy:
  name: "momentum"
  lookback_period: 20
  volatility_window: 30
  position_sizing: "volatility_scaled"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific strategy tests
pytest tests/strategies/test_momentum.py
```

## 📚 Documentation

- **Strategy Documentation**: `docs/strategy_docs/`
- **API Reference**: `docs/api/`
- **Architecture Guide**: `docs/architecture.md`
- **Example Notebooks**: `notebooks/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## ⚠️ Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk. Use at your own risk and never trade with money you cannot afford to lose.

## 🆘 Support

- Create an issue for bugs or feature requests
- Check the documentation in `docs/`
- Review example notebooks in `notebooks/`
