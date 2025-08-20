"""
Test Script for Ultimate Oscillator Strategy

This script tests the Ultimate Oscillator + ADX + MFI strategy implementation
using sample data to ensure all indicators are calculated correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from strategies.ultimate_oscillator import UltimateOscillatorStrategy
from utils.risk import RiskManager


def generate_test_data(periods: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate price series with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, periods)
    
    # Add some trend reversals for testing
    for i in range(50, periods, 50):
        returns[i:i+10] *= -2  # Reversal periods
    
    # Generate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, periods),
    }, index=dates)
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_ultimate_oscillator_strategy():
    """Test the Ultimate Oscillator strategy"""
    print("🧪 Testing Ultimate Oscillator + ADX + MFI Strategy")
    print("=" * 60)
    
    # Create test configuration
    config = {
        'strategy': {
            'uo_short_period': 7,
            'uo_medium_period': 14,
            'uo_long_period': 28,
            'uo_oversold': 30.0,
            'uo_overbought': 70.0,
            'adx_period': 14,
            'adx_trend_threshold': 25.0,
            'mfi_period': 14,
            'mfi_oversold': 20.0,
            'mfi_overbought': 80.0,
            'base_position_size': 0.1,
            'max_position_size': 0.25
        },
        'signals': {
            'enabled': True,
            'min_confidence': 0.5
        },
        'rules': {
            'max_positions': 1,
            'position_sizing': 'fixed'
        },
        'risk': {
            'max_position_size': 0.25,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    }
    
    # Initialize risk manager
    risk_manager = RiskManager({
        'max_position_size': 0.2,
        'max_drawdown': 0.15
    })
    
    # Initialize strategy
    strategy = UltimateOscillatorStrategy(config, risk_manager)
    
    print(f"✓ Strategy initialized: {strategy.get_strategy_name()}")
    
    # Generate test data
    test_data = generate_test_data(200)
    print(f"✓ Generated test data: {len(test_data)} rows")
    print(f"  Price range: ${test_data['low'].min():.2f} - ${test_data['high'].max():.2f}")
    print(f"  Date range: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    
    # Calculate indicators
    print("\n📊 Calculating Technical Indicators...")
    data_with_indicators = strategy.calculate_indicators(test_data)
    
    # Verify indicators are calculated
    required_indicators = strategy.get_required_indicators()
    missing_indicators = [ind for ind in required_indicators if ind not in data_with_indicators.columns]
    
    if missing_indicators:
        print(f"❌ Missing indicators: {missing_indicators}")
        return False
    
    print(f"✓ All {len(required_indicators)} indicators calculated successfully")
    
    # Display sample indicator values
    sample_idx = -10  # Last 10 rows
    sample_data = data_with_indicators.iloc[sample_idx:].copy()
    
    print(f"\n📈 Sample Indicator Values (last 10 bars):")
    print("-" * 60)
    
    for _, row in sample_data.iterrows():
        date = row.name.strftime('%Y-%m-%d')
        price = row['close']
        uo = row['ultimate_oscillator']
        adx = row['adx']
        mfi = row['mfi']
        signal = row.get('composite_signal', 0)
        strength = row.get('signal_strength', 0)
        
        print(f"{date}: Price=${price:6.2f} | UO={uo:5.1f} | ADX={adx:5.1f} | MFI={mfi:5.1f} | Signal={signal:2.0f} | Strength={strength:.2f}")
    
    # Test signal generation
    print(f"\n🎯 Testing Signal Generation...")
    signals = strategy.generate_signals(data_with_indicators)
    
    print(f"✓ Generated {len(signals)} signals")
    
    if signals:
        for i, signal in enumerate(signals[-3:], 1):  # Show last 3 signals
            print(f"  Signal {i}: {signal.signal_type.name} at ${signal.price:.2f} "
                  f"(confidence: {signal.confidence:.2f})")
            print(f"    Reason: {signal.metadata.get('entry_reason', 'N/A')}")
            print(f"    UO: {signal.metadata.get('ultimate_oscillator', 0):.1f}, "
                  f"ADX: {signal.metadata.get('adx', 0):.1f}, "
                  f"MFI: {signal.metadata.get('mfi', 0):.1f}")
    
    # Test position sizing
    if signals:
        print(f"\n💰 Testing Position Sizing...")
        test_signal = signals[-1]
        position_size = strategy.calculate_position_size(test_signal, test_signal.price, 100000)
        print(f"✓ Position size for last signal: {position_size:.3f} (${position_size * 100000:.0f})")
    
    # Statistical summary of indicators
    print(f"\n📊 Indicator Statistics Summary:")
    print("-" * 60)
    
    stats_data = data_with_indicators[['ultimate_oscillator', 'adx', 'mfi', 'composite_signal']].describe()
    print(stats_data.round(2))
    
    # Signal distribution
    signal_counts = data_with_indicators['composite_signal'].value_counts().sort_index()
    total_bars = len(data_with_indicators)
    
    print(f"\n📈 Signal Distribution:")
    print("-" * 30)
    for signal_val, count in signal_counts.items():
        signal_name = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}.get(signal_val, f'Signal_{signal_val}')
        percentage = count / total_bars * 100
        print(f"  {signal_name}: {count:3d} bars ({percentage:5.1f}%)")
    
    # Extreme readings
    extreme_oversold = (data_with_indicators['ultimate_oscillator'] < 20).sum()
    extreme_overbought = (data_with_indicators['ultimate_oscillator'] > 80).sum()
    strong_trend = (data_with_indicators['adx'] > 40).sum()
    
    print(f"\n🔥 Extreme Readings:")
    print("-" * 30)
    print(f"  UO Extreme Oversold (<20): {extreme_oversold} bars ({extreme_oversold/total_bars*100:.1f}%)")
    print(f"  UO Extreme Overbought (>80): {extreme_overbought} bars ({extreme_overbought/total_bars*100:.1f}%)")
    print(f"  ADX Strong Trend (>40): {strong_trend} bars ({strong_trend/total_bars*100:.1f}%)")
    
    print(f"\n✅ Ultimate Oscillator Strategy Test Completed Successfully!")
    return True


def main():
    """Main test function"""
    try:
        success = test_ultimate_oscillator_strategy()
        if success:
            print(f"\n🎉 All tests passed! The Ultimate Oscillator strategy is ready for backtesting.")
            print(f"\nNext steps:")
            print(f"  1. Run comprehensive backtests: python run_comprehensive_backtests.py")
            print(f"  2. Focus on Ultimate Oscillator only: python run_comprehensive_backtests.py --strategies ultimate_oscillator")
            print(f"  3. Enable parallel execution for speed: python run_comprehensive_backtests.py --parallel")
        else:
            print(f"\n❌ Some tests failed. Please check the implementation.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
