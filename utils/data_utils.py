"""
Data utility functions for the crypto trading bot.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pytz
from pathlib import Path


def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    volume_price = typical_price * df['volume']
    
    if period is None:
        # Cumulative VWAP
        return volume_price.cumsum() / df['volume'].cumsum()
    else:
        # Rolling VWAP
        return volume_price.rolling(period).sum() / df['volume'].rolling(period).sum()


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    
    return {
        'upper': sma + (std * std_dev),
        'middle': sma,
        'lower': sma - (std * std_dev)
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def clean_ohlcv_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    fill_gaps: bool = True,
    min_volume: float = 0.0,
    max_price_change: float = 0.5
) -> pd.DataFrame:
    """Clean OHLCV data by removing anomalies and filling gaps.
    
    Args:
        df: DataFrame with OHLCV data
        remove_duplicates: Whether to remove duplicate timestamps
        fill_gaps: Whether to fill gaps in data
        min_volume: Minimum volume threshold
        max_price_change: Maximum allowed price change between consecutive rows
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in df_clean.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    
    # Sort by timestamp
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # Remove rows with invalid OHLCV data
    df_clean = df_clean[
        (df_clean['open'] > 0) &
        (df_clean['high'] > 0) &
        (df_clean['low'] > 0) &
        (df_clean['close'] > 0) &
        (df_clean['volume'] >= min_volume)
    ]
    
    # Remove rows with extreme price changes
    if len(df_clean) > 1:
        price_changes = np.abs(df_clean['close'].pct_change())
        df_clean = df_clean[price_changes <= max_price_change]
    
    # Fill gaps if requested
    if fill_gaps and len(df_clean) > 1:
        # Detect time interval
        time_diff = df_clean['timestamp'].diff().median()
        
        # Create complete time range
        start_time = df_clean['timestamp'].min()
        end_time = df_clean['timestamp'].max()
        complete_range = pd.date_range(start=start_time, end=end_time, freq=time_diff)
        
        # Reindex and forward fill
        df_clean = df_clean.set_index('timestamp').reindex(complete_range)
        df_clean = df_clean.fillna(method='ffill')
        df_clean = df_clean.reset_index().rename(columns={'index': 'timestamp'})
    
    return df_clean


def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate OHLCV data for consistency.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    if errors:
        return False, errors
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        errors.append("Timestamp column must be datetime")
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column {col} must be numeric")
    
    # Check logical consistency
    if len(df) > 0:
        # High >= Low
        if not (df['high'] >= df['low']).all():
            errors.append("High price must be >= Low price")
        
        # High >= Open, Close
        if not (df['high'] >= df['open']).all():
            errors.append("High price must be >= Open price")
        if not (df['high'] >= df['close']).all():
            errors.append("High price must be >= Close price")
        
        # Low <= Open, Close
        if not (df['low'] <= df['open']).all():
            errors.append("Low price must be <= Open price")
        if not (df['low'] <= df['close']).all():
            errors.append("Low price must be <= Close price")
        
        # Volume >= 0
        if not (df['volume'] >= 0).all():
            errors.append("Volume must be >= 0")
    
    return len(errors) == 0, errors


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    agg_method: str = 'ohlc'
) -> pd.DataFrame:
    """Resample OHLCV data to different frequency.
    
    Args:
        df: DataFrame with OHLCV data
        freq: Target frequency (e.g., '1H', '4H', '1D')
        agg_method: Aggregation method ('ohlc' or 'last')
        
    Returns:
        Resampled DataFrame
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.set_index('timestamp')
    
    if agg_method == 'ohlc':
        # OHLC aggregation
        resampled = df_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    elif agg_method == 'last':
        # Last value aggregation
        resampled = df_copy.resample(freq).last()
    else:
        raise ValueError("agg_method must be 'ohlc' or 'last'")
    
    # Remove rows with all NaN values
    resampled = resampled.dropna()
    
    return resampled.reset_index()


def calculate_technical_indicators(
    df: pd.DataFrame,
    indicators: List[str] = None
) -> pd.DataFrame:
    """Calculate technical indicators for OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicators to calculate. If None, calculates all.
        
    Returns:
        DataFrame with added technical indicators
    """
    if indicators is None:
        indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
            'atr', 'vwap', 'stochastic', 'williams_r'
        ]
    
    df_indicators = df.copy()
    
    for indicator in indicators:
        if indicator == 'sma':
            df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
            df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
        
        elif indicator == 'ema':
            df_indicators['ema_12'] = df_indicators['close'].ewm(span=12).mean()
            df_indicators['ema_26'] = df_indicators['close'].ewm(span=26).mean()
        
        elif indicator == 'rsi':
            delta = df_indicators['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_indicators['rsi'] = 100 - (100 / (1 + rs))
        
        elif indicator == 'macd':
            ema_12 = df_indicators['close'].ewm(span=12).mean()
            ema_26 = df_indicators['close'].ewm(span=26).mean()
            df_indicators['macd'] = ema_12 - ema_26
            df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9).mean()
            df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
        
        elif indicator == 'bollinger_bands':
            sma_20 = df_indicators['close'].rolling(window=20).mean()
            std_20 = df_indicators['close'].rolling(window=20).std()
            df_indicators['bb_upper'] = sma_20 + (std_20 * 2)
            df_indicators['bb_middle'] = sma_20
            df_indicators['bb_lower'] = sma_20 - (std_20 * 2)
            df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
        
        elif indicator == 'atr':
            high_low = df_indicators['high'] - df_indicators['low']
            high_close = np.abs(df_indicators['high'] - df_indicators['close'].shift())
            low_close = np.abs(df_indicators['low'] - df_indicators['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df_indicators['atr'] = true_range.rolling(window=14).mean()
        
        elif indicator == 'vwap':
            typical_price = (df_indicators['high'] + df_indicators['low'] + df_indicators['close']) / 3
            df_indicators['vwap'] = (typical_price * df_indicators['volume']).rolling(window=20).sum() / df_indicators['volume'].rolling(window=20).sum()
        
        elif indicator == 'stochastic':
            lowest_low = df_indicators['low'].rolling(window=14).min()
            highest_high = df_indicators['high'].rolling(window=14).max()
            df_indicators['stoch_k'] = 100 * ((df_indicators['close'] - lowest_low) / (highest_high - lowest_low))
            df_indicators['stoch_d'] = df_indicators['stoch_k'].rolling(window=3).mean()
        
        elif indicator == 'williams_r':
            highest_high = df_indicators['high'].rolling(window=14).max()
            lowest_low = df_indicators['low'].rolling(window=14).min()
            df_indicators['williams_r'] = -100 * ((highest_high - df_indicators['close']) / (highest_high - lowest_low))
    
    return df_indicators


def detect_market_regime(
    df: pd.DataFrame,
    lookback: int = 20,
    volatility_threshold: float = 0.02,
    trend_threshold: float = 0.01
) -> pd.DataFrame:
    """Detect market regime (trending, ranging, high volatility).
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        lookback: Lookback period for regime detection
        volatility_threshold: Threshold for high volatility regime
        trend_threshold: Threshold for trend detection
        
    Returns:
        DataFrame with regime classification
    """
    df_regime = df.copy()
    
    # Calculate volatility (rolling standard deviation of returns)
    returns = df_regime['close'].pct_change()
    df_regime['volatility'] = returns.rolling(window=lookback).std()
    
    # Calculate trend strength (linear regression slope)
    def calculate_trend_strength(prices):
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope / prices.iloc[0]  # Normalize by initial price
    
    df_regime['trend_strength'] = df_regime['close'].rolling(window=lookback).apply(calculate_trend_strength)
    
    # Classify regime
    def classify_regime(row):
        if pd.isna(row['volatility']) or pd.isna(row['trend_strength']):
            return 'UNKNOWN'
        
        if row['volatility'] > volatility_threshold:
            return 'HIGH_VOL'
        elif abs(row['trend_strength']) > trend_threshold:
            return 'TRENDING'
        else:
            return 'RANGING'
    
    df_regime['market_regime'] = df_regime.apply(classify_regime, axis=1)
    
    return df_regime


def save_to_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    compression: str = 'snappy'
) -> None:
    """Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        compression: Compression algorithm
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(filepath, compression=compression, index=False)


def load_from_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet format.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(filepath)


def create_feature_engineer(
    df: pd.DataFrame,
    feature_config: Dict
) -> pd.DataFrame:
    """Create engineered features for ML models.
    
    Args:
        df: DataFrame with OHLCV data
        feature_config: Configuration for feature engineering
        
    Returns:
        DataFrame with engineered features
    """
    df_features = df.copy()
    
    # Price-based features
    if feature_config.get('price_features', True):
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        df_features['price_change'] = df_features['close'] - df_features['close'].shift(1)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df_features[f'returns_{window}d'] = df_features['returns'].rolling(window).mean()
            df_features[f'volatility_{window}d'] = df_features['returns'].rolling(window).std()
    
    # Volume-based features
    if feature_config.get('volume_features', True):
        df_features['volume_ma'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma']
        df_features['volume_change'] = df_features['volume'].pct_change()
    
    # Technical indicator features
    if feature_config.get('technical_features', True):
        df_features = calculate_technical_indicators(df_features)
    
    # Time-based features
    if feature_config.get('time_features', True):
        df_features['hour'] = pd.to_datetime(df_features['timestamp']).dt.hour
        df_features['day_of_week'] = pd.to_datetime(df_features['timestamp']).dt.dayofweek
        df_features['month'] = pd.to_datetime(df_features['timestamp']).dt.month
    
    # Lag features
    if feature_config.get('lag_features', True):
        for lag in [1, 2, 3, 5]:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    return df_features
