"""
Data cleaning and preprocessing for the crypto trading bot.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

from utils.logging import get_logger
from utils.data_utils import clean_ohlcv_data, validate_ohlcv_data


class DataCleaner:
    """Data cleaning and preprocessing system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize data cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger("data_cleaner")
        
        # Register cleaning functions
        self.cleaning_functions = self._register_cleaning_functions()
    
    def _get_default_config(self) -> Dict:
        """Get default cleaning configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'remove_duplicates': True,
            'fill_gaps': True,
            'min_volume': 0.0,
            'max_price_change': 0.5,
            'outlier_threshold': 3.0,
            'min_data_points': 100,
            'time_alignment': True,
            'timezone_standardization': 'UTC',
            'quality_thresholds': {
                'max_missing_pct': 0.1,
                'max_outlier_pct': 0.05,
                'min_volume_ratio': 0.1
            }
        }
    
    def _register_cleaning_functions(self) -> Dict[str, Callable]:
        """Register available cleaning functions.
        
        Returns:
            Dictionary mapping function names to functions
        """
        return {
            'remove_duplicates': self._remove_duplicates,
            'fill_gaps': self._fill_gaps,
            'remove_outliers': self._remove_outliers,
            'validate_ohlcv': self._validate_ohlcv,
            'standardize_timestamps': self._standardize_timestamps,
            'clean_volume': self._clean_volume,
            'smooth_prices': self._smooth_prices,
            'add_quality_metrics': self._add_quality_metrics
        }
    
    def clean_dataset(
        self,
        df: pd.DataFrame,
        cleaning_steps: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean dataset using specified cleaning steps.
        
        Args:
            df: Input DataFrame
            cleaning_steps: List of cleaning steps to apply
            **kwargs: Additional cleaning parameters
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        if cleaning_steps is None:
            cleaning_steps = list(self.cleaning_functions.keys())
        
        # Initialize cleaning report
        cleaning_report = {
            'original_rows': len(df),
            'original_columns': list(df.columns),
            'cleaning_steps': cleaning_steps,
            'step_results': {},
            'final_rows': len(df),
            'quality_metrics': {},
            'warnings': [],
            'errors': []
        }
        
        df_clean = df.copy()
        
        # Apply cleaning steps
        for step in cleaning_steps:
            if step in self.cleaning_functions:
                try:
                    self.logger.info(f"Applying cleaning step: {step}")
                    step_result = self.cleaning_functions[step](df_clean, **kwargs)
                    
                    if isinstance(step_result, tuple):
                        df_clean, step_info = step_result
                    else:
                        df_clean = step_result
                        step_info = {}
                    
                    cleaning_report['step_results'][step] = {
                        'rows_before': len(df_clean),
                        'rows_after': len(df_clean),
                        'info': step_info
                    }
                    
                except Exception as e:
                    error_msg = f"Error in cleaning step {step}: {e}"
                    self.logger.error(error_msg)
                    cleaning_report['errors'].append(error_msg)
            else:
                warning_msg = f"Unknown cleaning step: {step}"
                self.logger.warning(warning_msg)
                cleaning_report['warnings'].append(warning_msg)
        
        # Update final statistics
        cleaning_report['final_rows'] = len(df_clean)
        cleaning_report['final_columns'] = list(df_clean.columns)
        
        # Calculate quality metrics
        if len(df_clean) > 0:
            cleaning_report['quality_metrics'] = self._calculate_quality_metrics(df_clean)
        
        self.logger.info(f"Cleaning completed: {cleaning_report['original_rows']} -> {cleaning_report['final_rows']} rows")
        
        return df_clean, cleaning_report
    
    def _remove_duplicates(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        
        # Remove exact duplicates
        df_clean = df.drop_duplicates()
        
        # Remove duplicates based on timestamp (keep latest)
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp').drop_duplicates(
                subset=['timestamp'], keep='last'
            )
        
        removed_count = initial_rows - len(df_clean)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate rows")
        
        return df_clean
    
    def _fill_gaps(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fill gaps in time series data.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with gaps filled
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            return df
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Detect time interval
        time_diffs = df_sorted['timestamp'].diff().dropna()
        if len(time_diffs) == 0:
            return df
        
        median_interval = time_diffs.median()
        
        # Create complete time range
        start_time = df_sorted['timestamp'].min()
        end_time = df_sorted['timestamp'].max()
        
        # Generate complete time series
        complete_times = pd.date_range(start=start_time, end=end_time, freq=median_interval)
        
        # Reindex and forward fill
        df_complete = df_sorted.set_index('timestamp').reindex(complete_times)
        
        # Forward fill OHLCV data
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in df_complete.columns:
                df_complete[col] = df_complete[col].fillna(method='ffill')
        
        # Fill other columns with forward fill or interpolation
        for col in df_complete.columns:
            if col not in ohlcv_columns:
                if df_complete[col].dtype in ['float64', 'int64']:
                    df_complete[col] = df_complete[col].interpolate(method='linear')
                else:
                    df_complete[col] = df_complete[col].fillna(method='ffill')
        
        # Reset index
        df_filled = df_complete.reset_index().rename(columns={'index': 'timestamp'})
        
        # Remove rows that couldn't be filled
        df_filled = df_filled.dropna(subset=['open', 'high', 'low', 'close'])
        
        self.logger.info(f"Filled gaps: {len(df)} -> {len(df_filled)} rows")
        
        return df_filled
    
    def _remove_outliers(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Remove statistical outliers.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with outliers removed
        """
        threshold = kwargs.get('outlier_threshold', self.config['outlier_threshold'])
        initial_rows = len(df)
        
        df_clean = df.copy()
        
        # Remove outliers from OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in ohlcv_columns:
            if col in df_clean.columns:
                # Calculate z-scores
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                
                # Remove outliers
                outlier_mask = z_scores > threshold
                df_clean = df_clean[~outlier_mask]
        
        removed_count = initial_rows - len(df_clean)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outlier rows")
        
        return df_clean
    
    def _validate_ohlcv(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Validate OHLCV data consistency.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Validated DataFrame
        """
        is_valid, errors = validate_ohlcv_data(df)
        
        if not is_valid:
            self.logger.warning(f"OHLCV validation errors: {errors}")
            
            # Try to fix common issues
            df_fixed = self._fix_ohlcv_issues(df)
            return df_fixed
        
        return df
    
    def _fix_ohlcv_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common OHLCV data issues.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Fixed DataFrame
        """
        df_fixed = df.copy()
        
        # Fix high < low issues
        high_low_mask = df_fixed['high'] < df_fixed['low']
        if high_low_mask.any():
            # Swap high and low
            temp_high = df_fixed.loc[high_low_mask, 'high'].copy()
            df_fixed.loc[high_low_mask, 'high'] = df_fixed.loc[high_low_mask, 'low']
            df_fixed.loc[high_low_mask, 'low'] = temp_high
        
        # Fix open/close outside high/low range
        open_outside = (df_fixed['open'] > df_fixed['high']) | (df_fixed['open'] < df_fixed['low'])
        close_outside = (df_fixed['close'] > df_fixed['high']) | (df_fixed['close'] < df_fixed['low'])
        
        if open_outside.any():
            # Clip open to high/low range
            df_fixed.loc[open_outside, 'open'] = df_fixed.loc[open_outside, 'open'].clip(
                lower=df_fixed.loc[open_outside, 'low'],
                upper=df_fixed.loc[open_outside, 'high']
            )
        
        if close_outside.any():
            # Clip close to high/low range
            df_fixed.loc[close_outside, 'close'] = df_fixed.loc[close_outside, 'close'].clip(
                lower=df_fixed.loc[close_outside, 'low'],
                upper=df_fixed.loc[close_outside, 'high']
            )
        
        # Fix negative volume
        negative_volume = df_fixed['volume'] < 0
        if negative_volume.any():
            df_fixed.loc[negative_volume, 'volume'] = 0
        
        return df_fixed
    
    def _standardize_timestamps(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Standardize timestamp format and timezone.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with standardized timestamps
        """
        if 'timestamp' not in df.columns:
            return df
        
        df_clean = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']):
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        # Standardize timezone
        target_timezone = kwargs.get('timezone', self.config['timezone_standardization'])
        
        if target_timezone == 'UTC':
            # Convert to UTC
            if df_clean['timestamp'].dt.tz is not None:
                df_clean['timestamp'] = df_clean['timestamp'].dt.tz_convert('UTC')
            else:
                df_clean['timestamp'] = df_clean['timestamp'].dt.tz_localize('UTC')
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        return df_clean
    
    def _clean_volume(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Clean volume data.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with cleaned volume
        """
        if 'volume' not in df.columns:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Remove rows with zero volume
        min_volume = kwargs.get('min_volume', self.config['min_volume'])
        df_clean = df_clean[df_clean['volume'] >= min_volume]
        
        # Remove rows with extreme volume (e.g., > 10x median)
        if len(df_clean) > 0:
            median_volume = df_clean['volume'].median()
            max_volume = median_volume * 10
            df_clean = df_clean[df_clean['volume'] <= max_volume]
        
        removed_count = initial_rows - len(df_clean)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} rows with invalid volume")
        
        return df_clean
    
    def _smooth_prices(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply price smoothing to reduce noise.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with smoothed prices
        """
        window = kwargs.get('smoothing_window', 3)
        
        if window <= 1 or len(df) < window:
            return df
        
        df_smooth = df.copy()
        
        # Apply rolling median to OHLC
        ohlc_columns = ['open', 'high', 'low', 'close']
        for col in ohlc_columns:
            if col in df_smooth.columns:
                df_smooth[col] = df_smooth[col].rolling(window=window, center=True).median()
        
        # Remove NaN values from smoothing
        df_smooth = df_smooth.dropna()
        
        return df_smooth
    
    def _add_quality_metrics(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Add data quality metrics.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with quality metrics
        """
        df_metrics = df.copy()
        
        # Calculate data quality metrics
        if len(df_metrics) > 0:
            # Missing data percentage
            missing_pct = df_metrics.isnull().sum() / len(df_metrics) * 100
            df_metrics.attrs['missing_pct'] = missing_pct.to_dict()
            
            # Data completeness
            df_metrics.attrs['completeness'] = 1 - missing_pct.mean() / 100
            
            # Timestamp consistency
            if 'timestamp' in df_metrics.columns:
                time_diffs = df_metrics['timestamp'].diff().dropna()
                if len(time_diffs) > 0:
                    df_metrics.attrs['timestamp_consistency'] = time_diffs.std().total_seconds()
            
            # Price consistency
            if all(col in df_metrics.columns for col in ['open', 'high', 'low', 'close']):
                price_consistency = (
                    (df_metrics['high'] >= df_metrics['low']).mean() *
                    (df_metrics['high'] >= df_metrics['open']).mean() *
                    (df_metrics['high'] >= df_metrics['close']).mean() *
                    (df_metrics['low'] <= df_metrics['open']).mean() *
                    (df_metrics['low'] <= df_metrics['close']).mean()
                )
                df_metrics.attrs['price_consistency'] = price_consistency
        
        return df_metrics
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall data quality metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of quality metrics
        """
        if len(df) == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['total_rows'] = len(df)
        metrics['total_columns'] = len(df.columns)
        
        # Missing data
        missing_data = df.isnull().sum().sum()
        metrics['missing_data_pct'] = (missing_data / (len(df) * len(df.columns))) * 100
        
        # Duplicate rows
        duplicate_rows = len(df) - len(df.drop_duplicates())
        metrics['duplicate_rows_pct'] = (duplicate_rows / len(df)) * 100
        
        # Data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        metrics['numeric_columns_pct'] = (len(numeric_columns) / len(df.columns)) * 100
        
        # Timestamp consistency
        if 'timestamp' in df.columns:
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                metrics['avg_time_interval'] = time_diffs.mean().total_seconds()
                metrics['time_interval_std'] = time_diffs.std().total_seconds()
        
        # Price consistency (for OHLCV data)
        ohlc_columns = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_columns):
            price_checks = (
                (df['high'] >= df['low']).mean() *
                (df['high'] >= df['open']).mean() *
                (df['high'] >= df['close']).mean() *
                (df['low'] <= df['open']).mean() *
                (df['low'] <= df['close']).mean()
            )
            metrics['price_consistency'] = price_checks
        
        return metrics
    
    def get_cleaning_summary(self, cleaning_report: Dict[str, Any]) -> str:
        """Generate a summary of the cleaning process.
        
        Args:
            cleaning_report: Cleaning report dictionary
            
        Returns:
            Formatted summary string
        """
        summary_lines = [
            "Data Cleaning Summary",
            "=" * 50,
            f"Original rows: {cleaning_report['original_rows']:,}",
            f"Final rows: {cleaning_report['final_rows']:,}",
            f"Rows removed: {cleaning_report['original_rows'] - cleaning_report['final_rows']:,}",
            f"Removal rate: {((cleaning_report['original_rows'] - cleaning_report['final_rows']) / cleaning_report['original_rows'] * 100):.1f}%",
            "",
            "Cleaning steps applied:"
        ]
        
        for step in cleaning_report['cleaning_steps']:
            if step in cleaning_report['step_results']:
                step_result = cleaning_report['step_results'][step]
                summary_lines.append(f"  - {step}: {step_result['rows_before']:,} -> {step_result['rows_after']:,} rows")
        
        if cleaning_report['warnings']:
            summary_lines.extend(["", "Warnings:", "  " + "\n  ".join(cleaning_report['warnings'])])
        
        if cleaning_report['errors']:
            summary_lines.extend(["", "Errors:", "  " + "\n  ".join(cleaning_report['errors'])])
        
        if cleaning_report['quality_metrics']:
            summary_lines.extend(["", "Quality Metrics:"])
            for metric, value in cleaning_report['quality_metrics'].items():
                if isinstance(value, float):
                    summary_lines.append(f"  - {metric}: {value:.4f}")
                else:
                    summary_lines.append(f"  - {metric}: {value}")
        
        return "\n".join(summary_lines)
