"""
CSV Data Loader with Schema Normalization

This module provides functionality to load OHLCV data from various CSV formats
and normalize them to a standard schema with UTC timestamps.

Features:
- Automatic schema detection and normalization
- UTC timestamp conversion
- Data validation and cleaning
- Support for multiple datetime formats
- Configurable column mapping
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class DatetimeFormat(Enum):
    """Supported datetime formats"""
    ISO_DATE = "%Y-%m-%d"
    ISO_DATETIME = "%Y-%m-%d %H:%M:%S"
    ISO_DATETIME_MS = "%Y-%m-%d %H:%M:%S.%f"
    TIMESTAMP = "timestamp"
    UNIX_TIMESTAMP = "unix"


@dataclass
class CSVSchema:
    """CSV schema definition"""
    datetime_column: str
    datetime_format: DatetimeFormat
    open_column: str
    high_column: str
    low_column: str
    close_column: str
    volume_column: str
    symbol_column: Optional[str] = None
    has_header: bool = True
    delimiter: str = ','
    decimal: str = '.'


@dataclass
class NormalizedData:
    """Container for normalized OHLCV data"""
    data: pd.DataFrame
    symbol: str
    timeframe: str
    source_file: str
    original_schema: CSVSchema
    total_rows: int
    valid_rows: int
    date_range: Tuple[datetime, datetime]


class CSVDataLoader:
    """
    CSV Data Loader with automatic schema detection and normalization
    
    This class can load OHLCV data from various CSV formats and normalize
    them to a standard format with UTC timestamps.
    """
    
    # Common column name mappings
    COLUMN_MAPPINGS = {
        'datetime': ['datetime', 'date', 'timestamp', 'time', 'dt'],
        'open': ['open', 'Open', 'OPEN', 'o'],
        'high': ['high', 'High', 'HIGH', 'h'],
        'low': ['low', 'Low', 'LOW', 'l'],
        'close': ['close', 'Close', 'CLOSE', 'c'],
        'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'v']
    }
    
    # Common datetime formats to try
    DATETIME_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ"
    ]
    
    def __init__(self, data_directory: Union[str, Path] = "data/csvs"):
        """
        Initialize CSV Data Loader
        
        Args:
            data_directory: Directory containing CSV files
        """
        self.data_directory = Path(data_directory)
        self.loaded_files: Dict[str, NormalizedData] = {}
        
        if not self.data_directory.exists():
            logger.warning(f"Data directory {self.data_directory} does not exist")
    
    def detect_schema(self, file_path: Path, sample_rows: int = 10) -> Optional[CSVSchema]:
        """
        Automatically detect CSV schema from file
        
        Args:
            file_path: Path to CSV file
            sample_rows: Number of rows to sample for detection
            
        Returns:
            Detected schema or None if detection fails
        """
        try:
            # Try different delimiters and read sample
            for delimiter in [',', ';', '\t', '|']:
                try:
                    sample_df = pd.read_csv(
                        file_path, 
                        delimiter=delimiter, 
                        nrows=sample_rows,
                        low_memory=False
                    )
                    
                    if len(sample_df.columns) >= 5:  # Minimum OHLCV columns
                        break
                except:
                    continue
            else:
                logger.error(f"Could not read {file_path} with any delimiter")
                return None
            
            columns = [col.strip().lower() for col in sample_df.columns]
            logger.info(f"Detected columns in {file_path.name}: {columns}")
            
            # Map columns to OHLCV
            column_mapping = {}
            for target, candidates in self.COLUMN_MAPPINGS.items():
                for col_name in sample_df.columns:
                    if col_name.strip().lower() in [c.lower() for c in candidates]:
                        column_mapping[target] = col_name.strip()
                        break
            
            # Validate we have all required columns
            required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in column_mapping]
            
            if missing_columns:
                logger.error(f"Missing required columns in {file_path.name}: {missing_columns}")
                return None
            
            # Detect datetime format
            datetime_col = column_mapping['datetime']
            datetime_format = self._detect_datetime_format(sample_df[datetime_col])
            
            if datetime_format is None:
                logger.error(f"Could not detect datetime format in {file_path.name}")
                return None
            
            return CSVSchema(
                datetime_column=datetime_col,
                datetime_format=datetime_format,
                open_column=column_mapping['open'],
                high_column=column_mapping['high'],
                low_column=column_mapping['low'],
                close_column=column_mapping['close'],
                volume_column=column_mapping['volume'],
                has_header=True,
                delimiter=delimiter
            )
            
        except Exception as e:
            logger.error(f"Schema detection failed for {file_path}: {e}")
            return None
    
    def _detect_datetime_format(self, datetime_series: pd.Series) -> Optional[DatetimeFormat]:
        """
        Detect datetime format from sample data
        
        Args:
            datetime_series: Series containing datetime values
            
        Returns:
            Detected datetime format or None
        """
        # Get first non-null value
        sample_value = None
        for val in datetime_series.dropna():
            sample_value = str(val).strip()
            break
        
        if not sample_value:
            return None
        
        # Check if it's a unix timestamp
        try:
            timestamp_val = float(sample_value)
            # Unix timestamps are typically 10 or 13 digits
            if 1000000000 <= timestamp_val <= 9999999999999:
                return DatetimeFormat.UNIX_TIMESTAMP
        except:
            pass
        
        # Try standard formats
        for fmt in self.DATETIME_FORMATS:
            try:
                datetime.strptime(sample_value, fmt)
                if fmt == "%Y-%m-%d":
                    return DatetimeFormat.ISO_DATE
                elif fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    return DatetimeFormat.ISO_DATETIME
                elif fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ"]:
                    return DatetimeFormat.ISO_DATETIME_MS
                else:
                    return DatetimeFormat.ISO_DATETIME  # Default for other formats
            except:
                continue
        
        return None
    
    def load_csv(self, file_path: Union[str, Path], 
                 schema: Optional[CSVSchema] = None,
                 symbol: Optional[str] = None) -> Optional[NormalizedData]:
        """
        Load and normalize CSV data
        
        Args:
            file_path: Path to CSV file
            schema: Optional predefined schema (will auto-detect if None)
            symbol: Optional symbol name (will extract from filename if None)
            
        Returns:
            Normalized data or None if loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Auto-detect schema if not provided
        if schema is None:
            schema = self.detect_schema(file_path)
            if schema is None:
                return None
# Debug output removed for production
        
        try:
            # Load full dataset
            df = pd.read_csv(
                file_path,
                delimiter=schema.delimiter,
                decimal=schema.decimal,
                low_memory=False
            )
            
            original_rows = len(df)
            logger.info(f"Loaded {original_rows} rows from {file_path.name}")
            
            # Normalize data
            normalized_df = self._normalize_dataframe(df, schema)
            
            if normalized_df is None or normalized_df.empty:
                logger.error(f"Normalization failed for {file_path.name}")
                return None
            
            # Extract symbol from filename if not provided
            if symbol is None:
                symbol = self._extract_symbol_from_filename(file_path.name)
            
            # Extract timeframe from filename
            timeframe = self._extract_timeframe_from_filename(file_path.name)
            
            # Get date range
            date_range = (
                normalized_df.index.min().to_pydatetime(),
                normalized_df.index.max().to_pydatetime()
            )
            
            normalized_data = NormalizedData(
                data=normalized_df,
                symbol=symbol,
                timeframe=timeframe,
                source_file=str(file_path),
                original_schema=schema,
                total_rows=original_rows,
                valid_rows=len(normalized_df),
                date_range=date_range
            )
            
            # Cache the loaded data
            self.loaded_files[file_path.name] = normalized_data
            
            logger.info(f"Successfully normalized {file_path.name}: "
                       f"{normalized_data.valid_rows}/{normalized_data.total_rows} valid rows, "
                       f"symbol={symbol}, timeframe={timeframe}")
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def _normalize_dataframe(self, df: pd.DataFrame, schema: CSVSchema) -> Optional[pd.DataFrame]:
        """
        Normalize dataframe to standard format
        
        Args:
            df: Raw dataframe
            schema: CSV schema
            
        Returns:
            Normalized dataframe with standard columns and UTC timestamps
        """
        try:
            # Create normalized dataframe
            normalized = pd.DataFrame()
            
            # Convert datetime column
            datetime_series = df[schema.datetime_column]
            
            if schema.datetime_format == DatetimeFormat.UNIX_TIMESTAMP:
                # Handle unix timestamps
                timestamps = pd.to_numeric(datetime_series, errors='coerce')
                # Detect if milliseconds or seconds
                if timestamps.max() > 1e12:  # Milliseconds
                    timestamps = timestamps / 1000
                normalized.index = pd.to_datetime(timestamps, unit='s', utc=True)
            else:
                # Handle string datetime formats
                if schema.datetime_format == DatetimeFormat.ISO_DATE:
                    normalized.index = pd.to_datetime(datetime_series, format='%Y-%m-%d', utc=True)
                elif schema.datetime_format == DatetimeFormat.ISO_DATETIME:
                    # Try multiple datetime formats
                    parsed_dates = None
                    for fmt in self.DATETIME_FORMATS:
                        try:
                            parsed_dates = pd.to_datetime(datetime_series, format=fmt, utc=True)
                            break
                        except:
                            continue
                    
                    if parsed_dates is None:
                        parsed_dates = pd.to_datetime(datetime_series, utc=True)
                    
                    normalized.index = parsed_dates
                else:
                    normalized.index = pd.to_datetime(datetime_series, utc=True)
            
            # Copy OHLCV columns (using .values to ensure index alignment)
            normalized['open'] = pd.to_numeric(df[schema.open_column], errors='coerce').values
            normalized['high'] = pd.to_numeric(df[schema.high_column], errors='coerce').values
            normalized['low'] = pd.to_numeric(df[schema.low_column], errors='coerce').values
            normalized['close'] = pd.to_numeric(df[schema.close_column], errors='coerce').values
            normalized['volume'] = pd.to_numeric(df[schema.volume_column], errors='coerce').values
            
            # Add symbol column if available
            if schema.symbol_column and schema.symbol_column in df.columns:
                normalized['symbol'] = df[schema.symbol_column]
            
            # Data validation and cleaning
            normalized = self._clean_ohlcv_data(normalized)
            
            # Sort by datetime
            normalized = normalized.sort_index()
            
            # Remove duplicates
            normalized = normalized[~normalized.index.duplicated(keep='first')]
            
            return normalized
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return None
    
    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df = df[df[col] > 0]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        # Validate OHLC relationships
        valid_ohlc = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        
        invalid_count = (~valid_ohlc).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
            df = df[valid_ohlc]
        
        # Remove extreme outliers (prices that change more than 50% in one period)
        price_change = df['close'].pct_change().abs()
        outliers = price_change > 0.5
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            logger.warning(f"Removing {outlier_count} rows with extreme price changes")
            df = df[~outliers]
        
        return df
    
    def _extract_symbol_from_filename(self, filename: str) -> str:
        """Extract symbol from filename"""
        # Remove extension
        name = Path(filename).stem
        
        # Common patterns: BTC-1d-data, ETH_daily, BTCUSDT-4h
        patterns = [
            r'^([A-Z]{2,10})',  # Symbol at start
            r'([A-Z]{2,10})-',   # Symbol before dash
            r'([A-Z]{2,10})_',   # Symbol before underscore
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name.upper())
            if match:
                return match.group(1)
        
        # Fallback: use first part of filename
        parts = re.split(r'[-_.]', name)
        if parts:
            return parts[0].upper()
        
        return "UNKNOWN"
    
    def _extract_timeframe_from_filename(self, filename: str) -> str:
        """Extract timeframe from filename"""
        filename_lower = filename.lower()
        
        # Common timeframe patterns
        timeframe_patterns = {
            r'1m|1min': '1m',
            r'5m|5min': '5m',
            r'15m|15min': '15m',
            r'30m|30min': '30m',
            r'1h|1hr|hourly': '1h',
            r'4h|4hr': '4h',
            r'6h|6hr': '6h',
            r'12h|12hr': '12h',
            r'1d|1day|daily': '1d',
            r'1w|1week|weekly': '1w',
            r'1M|1month|monthly': '1M'
        }
        
        for pattern, timeframe in timeframe_patterns.items():
            if re.search(pattern, filename_lower):
                return timeframe
        
        return "unknown"
    
    def load_all_csvs(self) -> Dict[str, NormalizedData]:
        """
        Load all CSV files from the data directory
        
        Returns:
            Dictionary mapping filenames to normalized data
        """
        if not self.data_directory.exists():
            logger.error(f"Data directory {self.data_directory} does not exist")
            return {}
        
        csv_files = list(self.data_directory.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_directory}")
        
        loaded_data = {}
        
        for csv_file in csv_files:
            logger.info(f"Loading {csv_file.name}...")
            normalized_data = self.load_csv(csv_file)
            
            if normalized_data:
                loaded_data[csv_file.name] = normalized_data
                logger.info(f"✓ Successfully loaded {csv_file.name}")
            else:
                logger.warning(f"✗ Failed to load {csv_file.name}")
        
        logger.info(f"Successfully loaded {len(loaded_data)}/{len(csv_files)} CSV files")
        return loaded_data
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of all loaded data
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.loaded_files:
            return pd.DataFrame()
        
        summary_data = []
        
        for filename, data in self.loaded_files.items():
            summary_data.append({
                'Filename': filename,
                'Symbol': data.symbol,
                'Timeframe': data.timeframe,
                'Total Rows': data.total_rows,
                'Valid Rows': data.valid_rows,
                'Data Quality (%)': round(data.valid_rows / data.total_rows * 100, 1),
                'Start Date': data.date_range[0].strftime('%Y-%m-%d'),
                'End Date': data.date_range[1].strftime('%Y-%m-%d'),
                'Duration (Days)': (data.date_range[1] - data.date_range[0]).days,
                'Avg Volume': f"{data.data['volume'].mean():.0f}",
                'Price Range': f"{data.data['low'].min():.2f} - {data.data['high'].max():.2f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def validate_data_quality(self, data: NormalizedData, 
                             min_quality_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Validate data quality metrics
        
        Args:
            data: Normalized data to validate
            min_quality_threshold: Minimum acceptable data quality ratio
            
        Returns:
            Dictionary with validation results
        """
        df = data.data
        
        # Calculate quality metrics
        total_rows = len(df)
        
        # Check for gaps in data
        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if not time_diff.empty else pd.Timedelta(days=1)
        gaps = (time_diff > expected_freq * 1.5).sum()
        
        # Check for zero volume periods
        zero_volume = (df['volume'] == 0).sum()
        
        # Check for constant prices
        constant_prices = (df['high'] == df['low']).sum()
        
        # Overall quality score
        quality_issues = gaps + zero_volume + constant_prices
        quality_score = max(0, 1 - (quality_issues / total_rows))
        
        validation_result = {
            'quality_score': quality_score,
            'passes_threshold': quality_score >= min_quality_threshold,
            'total_rows': total_rows,
            'gaps_detected': gaps,
            'zero_volume_periods': zero_volume,
            'constant_price_periods': constant_prices,
            'expected_frequency': str(expected_freq),
            'date_range_days': (data.date_range[1] - data.date_range[0]).days
        }
        
        return validation_result
