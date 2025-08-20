"""
Data storage and management for the crypto trading bot.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.logging import get_logger


class DataStorage:
    """Data storage and management system."""
    
    def __init__(self, base_path: Union[str, Path] = "data"):
        """Initialize data storage.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.logger = get_logger("data_storage")
        
        # Create directory structure
        self._create_directories()
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
    
    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.base_path / "raw",
            self.base_path / "processed",
            self.base_path / "features",
            self.base_path / "models",
            self.base_path / "backtests",
            self.base_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created data directories in {self.base_path}")
    
    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_type: str = "raw"
    ) -> Path:
        """Save OHLCV data to storage.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_type: Data type (raw, processed, features)
            
        Returns:
            Path to saved file
        """
        with self._lock:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{exchange}_{timeframe}_{timestamp}.parquet"
            
            # Determine directory
            if data_type == "raw":
                directory = self.base_path / "raw" / exchange / symbol
            elif data_type == "processed":
                directory = self.base_path / "processed" / exchange / symbol
            elif data_type == "features":
                directory = self.base_path / "features" / exchange / symbol
            else:
                raise ValueError(f"Invalid data_type: {data_type}")
            
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            # Save to Parquet
            df.to_parquet(filepath, compression='snappy', index=False)
            
            # Save metadata
            self._save_metadata(filepath, {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_type': data_type,
                'rows': len(df),
                'columns': list(df.columns),
                'start_date': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end_date': df['timestamp'].max().isoformat() if len(df) > 0 else None,
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Saved {len(df)} rows to {filepath}")
            return filepath
    
    def load_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_type: str = "raw",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load OHLCV data from storage.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_type: Data type
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        with self._lock:
            # Determine directory
            if data_type == "raw":
                directory = self.base_path / "raw" / exchange / symbol
            elif data_type == "processed":
                directory = self.base_path / "processed" / exchange / symbol
            elif data_type == "features":
                directory = self.base_path / "features" / exchange / symbol
            else:
                raise ValueError(f"Invalid data_type: {data_type}")
            
            if not directory.exists():
                self.logger.warning(f"Directory {directory} does not exist")
                return pd.DataFrame()
            
            # Find all parquet files
            parquet_files = list(directory.glob("*.parquet"))
            
            if not parquet_files:
                self.logger.warning(f"No parquet files found in {directory}")
                return pd.DataFrame()
            
            # Sort files by creation time
            parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Load and combine data
            all_data = []
            for filepath in parquet_files:
                try:
                    df = pd.read_parquet(filepath)
                    all_data.append(df)
                except Exception as e:
                    self.logger.error(f"Failed to load {filepath}: {e}")
                    continue
            
            if not all_data:
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Apply date filters
            if start_date:
                combined_df = combined_df[combined_df['timestamp'] >= start_date]
            if end_date:
                combined_df = combined_df[combined_df['timestamp'] <= end_date]
            
            self.logger.info(f"Loaded {len(combined_df)} rows for {symbol} from {exchange}")
            return combined_df
    
    def save_order_book(
        self,
        order_book: Dict[str, pd.DataFrame],
        symbol: str,
        exchange: str
    ) -> Path:
        """Save order book data.
        
        Args:
            order_book: Order book data
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Path to saved file
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{exchange}_orderbook_{timestamp}.parquet"
            
            directory = self.base_path / "raw" / exchange / symbol / "orderbook"
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            # Save bids and asks separately
            bids_df = order_book['bids']
            asks_df = order_book['asks']
            
            # Combine into single DataFrame with type column
            bids_df['type'] = 'bid'
            asks_df['type'] = 'ask'
            combined_df = pd.concat([bids_df, asks_df], ignore_index=True)
            
            combined_df.to_parquet(filepath, compression='snappy', index=False)
            
            # Save metadata
            self._save_metadata(filepath, {
                'symbol': symbol,
                'exchange': exchange,
                'data_type': 'orderbook',
                'bids_count': len(bids_df),
                'asks_count': len(asks_df),
                'timestamp': order_book['timestamp'].isoformat(),
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Saved order book data to {filepath}")
            return filepath
    
    def save_trades(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> Path:
        """Save trade data.
        
        Args:
            df: DataFrame with trade data
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Path to saved file
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{exchange}_trades_{timestamp}.parquet"
            
            directory = self.base_path / "raw" / exchange / symbol / "trades"
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            df.to_parquet(filepath, compression='snappy', index=False)
            
            # Save metadata
            self._save_metadata(filepath, {
                'symbol': symbol,
                'exchange': exchange,
                'data_type': 'trades',
                'rows': len(df),
                'start_date': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end_date': df['timestamp'].max().isoformat() if len(df) > 0 else None,
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Saved {len(df)} trades to {filepath}")
            return filepath
    
    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        feature_set: str
    ) -> Path:
        """Save engineered features.
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            exchange: Exchange name
            feature_set: Name of feature set
            
        Returns:
            Path to saved file
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{exchange}_{feature_set}_{timestamp}.parquet"
            
            directory = self.base_path / "features" / exchange / symbol
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            df.to_parquet(filepath, compression='snappy', index=False)
            
            # Save metadata
            self._save_metadata(filepath, {
                'symbol': symbol,
                'exchange': exchange,
                'data_type': 'features',
                'feature_set': feature_set,
                'rows': len(df),
                'features': list(df.columns),
                'start_date': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end_date': df['timestamp'].max().isoformat() if len(df) > 0 else None,
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Saved features to {filepath}")
            return filepath
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        symbol: str,
        exchange: str,
        model_type: str = "ml"
    ) -> Path:
        """Save ML model.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            symbol: Trading symbol
            exchange: Exchange name
            model_type: Type of model
            
        Returns:
            Path to saved model
        """
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{exchange}_{model_name}_{timestamp}.pkl"
            
            directory = self.base_path / "models" / exchange / symbol / model_type
            directory.mkdir(parents=True, exist_ok=True)
            filepath = directory / filename
            
            # Save model (you might want to use joblib or pickle)
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            self._save_metadata(filepath, {
                'symbol': symbol,
                'exchange': exchange,
                'model_name': model_name,
                'model_type': model_type,
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"Saved model to {filepath}")
            return filepath
    
    def _save_metadata(self, filepath: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata for a file.
        
        Args:
            filepath: Path to the file
            metadata: Metadata dictionary
        """
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def get_data_info(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Get information about available data for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Dictionary with data information
        """
        info = {
            'symbol': symbol,
            'exchange': exchange,
            'raw_data': {},
            'processed_data': {},
            'features': {},
            'models': {}
        }
        
        # Check raw data
        raw_dir = self.base_path / "raw" / exchange / symbol
        if raw_dir.exists():
            for timeframe_dir in raw_dir.iterdir():
                if timeframe_dir.is_dir():
                    parquet_files = list(timeframe_dir.glob("*.parquet"))
                    if parquet_files:
                        info['raw_data'][timeframe_dir.name] = len(parquet_files)
        
        # Check processed data
        processed_dir = self.base_path / "processed" / exchange / symbol
        if processed_dir.exists():
            for timeframe_dir in processed_dir.iterdir():
                if timeframe_dir.is_dir():
                    parquet_files = list(timeframe_dir.glob("*.parquet"))
                    if parquet_files:
                        info['processed_data'][timeframe_dir.name] = len(parquet_files)
        
        # Check features
        features_dir = self.base_path / "features" / exchange / symbol
        if features_dir.exists():
            parquet_files = list(features_dir.glob("*.parquet"))
            if parquet_files:
                info['features'] = len(parquet_files)
        
        # Check models
        models_dir = self.base_path / "models" / exchange / symbol
        if models_dir.exists():
            for model_type_dir in models_dir.iterdir():
                if model_type_dir.is_dir():
                    model_files = list(model_type_dir.glob("*.pkl"))
                    if model_files:
                        info['models'][model_type_dir.name] = len(model_files)
        
        return info
    
    def cleanup_old_data(
        self,
        max_age_days: int = 30,
        keep_latest_files: int = 10
    ) -> Dict[str, int]:
        """Clean up old data files.
        
        Args:
            max_age_days: Maximum age of files to keep
            keep_latest_files: Number of latest files to keep per symbol/timeframe
            
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {'files_removed': 0, 'bytes_freed': 0}
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        with self._lock:
            # Process all data directories
            for data_type in ['raw', 'processed', 'features']:
                data_dir = self.base_path / data_type
                if not data_dir.exists():
                    continue
                
                for exchange_dir in data_dir.iterdir():
                    if not exchange_dir.is_dir():
                        continue
                    
                    for symbol_dir in exchange_dir.iterdir():
                        if not symbol_dir.is_dir():
                            continue
                        
                        # Find all parquet files
                        parquet_files = list(symbol_dir.rglob("*.parquet"))
                        
                        if len(parquet_files) <= keep_latest_files:
                            continue
                        
                        # Sort by modification time
                        parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        
                        # Keep latest files, remove old ones
                        files_to_remove = parquet_files[keep_latest_files:]
                        
                        for filepath in files_to_remove:
                            try:
                                # Check file age
                                file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                                if file_time < cutoff_time:
                                    # Remove file and metadata
                                    file_size = filepath.stat().st_size
                                    filepath.unlink()
                                    
                                    # Remove metadata file
                                    metadata_file = filepath.with_suffix('.json')
                                    if metadata_file.exists():
                                        metadata_file.unlink()
                                    
                                    cleanup_stats['files_removed'] += 1
                                    cleanup_stats['bytes_freed'] += file_size
                                    
                            except Exception as e:
                                self.logger.error(f"Failed to remove {filepath}: {e}")
        
        self.logger.info(f"Cleanup completed: {cleanup_stats['files_removed']} files removed, "
                        f"{cleanup_stats['bytes_freed'] / 1024 / 1024:.2f} MB freed")
        
        return cleanup_stats
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'total_size': 0,
            'file_count': 0,
            'data_types': {},
            'exchanges': {},
            'symbols': {}
        }
        
        with self._lock:
            # Calculate total size and count
            for filepath in self.base_path.rglob("*"):
                if filepath.is_file():
                    try:
                        file_size = filepath.stat().st_size
                        stats['total_size'] += file_size
                        stats['file_count'] += 1
                        
                        # Categorize by file type
                        file_ext = filepath.suffix
                        if file_ext not in stats['data_types']:
                            stats['data_types'][file_ext] = {'count': 0, 'size': 0}
                        stats['data_types'][file_ext]['count'] += 1
                        stats['data_types'][file_ext]['size'] += file_size
                        
                    except Exception as e:
                        self.logger.error(f"Failed to get stats for {filepath}: {e}")
            
            # Get exchange and symbol breakdown
            for data_type in ['raw', 'processed', 'features']:
                data_dir = self.base_path / data_type
                if data_dir.exists():
                    for exchange_dir in data_dir.iterdir():
                        if exchange_dir.is_dir():
                            exchange_name = exchange_dir.name
                            if exchange_name not in stats['exchanges']:
                                stats['exchanges'][exchange_name] = {'count': 0, 'size': 0}
                            
                            for symbol_dir in exchange_dir.iterdir():
                                if symbol_dir.is_dir():
                                    symbol_name = symbol_dir.name
                                    if symbol_name not in stats['symbols']:
                                        stats['symbols'][symbol_name] = {'count': 0, 'size': 0}
                                    
                                    # Count files in symbol directory
                                    for filepath in symbol_dir.rglob("*"):
                                        if filepath.is_file():
                                            try:
                                                file_size = filepath.stat().st_size
                                                stats['exchanges'][exchange_name]['count'] += 1
                                                stats['exchanges'][exchange_name]['size'] += file_size
                                                stats['symbols'][symbol_name]['count'] += 1
                                                stats['symbols'][symbol_name]['size'] += file_size
                                            except Exception:
                                                pass
        
        # Convert bytes to MB
        stats['total_size_mb'] = stats['total_size'] / 1024 / 1024
        
        return stats
