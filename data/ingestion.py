"""
Data ingestion from multiple crypto exchanges.
"""

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None
    CCXT_AVAILABLE = False
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import asyncio
from pathlib import Path
import logging

from utils.config import ExchangeConfig
from utils.logging import get_logger


class DataIngestion:
    """Data ingestion from multiple crypto exchanges."""
    
    def __init__(self, exchange_config: ExchangeConfig):
        """Initialize data ingestion.
        
        Args:
            exchange_config: Exchange configuration
        """
        if ccxt is None:
            raise ImportError("ccxt library is required for data ingestion. Install with: pip install ccxt")
        
        self.exchange_config = exchange_config
        self.logger = get_logger(f"data_ingestion.{exchange_config.name}")
        
        # Initialize exchange connection
        self.exchange = self._initialize_exchange()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60.0 / exchange_config.rate_limit
    
    def _initialize_exchange(self):
        """Initialize exchange connection.
        
        Returns:
            Exchange instance
        """
        exchange_class = getattr(ccxt, self.exchange_config.name)
        
        exchange_params = {
            'apiKey': self.exchange_config.api_key,
            'secret': self.exchange_config.api_secret,
            'sandbox': self.exchange_config.sandbox,
            'enableRateLimit': True,
            'rateLimit': int(self.min_request_interval * 1000)
        }
        
        if self.exchange_config.name == 'binance':
            exchange_params['options'] = {'defaultType': 'spot'}
        elif self.exchange_config.name == 'coinbase':
            exchange_params['sandbox'] = self.exchange_config.sandbox
        
        exchange = exchange_class(exchange_params)
        
        # Load markets
        try:
            exchange.load_markets()
            self.logger.info(f"Successfully connected to {self.exchange_config.name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.exchange_config.name}: {e}")
            raise
        
        return exchange
    
    def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[Union[int, datetime]] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get OHLCV data from exchange.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            since: Start time (timestamp or datetime)
            limit: Maximum number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()
        
        try:
            # Convert datetime to timestamp if needed
            if isinstance(since, datetime):
                since = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add symbol column
            df['symbol'] = symbol
            
            self.logger.info(f"Fetched {len(df)} OHLCV records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise
    
    def get_order_book(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """Get order book data.
        
        Args:
            symbol: Trading symbol
            limit: Number of order book levels
            
        Returns:
            Dictionary with 'bids' and 'asks' DataFrames
        """
        self._rate_limit()
        
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # Convert to DataFrames
            bids_df = pd.DataFrame(order_book['bids'], columns=['price', 'amount'])
            asks_df = pd.DataFrame(order_book['asks'], columns=['price', 'amount'])
            
            # Add timestamp
            timestamp = pd.Timestamp.now()
            bids_df['timestamp'] = timestamp
            asks_df['timestamp'] = timestamp
            
            # Add symbol
            bids_df['symbol'] = symbol
            asks_df['symbol'] = symbol
            
            result = {
                'bids': bids_df,
                'asks': asks_df,
                'timestamp': timestamp,
                'symbol': symbol
            }
            
            self.logger.info(f"Fetched order book for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch order book for {symbol}: {e}")
            raise
    
    def get_trades(
        self,
        symbol: str,
        since: Optional[Union[int, datetime]] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get recent trades.
        
        Args:
            symbol: Trading symbol
            since: Start time
            limit: Maximum number of trades
            
        Returns:
            DataFrame with trade data
        """
        self._rate_limit()
        
        try:
            # Convert datetime to timestamp if needed
            if isinstance(since, datetime):
                since = int(since.timestamp() * 1000)
            
            trades = self.exchange.fetch_trades(symbol, since, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Standardize columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add symbol
            df['symbol'] = symbol
            
            self.logger.info(f"Fetched {len(df)} trades for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {symbol}: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker information dictionary
        """
        self._rate_limit()
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Add timestamp and symbol
            ticker['timestamp'] = pd.Timestamp.now()
            ticker['symbol'] = symbol
            
            self.logger.info(f"Fetched ticker for {symbol}")
            return ticker
            
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise
    
    def get_historical_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """Get historical OHLCV data with automatic pagination.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        current_since = start_date
        
        while current_since < end_date:
            try:
                # Fetch batch of data
                batch = self.get_ohlcv(symbol, timeframe, current_since, 1000)
                
                if len(batch) == 0:
                    break
                
                all_data.append(batch)
                
                # Update timestamp for next batch
                last_timestamp = batch['timestamp'].max()
                current_since = last_timestamp + timedelta(hours=1)  # Adjust based on timeframe
                
                # Rate limiting
                time.sleep(self.min_request_interval)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch batch for {symbol}: {e}")
                break
        
        if all_data:
            # Combine all batches
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Filter by date range
            combined_df = combined_df[
                (combined_df['timestamp'] >= start_date) &
                (combined_df['timestamp'] <= end_date)
            ]
            
            self.logger.info(f"Fetched {len(combined_df)} historical OHLCV records for {symbol}")
            return combined_df
        else:
            self.logger.warning(f"No historical data fetched for {symbol}")
            return pd.DataFrame()
    
    def get_multiple_symbols_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        since: Optional[Union[int, datetime]] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            since: Start time
            limit: Maximum number of candles per symbol
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_ohlcv(symbol, timeframe, since, limit)
                results[symbol] = df
                
                # Rate limiting between symbols
                time.sleep(self.min_request_interval)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information and supported symbols.
        
        Returns:
            Exchange information dictionary
        """
        try:
            info = {
                'name': self.exchange_config.name,
                'sandbox': self.exchange_config.sandbox,
                'markets': list(self.exchange.markets.keys()),
                'timeframes': self.exchange.timeframes if hasattr(self.exchange, 'timeframes') else {},
                'fees': self.exchange.fees if hasattr(self.exchange, 'fees') else {},
                'limits': self.exchange.limits if hasattr(self.exchange, 'limits') else {}
            }
            
            self.logger.info(f"Retrieved exchange info for {self.exchange_config.name}")
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get exchange info: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test exchange connection.
        
        Returns:
            True if connection successful
        """
        try:
            # Try to fetch a simple endpoint
            self.exchange.fetch_ticker('BTC/USDT')
            self.logger.info(f"Connection test successful for {self.exchange_config.name}")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed for {self.exchange_config.name}: {e}")
            return False


class MultiExchangeIngestion:
    """Data ingestion from multiple exchanges simultaneously."""
    
    def __init__(self, exchange_configs: List[ExchangeConfig]):
        """Initialize multi-exchange ingestion.
        
        Args:
            exchange_configs: List of exchange configurations
        """
        self.exchange_configs = exchange_configs
        self.ingestors = {}
        
        # Initialize ingestors for each exchange
        for config in exchange_configs:
            try:
                self.ingestors[config.name] = DataIngestion(config)
            except Exception as e:
                logging.error(f"Failed to initialize {config.name}: {e}")
    
    def get_ohlcv_from_all(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[Union[int, datetime]] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data from all exchanges.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            since: Start time
            limit: Maximum number of candles
            
        Returns:
            Dictionary mapping exchange names to DataFrames
        """
        results = {}
        
        for exchange_name, ingestor in self.ingestors.items():
            try:
                # Map symbol to exchange-specific format
                exchange_symbol = self._map_symbol(symbol, exchange_name)
                df = ingestor.get_ohlcv(exchange_symbol, timeframe, since, limit)
                results[exchange_name] = df
            except Exception as e:
                logging.error(f"Failed to fetch from {exchange_name}: {e}")
                results[exchange_name] = pd.DataFrame()
        
        return results
    
    def _map_symbol(self, symbol: str, exchange_name: str) -> str:
        """Map symbol to exchange-specific format.
        
        Args:
            symbol: Generic symbol (e.g., 'BTC/USDT')
            exchange_name: Exchange name
            
        Returns:
            Exchange-specific symbol
        """
        # Simple mapping - in production, you'd want a more sophisticated mapping
        if exchange_name == 'binance':
            return symbol.replace('/', '')
        elif exchange_name == 'coinbase':
            return symbol.replace('/', '-')
        else:
            return symbol
    
    def get_best_price(self, symbol: str) -> Dict[str, float]:
        """Get best bid/ask from all exchanges.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with best prices from each exchange
        """
        best_prices = {}
        
        for exchange_name, ingestor in self.ingestors.items():
            try:
                exchange_symbol = self._map_symbol(symbol, exchange_name)
                ticker = ingestor.get_ticker(exchange_symbol)
                
                best_prices[exchange_name] = {
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0),
                    'last': ticker.get('last', 0),
                    'timestamp': ticker.get('timestamp', pd.Timestamp.now())
                }
            except Exception as e:
                logging.error(f"Failed to get price from {exchange_name}: {e}")
        
        return best_prices
