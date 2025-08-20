"""
Data ingestion, cleaning, and storage for the crypto trading bot.
"""

from data.storage import DataStorage
from data.cleaner import DataCleaner

# DataIngestion requires ccxt, so import it conditionally
try:
    from data.ingestion import DataIngestion
    __all__ = ["DataIngestion", "DataStorage", "DataCleaner"]
except ImportError:
    __all__ = ["DataStorage", "DataCleaner"]
