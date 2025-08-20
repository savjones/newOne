"""
Configuration management for the crypto trading bot.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="crypto_trading")
    user: str = Field(default="postgres")
    password: str = Field(default="")
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "crypto_trading"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )


class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    name: str = Field(..., description="Exchange name (binance, coinbase, bybit)")
    api_key: str = Field(default="")
    api_secret: str = Field(default="")
    sandbox: bool = Field(default=True, description="Use sandbox/testnet")
    rate_limit: int = Field(default=100, description="Requests per minute")
    
    @classmethod
    def from_env(cls, exchange_name: str) -> "ExchangeConfig":
        """Create config from environment variables."""
        prefix = exchange_name.upper()
        return cls(
            name=exchange_name,
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            api_secret=os.getenv(f"{prefix}_API_SECRET", ""),
            sandbox=os.getenv(f"{prefix}_SANDBOX", "true").lower() == "true",
            rate_limit=int(os.getenv(f"{prefix}_RATE_LIMIT", "100"))
        )


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size: float = Field(default=0.1, description="Max % of portfolio per position")
    max_drawdown: float = Field(default=0.2, description="Max portfolio drawdown")
    stop_loss: float = Field(default=0.05, description="Default stop loss %")
    take_profit: float = Field(default=0.15, description="Default take profit %")
    max_leverage: float = Field(default=1.0, description="Maximum leverage allowed")


class ConfigManager:
    """Configuration manager for the trading bot."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = Path(config_path) if config_path else Path("configs")
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from files and environment."""
        # Load main config
        main_config_path = self.config_path / "main.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r') as f:
                self.config.update(yaml.safe_load(f))
        
        # Load environment-specific config
        env = os.getenv("ENVIRONMENT", "development")
        env_config_path = self.config_path / f"{env}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                self.config.update(yaml.safe_load(f))
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration dictionary
        """
        strategy_config_path = self.config_path / "strategies" / f"{strategy_name}.yaml"
        if strategy_config_path.exists():
            with open(strategy_config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Get exchange configuration.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            Exchange configuration object
        """
        return ExchangeConfig.from_env(exchange_name)
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration.
        
        Returns:
            Database configuration object
        """
        return DatabaseConfig.from_env()
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration.
        
        Returns:
            Risk configuration object
        """
        risk_config = self.config.get("risk", {})
        return RiskConfig(**risk_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def save_config(self, filename: str = "config.yaml") -> None:
        """Save current configuration to file.
        
        Args:
            filename: Name of the configuration file
        """
        config_file = self.config_path / filename
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


# Global configuration instance
config_manager = ConfigManager()
