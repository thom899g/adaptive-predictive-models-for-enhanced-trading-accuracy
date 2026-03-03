"""
Configuration management for Adaptive Trading System
Architectural Choice: Centralized config with environment-based overrides
enables deployment flexibility across development/production environments
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation"""
    project_id: str
    private_key: str
    client_email: str
    database_url: str = "https://adaptive-trading-system.firebaseio.com"
    
    def __post_init__(self):
        """Validate critical Firebase credentials"""
        if not all([self.project_id, self.private_key, self.client_email]):
            raise ValueError("Missing required Firebase credentials")
        # Replace escaped newlines in private key
        self.private_key = self.private_key.replace('\\n', '\n')

@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    exchange_id: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit: int = 1200  # ms between requests

@dataclass
class ModelConfig:
    """Model hyperparameters and architecture"""
    dnn_hidden_layers: tuple = (128, 64, 32)
    lstm_units: tuple = (50, 25)
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 60  # Time steps for LSTM
    validation_split: float = 0.15

@dataclass
class TradingConfig:
    """Trading execution parameters"""
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_concurrent_positions: int = 5
    min_confidence_threshold: float = 0.65

class Config:
    """Main configuration singleton"""
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration from environment variables"""
        self.firebase = FirebaseConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
            private_key=os.getenv("FIREBASE_PRIVATE_KEY", ""),
            client_email=os.getenv("FIREBASE_CLIENT_EMAIL", "")
        )
        
        self.exchange = ExchangeConfig(
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", "")
        )
        
        self.model = ModelConfig()
        self.trading = TradingConfig()
        
        # System settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.data_collection_interval = 300  # 5 minutes
        self.model_retrain_hours = 24
        
        # Symbols to trade (BTC/USDT, ETH/USDT as default)
        self.trading_pairs = [
            "BTC/USDT",
            "ETH/USDT", 
            "SOL/USDT",
            "ADA/USDT"
        ]

config = Config()