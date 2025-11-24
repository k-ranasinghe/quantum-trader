from pydantic_settings import BaseSettings
from typing import List, Optional
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    SECURITY = "security"


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Quantum Trading System"

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "quantum"
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str = "quantum_trading"

    # InfluxDB (Time-series)
    INFLUX_URL: str = "http://localhost:8086"
    INFLUX_TOKEN: str
    INFLUX_ORG: str = "quantum-trading"
    INFLUX_BUCKET: str = "market-data"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # RabbitMQ
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "quantum"
    RABBITMQ_PASSWORD: str

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "quantum-trading"

    # Data Sources
    POLYGON_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None

    # Brokers
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # Model Settings
    MODEL_UPDATE_FREQUENCY: int = 7  # days
    DRIFT_THRESHOLD: float = 0.2
    CONFIDENCE_THRESHOLD: float = 0.65

    # Trading
    INITIAL_CAPITAL: float = 100000.0
    MAX_POSITION_SIZE: float = 0.05  # 5% of capital
    TRANSACTION_COST_PCT: float = 0.001
    SLIPPAGE_PCT: float = 0.0005

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Grafana
    GRAFANA_PASSWORD: str

    class Config:
        env_file = "../.env"
        case_sensitive = True


SETTINGS = Settings()