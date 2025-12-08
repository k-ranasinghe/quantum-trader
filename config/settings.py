import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "QuantumTrader"
    ENV: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/quant_trader"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # RabbitMQ
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672//"

    # MLFlow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
    MODEL_REGISTRY_DIR: str = os.path.join(RESULTS_DIR, "registry")

    # Model Config
    DEVICE: str = "cpu"  # Will be updated based on availability

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.MODEL_REGISTRY_DIR, exist_ok=True)
