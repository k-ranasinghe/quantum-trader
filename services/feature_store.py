import redis
import pandas as pd
import json
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from config.settings import settings
from config.constants import CONSTANTS
from utilities.logger import logger


class FeatureStore:
    """
    Custom lightweight Feature Store backed by Redis (online) and yfinance (offline/source).
    """

    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.feature_cols = CONSTANTS.FEATURE_COLS

    def get_online_features(self, asset_symbol: str) -> Optional[pd.DataFrame]:
        """
        Get the latest features for inference from Redis.
        If not in Redis or stale, fetch from source and update Redis.
        """
        key = f"features:{asset_symbol}"
        data_json = self.redis_client.get(key)

        if data_json:
            try:
                # Deserialize
                df_dict = json.loads(data_json)
                df = pd.DataFrame.from_dict(df_dict)
                df.index = pd.to_datetime(df.index)

                # Check freshness (simple check: last timestamp)
                last_ts = df.index[-1]
                if datetime.now() - last_ts > timedelta(hours=2):  # Assuming 1h timeframe
                    logger.info(f"Cached features for {asset_symbol} are stale. Refreshing.")
                    return self.refresh_features(asset_symbol)

                return df
            except Exception as e:
                logger.error(f"Error deserializing features for {asset_symbol}: {e}")

        # If missing or error, refresh
        logger.info(f"No valid online features for {asset_symbol}. Fetching from source.")
        return self.refresh_features(asset_symbol)

    def refresh_features(self, asset_symbol: str) -> pd.DataFrame:
        """
        Fetch latest data, compute features, and cache in Redis.
        """
        # Fetch data sufficient for calculating indicators (e.g., last 60 days for 1h timeframe)
        # We need enough history for rolling windows (e.g. MA60)
        lookback = 60  # days

        df = self._fetch_data(asset_symbol, lookback_days=lookback)
        if df.empty:
            raise ValueError(f"No data found for {asset_symbol}")

        df = self._compute_features(df)

        # We only need to store enough recent data for the sequence length needed by the model
        # But we might want a bit more buffer.
        # Store last 100 rows
        df_recent = df.tail(100)

        # Cache to Redis
        key = f"features:{asset_symbol}"
        # Store as JSON orient='index' to preserve index (timestamps)
        # However, JSON keys are strings, so we handle that in deserialization
        self.redis_client.set(key, json.dumps(df_recent.to_json(orient='columns')))

        return df_recent

    def get_training_features(self, asset_symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Get historical features for training.
        """
        # Convert period to days approximate if needed, but yfinance handles strings like "2y"
        # Logic similar to TrainService.download_yfinance but simplified
        logger.info(f"Fetching training data for {asset_symbol} over {period}")

        # For simplicity, using yfinance directly here.
        # In a real feature store, this would query an offline store (like Parquet on S3).
        df = yf.download(
            tickers=asset_symbol,
            period=period,
            interval=CONSTANTS.TIMEFRAME,
            auto_adjust=True,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError("No data fetched for training")

        df = self._compute_features(df)
        return df

    def _fetch_data(self, asset_symbol: str, lookback_days: int) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        df = yf.download(
            asset_symbol,
            start=start_date,
            end=end_date,
            interval=CONSTANTS.TIMEFRAME,
            auto_adjust=True,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Centralized feature computation logic.
        Must match exactly what was in InferenceService/TrainService.
        """
        df = df.copy()

        # Returns and volatility
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()

        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Moving averages
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma60'] = df['Close'].rolling(60).mean()

        # Price ratio
        df['price_ratio'] = df['Close'] / (df['ma20'] + 1e-8)

        # Volatility Spike
        df['vol_spike'] = (df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-8)) - 1

        # Bollinger Bands
        df['bb_upper'] = df['ma20'] + 2 * df['volatility']
        df['bb_lower'] = df['ma20'] - 2 * df['volatility']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        df.dropna(inplace=True)

        # Validate all feature columns exist
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing feature columns: {missing}")

        return df
