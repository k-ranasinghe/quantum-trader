import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import RobustScaler


class FeatureEngineer:
    """
    Comprehensive feature engineering for financial data
    """

    def __init__(self):
        self.scalers = {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        df = df.copy()

        # Trend indicators
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()

        # Momentum indicators
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Volatility indicators
        bb = BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()

        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Remove NaN values
        df = df.bfill().ffill()

        # Log to check for NaN values
        if df.isna().any().any():
            print(f"Warning: NaN values found after filling: {df.isna().sum()}")

        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators"""
        df = df.copy()

        # Trend strength
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['close']

        # Volatility regime
        vol_percentile = df['volatility'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x)
        )
        df['vol_regime'] = pd.cut(vol_percentile, bins=3, labels=['low', 'medium', 'high'], duplicates='drop')

        return df

    def scale_features(self, df: pd.DataFrame, feature_cols: list, asset: str) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        df = df.copy()

        if asset not in self.scalers:
            self.scalers[asset] = RobustScaler()
            df[feature_cols] = self.scalers[asset].fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scalers[asset].transform(df[feature_cols])

        return df