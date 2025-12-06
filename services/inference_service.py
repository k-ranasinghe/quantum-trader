import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import pickle
import json
from datetime import datetime, timedelta
import warnings

from config.constants import CONSTANTS
from controllers.ensemble import SignalEnsemble
from environments.dummy_env import DummySignalEnv
from generators.rl_signal_generator import RLSignalGenerator
from generators.signal_generator import SignalGenerator

warnings.filterwarnings("ignore")


class InferenceService:
    """
    Runtime system for generating trading signals
    """

    def __init__(
            self,
            model_paths: dict,
            asset: str = CONSTANTS.ASSET,
            timeframe: str = CONSTANTS.TIMEFRAME,
            seq_len: int = CONSTANTS.SEQ_LEN,
            device: str = CONSTANTS.DEVICE
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.seq_len = seq_len
        self.device = device

        # Load scaler
        print("Loading scaler...")
        with open(f'{CONSTANTS.SAVE_PATH}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        # Load models
        print("Loading models...")
        self.models = self._load_models(model_paths)

        # Load regime detector
        print("Loading regime detector...")
        self.regime_detector = xgb.XGBClassifier()
        self.regime_detector.load_model(f'{CONSTANTS.SAVE_PATH}/regime_detector.json')

        # Load meta model
        print("Loading meta model...")
        self.meta_model = xgb.XGBClassifier()
        self.meta_model.load_model(f'{CONSTANTS.SAVE_PATH}/meta_model.json')

        # Create ensemble
        print("Creating ensemble...")
        self.ensemble = SignalEnsemble(
            models=self.models,
            regime_detector=self.regime_detector,
            meta_model=self.meta_model,
            asset_name=self.asset,
            timeframe=self.timeframe
        )

        print("Runtime system ready!")

    def _load_models(self, model_paths: dict) -> dict:
        """Load all trained models"""
        models = {}

        for name, path in model_paths.items():
            if 'rl' in name.lower():
                # Load RL agent
                rl_agent = RLSignalGenerator(
                    env_fn=lambda: DummySignalEnv(),  # Dummy env
                    learning_rate=3e-4
                )
                rl_agent.load(path)
                models[name] = rl_agent
            else:
                # Load PyTorch model
                model = SignalGenerator(input_dim=CONSTANTS.INPUT_DIM, seq_len=self.seq_len)
                model.load(path, device=self.device)
                model.to(self.device)
                model.eval()
                models[name] = model

        return models

    def fetch_market_data(self, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch recent market data"""
        print(f"Fetching {self.asset} data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        df = yf.download(
            self.asset,
            start=start_date,
            end=end_date,
            interval=self.timeframe,
            auto_adjust=True,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features"""
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
        df['price_ratio'] = df['Close'] / (df['ma20'] + 1e-8)
        df['vol_spike'] = (df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-8)) - 1

        # Bollinger Bands
        df['bb_upper'] = df['ma20'] + 2 * df['volatility']
        df['bb_lower'] = df['ma20'] - 2 * df['volatility']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        df.dropna(inplace=True)

        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def generate_signal(self, force_refresh: bool = False):
        """
        Generate a trading signal from current market data

        Args:
            force_refresh: If True, fetch new data; otherwise use cached

        Returns:
            EnhancedSignal object
        """
        # Fetch and prepare data
        df = self.fetch_market_data(lookback_days=60)
        df = self.prepare_features(df)

        if len(df) < self.seq_len + 20:
            raise ValueError(f"Insufficient data: need at least {self.seq_len + 20} bars")

        # Calculate ATR
        atr = self.calculate_atr(df)
        current_atr = atr.iloc[-1]

        # Get current price
        current_price = df['Close'].iloc[-1]

        # Scale features
        df_scaled = df.copy()
        df_scaled[CONSTANTS.FEATURE_COLS] = self.scaler.transform(df[CONSTANTS.FEATURE_COLS])

        # Get sequence
        observation = df_scaled[CONSTANTS.FEATURE_COLS].iloc[-self.seq_len:].values

        # Generate signal
        print("\nGenerating signal...")
        signal = self.ensemble.generate_signal(
            observation=observation,
            current_price=current_price,
            current_atr=current_atr,
            market_data=df,
            signal_id=f"{self.asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        return signal

    def generate_and_display_signal(self):
        """Generate signal and display in formatted output"""
        signal = self.generate_signal()

        print("\n" + "=" * 70)
        print("TRADING SIGNAL GENERATED")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Signal ID: {signal.id}")
        print("-" * 70)
        print(f"Asset:     {signal.asset}")
        print(f"Action:    {signal.action}")
        print(f"Regime:    {signal.regime}")
        print("-" * 70)
        print("ENTRY:")
        print(f"  Range:   ${signal.entry_min:.2f} - ${signal.entry_max:.2f}")
        print("-" * 70)
        print("EXIT TARGETS:")
        print(f"  Stop Loss:      ${signal.stop_loss:.2f}")
        print(f"  Take Profit 1:  ${signal.take_profit_1:.2f}  (Conservative)")
        print(f"  Take Profit 2:  ${signal.take_profit_2:.2f}  (Target)")
        print(f"  Take Profit 3:  ${signal.take_profit_3:.2f}  (Aggressive)")
        print("-" * 70)
        print("RISK MANAGEMENT:")
        print(f"  Leverage:         {signal.leverage}x")
        print(f"  Risk/Reward:      {signal.risk_reward_ratio:.2f}")
        print(f"  Confidence:       {signal.confidence:.1%}")
        print(f"  Models Agreement: {signal.models_agreement:.1%}")
        print("-" * 70)
        print("TIMING:")
        print(f"  Expected Hold:    {signal.expected_hold_duration}")
        print(f"  Volatility (ATR): ${signal.volatility:.2f}")
        print("=" * 70)

        return signal

    def save_signal(self, signal, path, filename: str = None):
        """Save signal to JSON file"""
        if filename is None:
            filename = f"{path}/signal_{signal.id}.json"

        with open(filename, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)

        print(f"\nSignal saved to {filename}")

    def batch_generate_signals(self, assets: list, save: bool = True):
        """Generate signals for multiple assets"""
        signals = []

        for asset in assets:
            print(f"\n{'=' * 70}")
            print(f"Processing {asset}...")
            print('=' * 70)

            try:
                self.asset = asset
                self.ensemble.asset_name = asset

                signal = self.generate_signal()
                signals.append(signal)

                self.generate_and_display_signal()

                if save:
                    self.save_signal(signal)

            except Exception as e:
                print(f"Error generating signal for {asset}: {e}")
                continue

        return signals
