import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import os
from datetime import datetime, timedelta
import warnings

from config.constants import CONSTANTS
from config.models import Signal as SignalDTO
from config.settings import settings
from controllers.ensemble import SignalEnsemble
from environments.dummy_env import DummySignalEnv
from generators.rl_signal_generator import RLSignalGenerator
from generators.signal_generator import SignalGenerator
from services.feature_store import FeatureStore
from utilities.logger import logger

warnings.filterwarnings("ignore")


class InferenceService:
    """
    Runtime system for generating trading signals
    """

    def __init__(
            self,
            asset_symbol: str,
            model_registry_path: str = settings.MODEL_REGISTRY_DIR,
            timeframe: str = CONSTANTS.TIMEFRAME,
            seq_len: int = CONSTANTS.SEQ_LEN,
            device: str = CONSTANTS.DEVICE
    ):
        self.asset = asset_symbol
        self.timeframe = timeframe
        self.seq_len = seq_len
        self.device = device
        self.registry_path = model_registry_path
        self.feature_store = FeatureStore()

        # Load artifacts
        try:
            self._load_artifacts()
            logger.info(f"Inference Service ready for {self.asset}")
        except Exception as e:
            logger.error(f"Failed to initialize Inference Service for {self.asset}: {e}")
            raise e

    def _load_artifacts(self):
        """Load scaler and models from registry"""
        # In a real system, we might query the DB to get the specific version path
        # For now, we assume a standard naming convention in the shared volume

        # Load scaler
        scaler_path = os.path.join(self.registry_path, "scaler.pkl")
        if not os.path.exists(scaler_path):
            # Try to find it in the local dir if not in registry (fallback)
            scaler_path = "scaler.pkl"

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            logger.warning(f"Scaler not found at {scaler_path}, creating new one (WARNING: Uncalibrated)")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()

        # Load models
        # We look for files named like "best_EnhancedModel1.pth"
        model_paths = {
            'model1': os.path.join(self.registry_path, "best_EnhancedModel1.pth"),
            'model2': os.path.join(self.registry_path, "best_EnhancedModel2.pth"),
            'rl_agent': os.path.join(self.registry_path, "rl_agent")  # This is a zip usually
        }

        self.models = {}

        # Load PyTorch models
        for name, path in model_paths.items():
            if name == 'rl_agent':
                continue

            if os.path.exists(path):
                model = SignalGenerator(input_dim=CONSTANTS.INPUT_DIM, seq_len=self.seq_len)
                try:
                    model.load(path, device=self.device)
                    model.to(self.device)
                    model.eval()
                    self.models[name] = model
                except Exception as e:
                    logger.error(f"Error loading {name} from {path}: {e}")
            else:
                logger.warning(f"Model artifact {name} not found at {path}")

        # Load RL Agent
        # Note: StableBaselines3 loads from zip
        rl_path = model_paths['rl_agent']
        if os.path.exists(rl_path + ".zip") or os.path.exists(rl_path):
            try:
                rl_agent = RLSignalGenerator(
                    env_fn=lambda: DummySignalEnv(),
                    learning_rate=3e-4
                )
                rl_agent.load(rl_path)
                self.models['rl_agent'] = rl_agent
            except Exception as e:
                logger.error(f"Error loading RL agent: {e}")

        # Load helper models
        self.regime_detector = xgb.XGBClassifier()
        regime_path = os.path.join(self.registry_path, "regime_detector.json")
        if os.path.exists(regime_path):
            self.regime_detector.load_model(regime_path)

        self.meta_model = xgb.XGBClassifier()
        meta_path = os.path.join(self.registry_path, "meta_model.json")
        if os.path.exists(meta_path):
            self.meta_model.load_model(meta_path)

        # Create ensemble
        self.ensemble = SignalEnsemble(
            models=self.models,
            regime_detector=self.regime_detector,
            meta_model=self.meta_model,
            asset_name=self.asset,
            timeframe=self.timeframe
        )

    def generate_signal(self) -> SignalDTO:
        """
        Generate a trading signal from current market data
        """
        # Fetch features via Feature Store (Online)
        df = self.feature_store.get_online_features(self.asset)

        if len(df) < self.seq_len:
            raise ValueError(f"Insufficient data: need at least {self.seq_len} bars, got {len(df)}")

        # Calculate ATR
        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]
        current_price = df['Close'].iloc[-1]

        # Scale features
        # Note: We must use the same scaler fitted during training
        df_scaled = df.copy()
        # Ensure we only try to scale columns that exist and were fitted
        try:
            df_scaled[CONSTANTS.FEATURE_COLS] = self.scaler.transform(df[CONSTANTS.FEATURE_COLS])
        except Exception as e:
            logger.warning(f"Scaler transform failed (likely feature mismatch), using raw data: {e}")
            # Fallback/Risk: might produce bad signals but prevents crash

        # Get sequence
        observation = df_scaled[CONSTANTS.FEATURE_COLS].iloc[-self.seq_len:].values

        # Generate signal
        logger.info(f"Generating signal for {self.asset}...")

        # Generate ID
        signal_id = f"{self.asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not self.models:
            logger.warning("No models loaded! returning dummy signal.")
            return self._create_dummy_signal(signal_id, current_price)

        try:
            signal = self.ensemble.generate_signal(
                observation=observation,
                current_price=current_price,
                current_atr=current_atr,
                market_data=df,
                signal_id=signal_id
            )
            return signal
        except Exception as e:
            logger.error(f"Ensemble generation failed: {e}")
            return self._create_dummy_signal(signal_id, current_price)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr

    def _create_dummy_signal(self, id, price):
        return SignalDTO(
            id=id, asset=self.asset, action="Hold",
            entry_min=price, entry_max=price, stop_loss=price * 0.95,
            take_profit_1=price * 1.05, take_profit_2=price * 1.1, take_profit_3=price * 1.15,
            leverage=1, confidence=0.0, models_agreement=0.0, risk_reward_ratio=0.0,
            regime="Unknown", volatility=0.0, expected_hold_duration="0h"
        )
