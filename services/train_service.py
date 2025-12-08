import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import mlflow
import mlflow.pytorch
import mlflow.xgboost
import warnings
import pickle
from typing import Dict, Tuple, Optional

from config.constants import CONSTANTS
from config.settings import settings
from controllers.ensemble import SignalEnsemble
from environments.signal_env import SignalEnv
from generators.rl_signal_generator import RLSignalGenerator
from generators.signal_generator import SignalGenerator
from labellers.triple_barrier import TripleBarrierLabeler
from models.loss_function import TradingLoss
from services.feature_store import FeatureStore
from utilities.logger import logger

warnings.filterwarnings("ignore")


class TrainService:
    """
    End-to-end training service with MLFlow integration.
    """

    def __init__(
            self,
            asset: str = CONSTANTS.ASSET,
            model_registry_path: str = settings.MODEL_REGISTRY_DIR,
            timeframe: str = CONSTANTS.TIMEFRAME,
            period: str = CONSTANTS.PERIOD,
            seq_len: int = CONSTANTS.SEQ_LEN,
            device: str = CONSTANTS.DEVICE
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.period = period
        self.seq_len = seq_len
        self.device = device
        self.registry_path = model_registry_path
        self.feature_store = FeatureStore()

        self.df: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.labeler: Optional[TripleBarrierLabeler] = None

        # Models
        self.models: Dict[str, torch.nn.Module] = {}
        self.rl_agent: Optional[RLSignalGenerator] = None
        self.regime_detector: Optional[xgb.XGBClassifier] = None
        self.meta_model: Optional[xgb.XGBClassifier] = None
        self.ensemble: Optional[SignalEnsemble] = None

        logger.info(f"Training Service initialized for {asset}")
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"quant_trader_{asset}")

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Fetch data from Feature Store"""
        logger.info(f"Loading data for {self.asset}...")
        self.df = self.feature_store.get_training_features(self.asset, self.period)
        logger.info(f"Final data shape: {self.df.shape}")
        return self.df

    def create_labeled_dataset(self) -> Tuple[np.ndarray, dict, StandardScaler]:
        """Generate features, labels, and fit scaler"""
        logger.info("Creating labeled dataset...")

        self.labeler = TripleBarrierLabeler(
            atr_period=14,
            entry_spread_multiplier=0.5,
            sl_atr_multiplier=2.0,
            tp1_atr_multiplier=2.0,
            tp2_atr_multiplier=4.0,
            tp3_atr_multiplier=6.0,
            max_hold_hours=48,
            timeframe=self.timeframe
        )

        self.scaler = StandardScaler()
        self.df[CONSTANTS.FEATURE_COLS] = self.scaler.fit_transform(self.df[CONSTANTS.FEATURE_COLS])

        X, y = self.labeler.create_training_data(
            self.df, sequence_length=self.seq_len, asset_name=self.asset
        )

        logger.info(f"Labeled samples: {len(X)}")
        return X, y, self.scaler

    def train_model(self, model, X_train, y_train, X_val, y_val, name):
        """Train PyTorch model and log to MLFlow"""
        criterion = TradingLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CONSTANTS.LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), y_train['action'], y_train['entry_range'],
            y_train['stop_loss'], y_train['take_profits'], y_train['leverage'],
            y_train['hold_time'], y_train['confidence'], y_train['volatility']
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), y_val['action'], y_val['entry_range'],
            y_val['stop_loss'], y_val['take_profits'], y_val['leverage'],
            y_val['hold_time'], y_val['confidence'], y_val['volatility']
        )

        train_loader = DataLoader(train_dataset, batch_size=CONSTANTS.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONSTANTS.BATCH_SIZE)

        best_loss = float('inf')
        patience = 0

        model.to(self.device)

        logger.info(f"Training {name}...")

        # Determine save filename
        # We need this file to persist for the Celery task to copy it later
        save_path = f"{self.registry_path}/best_{name}.pth"

        with mlflow.start_run(run_name=f"train_{name}", nested=True):
            mlflow.log_param("model_name", name)
            mlflow.log_param("batch_size", CONSTANTS.BATCH_SIZE)
            mlflow.log_param("learning_rate", CONSTANTS.LEARNING_RATE)

            for epoch in range(CONSTANTS.EPOCHS):
                model.train()
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)

                model.eval()
                val_loss = self._validate_epoch(model, val_loader, criterion)

                scheduler.step(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    # Save best model locally
                    torch.save(model.state_dict(), save_path)
                else:
                    patience += 1

                if epoch % 10 == 0:
                    logger.info(f"  Epoch {epoch + 1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

                if patience >= CONSTANTS.PATIENCE:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

            # Load best weights to return
            model.load_state_dict(torch.load(save_path))
            # We do NOT remove save_path here, so the task can pick it up.

            # Log model to MLFlow
            mlflow.pytorch.log_model(model, name)
            mlflow.log_metric("best_val_loss", best_loss)

        return model, {"best_val_loss": best_loss}

    def _train_epoch(self, model, loader, optimizer, criterion):
        losses = []
        for batch in loader:
            x = batch[0].to(self.device)
            targets = {k: v.to(self.device) for k, v in zip(
                ['action', 'entry_range', 'stop_loss', 'take_profits', 'leverage', 'hold_time', 'confidence',
                 'volatility'],
                batch[1:]
            )}
            optimizer.zero_grad()
            pred = model(x)
            loss, _ = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def _validate_epoch(self, model, loader, criterion):
        losses = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                targets = {k: v.to(self.device) for k, v in zip(
                    ['action', 'entry_range', 'stop_loss', 'take_profits', 'leverage', 'hold_time', 'confidence',
                     'volatility'],
                    batch[1:]
                )}
                pred = model(x)
                loss, _ = criterion(pred, targets)
                losses.append(loss.item())
        return np.mean(losses)

    def train_regime_detector(self):
        """Train market regime detector"""
        logger.info("Training regime detector...")

        # Prepare data (same logic as before)
        features = []
        labels = []

        for i in range(100, len(self.df) - 20):
            window = self.df.iloc[i - 100:i]
            feat = [
                window['returns'].mean(),
                window['returns'].std(),
                window['volatility'].mean(),
                window['volatility'].max(),
                np.percentile(window['returns'], 25),
                np.percentile(window['returns'], 75)
            ]
            features.append(feat)
            future_vol = self.df['volatility'].iloc[i:i + 20].mean()
            current_vol = window['volatility'].mean()
            if future_vol < current_vol * 0.8:
                label = 0
            elif future_vol > current_vol * 1.2:
                label = 2
            else:
                label = 1
            labels.append(label)

        X = np.array(features)
        y = np.array(labels)

        self.regime_detector = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

        with mlflow.start_run(run_name="train_regime_detector", nested=True):
            self.regime_detector.fit(X, y)
            acc = self.regime_detector.score(X, y)
            mlflow.log_metric("accuracy", acc)
            mlflow.xgboost.log_model(self.regime_detector, "regime_detector")
            logger.info(f"Regime detector accuracy: {acc:.3f}")

            # Save locally for task copy
            self.regime_detector.save_model(f"{self.registry_path}/regime_detector.json")

        return self.regime_detector

    def train_meta_model(self, models: dict, X_val, y_val):
        """Train meta-model for confidence"""
        logger.info("Training meta-model...")

        # ... (Prepare meta features) ...
        meta_features = []
        meta_labels = []

        for i in range(len(X_val)):
            x = torch.FloatTensor(X_val[i]).unsqueeze(0).to(self.device)
            predictions = []
            for model in models.values():
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    probs = torch.softmax(pred['action_logits'], dim=1).cpu().numpy()[0]
                    conf = pred['confidence'].item()
                    predictions.extend([probs[0], probs[1], conf])
            meta_features.append(predictions)
            label = 1 if y_val['confidence'][i].item() > 0.5 else 0
            meta_labels.append(label)

        meta_X = np.array(meta_features)
        meta_y = np.array(meta_labels)

        with mlflow.start_run(run_name="train_meta_model", nested=True):
            if len(np.unique(meta_y)) < 2:
                self.meta_model = DummyClassifier(strategy='constant', constant=np.unique(meta_y)[0])
                self.meta_model.fit(meta_X, meta_y)
            else:
                self.meta_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
                self.meta_model.fit(meta_X, meta_y)
                acc = self.meta_model.score(meta_X, meta_y)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(self.meta_model, "meta_model")
                logger.info(f"Meta-model accuracy: {acc:.3f}")

            # Save locally for task copy
            # XGBoost save
            if hasattr(self.meta_model, "save_model"):
                self.meta_model.save_model(f"{self.registry_path}/meta_model.json")
            else:
                # DummyClassifier or other sklearn
                with open(f"{self.registry_path}/meta_model.json", "wb") as f:
                    pickle.dump(self.meta_model, f)

        return self.meta_model

    def train_rl_agent(self):
        logger.info("Training RL agent...")

        data_scaled = self.scaler.transform(self.df[CONSTANTS.FEATURE_COLS])
        prices = self.df['Close'].values
        atr = self._calculate_atr(self.df).values

        def env_fn():
            return SignalEnv(data=data_scaled, prices=prices, atr=atr, seq_len=self.seq_len)

        self.rl_agent = RLSignalGenerator(env_fn, learning_rate=3e-4)

        with mlflow.start_run(run_name="train_rl_agent", nested=True):
            self.rl_agent.train(total_timesteps=100000)

            # Define paths
            rl_agent_dir = f"{self.registry_path}/rl_agent"
            rl_agent_zip = f"{self.registry_path}/rl_agent.zip"

            os.makedirs(rl_agent_dir, exist_ok=True)
            self.rl_agent.save(rl_agent_dir)
            mlflow.log_artifacts(rl_agent_dir, artifact_path="rl_agent")
            shutil.make_archive(rl_agent_dir, 'zip', rl_agent_dir)
            logger.info(f"RL agent saved to {rl_agent_zip} and logged to MLflow")

        return self.rl_agent

    def _calculate_atr(self, df, period=14):
        high, low, close = df['High'], df['Low'], df['Close']
        tr0 = high - low
        tr1 = np.abs(high - close.shift(1))
        tr2 = np.abs(low - close.shift(1))
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def run_full_training_pipeline(self):
        """Execute the full training pipeline and save artifacts"""
        with mlflow.start_run(run_name=f"pipeline_{self.asset}"):
            # 1. Load Data
            self.load_and_prepare_data()

            # 2. Create Dataset
            X, y, self.scaler = self.create_labeled_dataset()

            # Save scaler
            scaler_path = os.path.join(self.registry_path, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            mlflow.log_artifact(scaler_path)
            # Keep scaler.pkl for task copy

            # 3. Split
            X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split_dict(X, y)

            # 4. Train Models
            input_dim = X_train.shape[2]

            model1 = SignalGenerator(input_dim=input_dim, seq_len=self.seq_len)
            self.model1, _ = self.train_model(model1, X_train, y_train, X_val, y_val, "EnhancedModel1")

            model2 = SignalGenerator(input_dim=input_dim, seq_len=self.seq_len, d_model=512)
            self.model2, _ = self.train_model(model2, X_train, y_train, X_val, y_val, "EnhancedModel2")

            # 5. Train RL
            self.rl_agent = self.train_rl_agent()

            # 6. Train Helpers
            self.regime_detector = self.train_regime_detector()
            self.meta_model = self.train_meta_model(
                {'model1': self.model1, 'model2': self.model2}, X_val, y_val
            )

            logger.info("Training pipeline completed successfully.")
            return True

    @staticmethod
    def train_test_split_dict(X, y_dict, val_ratio=0.15, test_ratio=0.15):
        n = len(X)
        val_start = int(n * (1 - test_ratio - val_ratio))
        test_start = int(n * (1 - test_ratio))

        split = lambda arr: (arr[:val_start], arr[val_start:test_start], arr[test_start:])
        X_train, X_val, X_test = split(X)
        y_train = {k: split(v)[0] for k, v in y_dict.items()}
        y_val = {k: split(v)[1] for k, v in y_dict.items()}
        y_test = {k: split(v)[2] for k, v in y_dict.items()}

        logger.info(f"Train/Val/Test split: {len(X_train)} / {len(X_val)} / {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
