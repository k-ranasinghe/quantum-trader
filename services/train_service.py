import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict,Tuple, Optional

from config.constants import CONSTANTS
from environments.signal_env import SignalEnv
from generators.rl_signal_generator import RLSignalGenerator
from labellers.triple_barrier import TripleBarrierLabeler
from models.loss_function import TradingLoss
from controllers.ensemble import SignalEnsemble

warnings.filterwarnings("ignore")


class TrainService:
    """
    End-to-end training service for the trading signal generation system.
    """

    def __init__(
        self,
        asset: str = CONSTANTS.ASSET,
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

        self.df: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.labeler: Optional[TripleBarrierLabeler] = None
        self.models: Dict[str, torch.nn.Module] = {}
        self.rl_agent: Optional[RLSignalGenerator] = None
        self.regime_detector: Optional[xgb.XGBClassifier] = None
        self.meta_model: Optional[xgb.XGBClassifier] = None
        self.ensemble: Optional[SignalEnsemble] = None

        print("Training Service initialized.")

    def download_yfinance(self, ticker: str) -> pd.DataFrame:
        """Robust yfinance downloader with chunking for large periods"""
        end_date = datetime.now()
        period_lower = self.period.lower()

        if period_lower == "max":
            start_date = datetime(1970, 1, 1)
        elif period_lower == "ytd":
            start_date = datetime(end_date.year, 1, 1)
        elif period_lower.endswith("d"):
            start_date = end_date - timedelta(days=int(period_lower[:-1]))
        elif period_lower.endswith("mo"):
            start_date = end_date - relativedelta(months=int(period_lower[:-2]))
        elif period_lower.endswith("y"):
            start_date = end_date - relativedelta(years=int(period_lower[:-1]))
        else:
            start_date = end_date - timedelta(days=int(period_lower or 730))

        max_delta = CONSTANTS.INTERVAL_MAX_PERIOD.get(self.timeframe, timedelta(days=730))
        print(f"Downloading {ticker} | {self.timeframe} | from {start_date.date()} to {end_date.date()}")

        dfs = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + max_delta, end_date)
            chunk = yf.download(
                tickers=ticker,
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval=self.timeframe,
                auto_adjust=True,
                progress=False,
                threads=True,
                repair=True
            )
            if not chunk.empty:
                dfs.append(chunk)
                print(f"  Fetched {len(chunk)} rows → {current_start.date()} to {current_end.date()}")
            current_start = current_end + timedelta(seconds=1)

        if not dfs:
            raise ValueError("No data downloaded")

        df = pd.concat(dfs)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Download and engineer features"""
        print(f"\nLoading data for {self.asset}...")
        self.df = self.download_yfinance(self.asset)

        print(f"Raw data: {len(self.df)} rows")

        # Feature engineering (same as inference)
        self.df['returns'] = self.df['Close'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(20).std()

        delta = self.df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        self.df['rsi'] = 100 - (100 / (1 + rs))

        self.df['ma20'] = self.df['Close'].rolling(20).mean()
        self.df['ma60'] = self.df['Close'].rolling(60).mean()
        self.df['price_ratio'] = self.df['Close'] / (self.df['ma20'] + 1e-8)
        self.df['vol_spike'] = (self.df['volatility'] / (self.df['volatility'].rolling(50).mean() + 1e-8)) - 1

        self.df['bb_upper'] = self.df['ma20'] + 2 * self.df['volatility']
        self.df['bb_lower'] = self.df['ma20'] - 2 * self.df['volatility']
        self.df['bb_position'] = (self.df['Close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'] + 1e-8)

        self.df.dropna(inplace=True)
        print(f"Final data shape: {self.df.shape}")
        return self.df

    def create_labeled_dataset(self) -> Tuple[np.ndarray, dict, StandardScaler]:
        """Generate features, labels, and fit scaler"""
        print("\nCreating labeled dataset...")
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

        print(f"Labeled samples: {len(X)}")
        print(f"Action distribution: {torch.bincount(y['action'])}")
        return X, y, self.scaler

    def train_model(self, model, X_train, y_train, X_val, y_val, name):
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

        print(f"\nTraining {name}...")
        for epoch in range(CONSTANTS.EPOCHS):
            model.train()
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            model.eval()
            val_loss = self._validate_epoch(model, val_loader, criterion)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f"{CONSTANTS.SAVE_PATH}/best_{name}.pth")
            else:
                patience += 1

            if epoch % 10 == 0 or epoch == CONSTANTS.EPOCHS - 1:
                print(f"  Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if patience >= CONSTANTS.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(f"{CONSTANTS.SAVE_PATH}/best_{name}.pth"))
        return model, {"best_val_loss": best_loss}

    def _train_epoch(self, model, loader, optimizer, criterion):
        losses = []
        for batch in loader:
            x = batch[0].to(self.device)
            targets = {k: v.to(self.device) for k, v in zip(
                ['action', 'entry_range', 'stop_loss', 'take_profits', 'leverage', 'hold_time', 'confidence', 'volatility'],
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
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                targets = {k: v.to(self.device) for k, v in zip(
                    ['action', 'entry_range', 'stop_loss', 'take_profits', 'leverage', 'hold_time', 'confidence', 'volatility'],
                    batch[1:]
                )}
                pred = model(x)
                loss, _ = criterion(pred, targets)
                losses.append(loss.item())
        return np.mean(losses)

    def train_regime_detector(self):
        """Train market regime detector"""
        print("\nTraining regime detector...")

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

        self.regime_detector = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.regime_detector.fit(X, y)

        print(f"Regime detector accuracy: {self.regime_detector.score(X, y):.3f}")
        return self.regime_detector

    def train_meta_model(self, models: dict, X_val, y_val):
        """Train meta-model for confidence"""
        print("\nTraining meta-model...")

        meta_features = []
        meta_labels = []

        for i in range(len(X_val)):
            x = torch.FloatTensor(X_val[i]).unsqueeze(0).to(CONSTANTS.DEVICE)

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

        unique_classes = np.unique(meta_y)
        print(f"Meta-label distribution: {np.bincount(meta_y)}")

        if len(unique_classes) < 2:
            print("Only one class in meta-labels → using dummy classifier (always predict high confidence)")
            self.meta_model = DummyClassifier(strategy='constant', constant=unique_classes[0])
            self.meta_model.fit(meta_X, meta_y)
        else:
            self.meta_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.meta_model.fit(meta_X, meta_y)
            print(f"Meta-model accuracy: {self.meta_model.score(meta_X, meta_y):.3f}")
        return self.meta_model

    def train_rl_agent(self):
        print("\nTraining RL agent...")
        data_scaled = self.scaler.transform(self.df[CONSTANTS.FEATURE_COLS])
        prices = self.df['Close'].values
        atr = self._calculate_atr(self.df).values

        def env_fn():
            return SignalEnv(data=data_scaled, prices=prices, atr=atr, seq_len=self.seq_len)

        self.rl_agent = RLSignalGenerator(env_fn, learning_rate=3e-4)
        self.rl_agent.train(total_timesteps=100000)
        return self.rl_agent

    def _calculate_atr(self, df, period=14):
        high, low, close = df['High'], df['Low'], df['Close']
        tr0 = high - low
        tr1 = np.abs(high - close.shift(1))
        tr2 = np.abs(low - close.shift(1))
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def evaluate_ensemble(self, ensemble, X_test, y_test, df_test):
        """Evaluate ensemble"""
        print("\nEvaluating Ensemble...")
        self.ensemble = ensemble

        # Calculate ATR for test set
        atr = self._calculate_atr(df_test)

        signals = []
        for i in range(len(X_test)):
            try:
                idx = CONSTANTS.SEQ_LEN + i
                if idx >= len(df_test):
                    break

                signal = self.ensemble.generate_signal(
                    observation=X_test[i],
                    current_price=df_test['Close'].iloc[idx],
                    current_atr=atr.iloc[idx],
                    market_data=df_test.iloc[:idx],
                    signal_id=f"TEST_{i}"
                )
                signals.append(signal)
            except Exception as e:
                print(f"Error generating signal {i}: {e}")
                continue

        print(f"\nGenerated {len(signals)} signals")

        # Analyze
        actions = [1 if s.action == "Long" else 0 for s in signals]
        print(f"Actions: Long={sum(actions)}, Short={len(actions) - sum(actions)}")
        print(f"Avg Confidence: {np.mean([s.confidence for s in signals]):.3f}")
        print(f"Avg Agreement: {np.mean([s.models_agreement for s in signals]):.3f}")
        print(f"Avg Leverage: {np.mean([s.leverage for s in signals]):.1f}")
        print(f"Avg Risk/Reward: {np.mean([s.risk_reward_ratio for s in signals]):.2f}")

        return signals

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

        print(f"Train/Val/Test split: {len(X_train)} / {len(X_val)} / {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
