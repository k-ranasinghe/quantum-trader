import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from config.constants import CONSTANTS


class TripleBarrierLabeler:
    """
    Signal labeler with entry ranges and multiple take profits
    """

    def __init__(
        self,
        atr_period: int = 14,
        entry_spread_multiplier: float = 0.5,  # ATR multiplier for entry range
        sl_atr_multiplier: float = 2.0,
        tp1_atr_multiplier: float = 2.0,  # Conservative TP
        tp2_atr_multiplier: float = 4.0,  # Target TP
        tp3_atr_multiplier: float = 6.0,  # Aggressive TP
        max_hold_hours: int = 48,
        min_leverage: int = 1,
        max_leverage: int = 20,
        timeframe: str = '1h'  # '1m', '5m', '1h', '1d'
    ):
        self.atr_period = atr_period
        self.entry_spread_multiplier = entry_spread_multiplier
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp1_atr_multiplier = tp1_atr_multiplier
        self.tp2_atr_multiplier = tp2_atr_multiplier
        self.tp3_atr_multiplier = tp3_atr_multiplier
        self.max_hold_hours = max_hold_hours
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.timeframe = timeframe

        # Timeframe to hours mapping
        self.timeframe_hours = {
            '1m': 1/60,
            '5m': 5/60,
            '15m': 15/60,
            '1h': 1,
            '4h': 4,
            '1d': 24
        }

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR (Average True Range) for volatility measurement
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR as exponential moving average of TR
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()

        return atr

    def calculate_leverage(
        self,
        confidence: float,
        volatility_percentile: float
    ) -> int:
        """
        Calculate leverage based on confidence and volatility
        High confidence + low volatility = higher leverage
        """
        # Base leverage from confidence
        base_leverage = confidence * self.max_leverage

        # Adjust for volatility (inverse relationship)
        volatility_adjustment = 1.0 - (volatility_percentile * 0.5)

        leverage = base_leverage * volatility_adjustment
        leverage = int(np.clip(leverage, self.min_leverage, self.max_leverage))

        return max(leverage, self.min_leverage)

    def generate_signal_parameters(
        self,
        df: pd.DataFrame,
        idx: int,
        action: int,
        atr: pd.Series
    ) -> Dict:
        """
        Generate all signal parameters for a given entry point
        """
        current_price = df['Close'].iloc[idx]
        current_atr = atr.iloc[idx]

        # Calculate entry range (buy/sell zone)
        entry_spread = current_atr * self.entry_spread_multiplier

        if action == 0:  # Long
            entry_min = current_price - entry_spread
            entry_max = current_price

            # Stop loss below entry
            stop_loss = entry_min - (current_atr * self.sl_atr_multiplier)

            # Take profits above entry (TP1 < TP2 < TP3)
            take_profit_1 = entry_max + (current_atr * self.tp1_atr_multiplier)
            take_profit_2 = entry_max + (current_atr * self.tp2_atr_multiplier)
            take_profit_3 = entry_max + (current_atr * self.tp3_atr_multiplier)

        else:  # Short
            entry_min = current_price
            entry_max = current_price + entry_spread

            # Stop loss above entry
            stop_loss = entry_max + (current_atr * self.sl_atr_multiplier)

            # Take profits below entry (TP1 > TP2 > TP3)
            take_profit_1 = entry_min - (current_atr * self.tp1_atr_multiplier)
            take_profit_2 = entry_min - (current_atr * self.tp2_atr_multiplier)
            take_profit_3 = entry_min - (current_atr * self.tp3_atr_multiplier)

        # Calculate confidence based on future price movement
        future_window = min(self.max_hold_hours, len(df) - idx - 1)
        if future_window > 10:
            future_prices = df['Close'].iloc[idx:idx + future_window].values

            if action == 0:
                max_gain = (future_prices.max() - current_price) / current_price
                max_loss = (current_price - future_prices.min()) / current_price
            else:
                max_gain = (current_price - future_prices.min()) / current_price
                max_loss = (future_prices.max() - current_price) / current_price

            # Confidence based on gain/loss ratio
            if max_loss > 0:
                confidence = max_gain / (max_gain + max_loss)
            else:
                confidence = 0.9

            confidence = float(np.clip(confidence, 0.0, 1.0))
        else:
            confidence = 0.5

        # Calculate volatility percentile
        recent_atr = atr.iloc[max(0, idx - 100):idx].values
        if len(recent_atr) > 0:
            volatility_percentile = (current_atr - recent_atr.min()) / (recent_atr.max() - recent_atr.min() + 1e-8)
        else:
            volatility_percentile = 0.5

        # Calculate leverage
        leverage = self.calculate_leverage(confidence, volatility_percentile)

        # Risk-reward ratio
        avg_entry = (entry_min + entry_max) / 2
        risk = abs(avg_entry - stop_loss)
        reward = abs(take_profit_2 - avg_entry)  # Use TP2 as target
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        return {
            'entry_min': entry_min,
            'entry_max': entry_max,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'take_profit_3': take_profit_3,
            'leverage': leverage,
            'confidence': confidence,
            'risk_reward_ratio': risk_reward_ratio,
            'volatility': current_atr
        }

    def apply_barriers(
        self,
        df: pd.DataFrame,
        idx: int,
        params: Dict,
        action: int
    ) -> Tuple[float, float, int, str]:
        """
        Apply triple barrier method with multiple take profits
        """
        entry_avg = (params['entry_min'] + params['entry_max']) / 2

        max_idx = min(idx + self.max_hold_hours, len(df) - 1)

        for i in range(idx + 1, max_idx + 1):
            current_price = df['Close'].iloc[i]

            if action == 0:  # Long
                if current_price >= params['take_profit_3']:
                    profit = (current_price - entry_avg) / entry_avg
                    return current_price, profit, i - idx, 'tp3'
                elif current_price >= params['take_profit_2']:
                    profit = (current_price - entry_avg) / entry_avg
                    return current_price, profit, i - idx, 'tp2'
                elif current_price >= params['take_profit_1']:
                    profit = (current_price - entry_avg) / entry_avg
                    return current_price, profit, i - idx, 'tp1'
                elif current_price <= params['stop_loss']:
                    profit = (current_price - entry_avg) / entry_avg
                    return current_price, profit, i - idx, 'sl'
            else:  # Short
                if current_price <= params['take_profit_3']:
                    profit = (entry_avg - current_price) / entry_avg
                    return current_price, profit, i - idx, 'tp3'
                elif current_price <= params['take_profit_2']:
                    profit = (entry_avg - current_price) / entry_avg
                    return current_price, profit, i - idx, 'tp2'
                elif current_price <= params['take_profit_1']:
                    profit = (entry_avg - current_price) / entry_avg
                    return current_price, profit, i - idx, 'tp1'
                elif current_price >= params['stop_loss']:
                    profit = (entry_avg - current_price) / entry_avg
                    return current_price, profit, i - idx, 'sl'

        # Time barrier
        exit_price = df['Close'].iloc[max_idx]
        if action == 0:
            profit = (exit_price - entry_avg) / entry_avg
        else:
            profit = (entry_avg - exit_price) / entry_avg

        return exit_price, profit, max_idx - idx, 'time'

    def generate_labels(
        self,
        df: pd.DataFrame,
        asset_name: str = "BTC-USD"
    ) -> pd.DataFrame:
        """
        Generate complete signal labels
        """
        # Calculate ATR
        atr = self.calculate_atr(df)

        # Identify events using CUSUM filter
        event_indices = self._identify_events(df, atr)

        signals = []

        for idx in event_indices:
            if idx >= len(df) - self.max_hold_hours or idx < self.atr_period:
                continue

            # Determine action based on forward returns
            future_prices = df['Close'].iloc[idx:idx + 20].values
            current_price = df['Close'].iloc[idx]

            long_potential = (future_prices.max() - current_price) / current_price
            short_potential = (current_price - future_prices.min()) / current_price

            action = 0 if long_potential > short_potential else 1

            # Generate signal parameters
            params = self.generate_signal_parameters(df, idx, action, atr)

            # Apply barriers
            exit_price, profit, hold_bars, barrier = self.apply_barriers(
                df, idx, params, action
            )

            # Convert hold time to absolute duration
            hours_per_bar = self.timeframe_hours.get(self.timeframe, 1)
            hold_hours = hold_bars * hours_per_bar
            hold_duration = pd.Timedelta(hours=hold_hours)

            # Create signal
            signal = {
                'asset': asset_name,
                'timestamp': df.index[idx],
                'action': action,
                'entry_min': params['entry_min'],
                'entry_max': params['entry_max'],
                'stop_loss': params['stop_loss'],
                'take_profit_1': params['take_profit_1'],
                'take_profit_2': params['take_profit_2'],
                'take_profit_3': params['take_profit_3'],
                'leverage': params['leverage'],
                'confidence': params['confidence'],
                'risk_reward_ratio': params['risk_reward_ratio'],
                'expected_hold_hours': hold_hours,
                'volatility': params['volatility'],
                'exit_price': exit_price,
                'profit': profit,
                'barrier_hit': barrier,
                'actual_hold_hours': hold_bars * hours_per_bar
            }

            signals.append(signal)

        return pd.DataFrame(signals)

    def create_training_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        asset_name: str = "BTC-USD"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create sequences and target labels for training
        """
        # Generate labels
        labels_df = self.generate_labels(df, asset_name)

        X, y = [], {
            'action': [],
            'entry_range': [],
            'stop_loss': [],
            'take_profits': [],
            'leverage': [],
            'hold_time': [],
            'confidence': [],
            'volatility': []
        }

        for _, label in labels_df.iterrows():
            idx = df.index.get_loc(label['timestamp'])

            if idx < sequence_length:
                continue

            # Get sequence
            seq = df[CONSTANTS.FEATURE_COLS].iloc[idx - sequence_length:idx].values
            X.append(seq)

            # Current price for normalization
            current_price = df['Close'].iloc[idx]

            # Get targets (as percentage offsets)
            y['action'].append(label['action'])

            entry_min_offset = (label['entry_min'] - current_price) / current_price
            entry_max_offset = (label['entry_max'] - current_price) / current_price
            y['entry_range'].append([entry_min_offset, entry_max_offset])

            sl_offset = (label['stop_loss'] - current_price) / current_price
            y['stop_loss'].append(sl_offset)

            tp1_offset = (label['take_profit_1'] - current_price) / current_price
            tp2_offset = (label['take_profit_2'] - current_price) / current_price
            tp3_offset = (label['take_profit_3'] - current_price) / current_price
            y['take_profits'].append([tp1_offset, tp2_offset, tp3_offset])

            y['leverage'].append(label['leverage'])
            y['hold_time'].append(label['expected_hold_hours'])
            y['confidence'].append(label['confidence'])
            y['volatility'].append(label['volatility'])

        # Convert to arrays
        X = np.array(X, dtype=np.float32)
        y = {
            'action': torch.LongTensor(y['action']),
            'entry_range': torch.FloatTensor(y['entry_range']),
            'stop_loss': torch.FloatTensor(y['stop_loss']).unsqueeze(-1),
            'take_profits': torch.FloatTensor(y['take_profits']),
            'leverage': torch.FloatTensor(y['leverage']).unsqueeze(-1),
            'hold_time': torch.FloatTensor(y['hold_time']).unsqueeze(-1),
            'confidence': torch.FloatTensor(y['confidence']).unsqueeze(-1),
            'volatility': torch.FloatTensor(y['volatility']).unsqueeze(-1)
        }

        return X, y

    def _identify_events(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        threshold_multiplier: float = 1.0
    ) -> np.ndarray:
        """
        Identify potential trading events using CUSUM filter
        """
        returns = df['Close'].pct_change().fillna(0)

        # Dynamic threshold based on ATR
        threshold = atr * threshold_multiplier

        pos_cusum = np.zeros(len(returns))
        neg_cusum = np.zeros(len(returns))
        events = []

        for i in range(1, len(returns)):
            if pd.isna(threshold.iloc[i]) or threshold.iloc[i] == 0:
                continue

            pos_cusum[i] = max(0, pos_cusum[i-1] + returns.iloc[i])
            neg_cusum[i] = min(0, neg_cusum[i-1] + returns.iloc[i])

            if pos_cusum[i] > threshold.iloc[i]:
                events.append(i)
                pos_cusum[i] = 0
            elif neg_cusum[i] < -threshold.iloc[i]:
                events.append(i)
                neg_cusum[i] = 0

        return np.array(events)
