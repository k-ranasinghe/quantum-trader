import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict


class RLSignalGenerator:
    """Wrapper for RL-based signal generation"""

    def __init__(
        self,
        env_fn,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64
    ):
        self.env_fn = env_fn
        self.model = PPO(
            "MlpPolicy",
            DummyVecEnv([env_fn]),
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=1
        )

    def train(self, total_timesteps: int = 100000):
        """Train the RL agent"""
        print("Training RL Signal Generator...")
        self.model.learn(total_timesteps=total_timesteps)
        print("RL training complete")

    def predict_signal(
        self,
        observation: np.ndarray,
        current_price: float,
        current_atr: float
    ) -> Dict:
        """
        Generate signal from observation
        """
        action, _ = self.model.predict(observation, deterministic=True)

        # Decode action to signal parameters
        action_type = 0 if action[0] < 0.5 else 1

        # Entry range
        entry_spread = action[1] * current_atr * 0.5
        if action_type == 0:
            entry_min = current_price - entry_spread
            entry_max = current_price
        else:
            entry_min = current_price
            entry_max = current_price + entry_spread

        # Stop loss
        sl_multiplier = action[2] * 3.0 + 1.0
        if action_type == 0:
            stop_loss = entry_min - (current_atr * sl_multiplier)
        else:
            stop_loss = entry_max + (current_atr * sl_multiplier)

        # Take profits
        tp1_mult = action[3] * 3.0 + 1.0
        tp2_mult = action[4] * 5.0 + 2.0
        tp3_mult = action[5] * 7.0 + 3.0

        if action_type == 0:
            tp1 = entry_max + (current_atr * tp1_mult)
            tp2 = entry_max + (current_atr * tp2_mult)
            tp3 = entry_max + (current_atr * tp3_mult)
        else:
            tp1 = entry_min - (current_atr * tp1_mult)
            tp2 = entry_min - (current_atr * tp2_mult)
            tp3 = entry_min - (current_atr * tp3_mult)

        # Leverage
        leverage = int(action[6] * 19) + 1

        # Hold time
        hold_hours = int(action[7] * 99) + 1

        # Risk-reward ratio
        entry_avg = (entry_min + entry_max) / 2
        risk = abs(entry_avg - stop_loss)
        reward = abs(tp2 - entry_avg)
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        return {
            'action': action_type,
            'entry_min': entry_min,
            'entry_max': entry_max,
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'take_profit_3': tp3,
            'leverage': leverage,
            'hold_hours': hold_hours,
            'confidence': 0.7,
            'volatility': current_atr,
            'risk_reward_ratio': risk_reward_ratio
        }

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path)
