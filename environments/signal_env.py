import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple


class SignalEnv(gym.Env):
    """
    RL environment for complete signal generation

    Action Space: Continuous
        [0] = action type (0-0.5=Long, 0.5-1.0=Short)
        [1] = entry_spread (0-1, normalized)
        [2] = stop_loss_pct (0-1, normalized to ATR multiples)
        [3] = take_profit_1_pct (0-1)
        [4] = take_profit_2_pct (0-1)
        [5] = take_profit_3_pct (0-1)
        [6] = leverage_norm (0-1, scaled to 1-20)
        [7] = hold_time_norm (0-1)
    """

    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        atr: np.ndarray,
        seq_len: int = 60,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_leverage: int = 20
    ):
        super().__init__()

        self.data = data
        self.prices = prices
        self.atr = atr
        self.seq_len = seq_len
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_leverage = max_leverage

        # Observation space: sequence + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, data.shape[1]),
            dtype=np.float32
        )

        # Action space: 8 continuous values
        self.action_space = spaces.Box(
            low=np.array([0.0] * 8),
            high=np.array([1.0] * 8),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.seq_len
        self.balance = self.initial_balance
        self.position = None
        self.total_profit = 0.0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current state"""
        # Historical sequence
        seq = self.data[self.current_step - self.seq_len:self.current_step]
        return seq.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple:
        """Execute action and return result"""

        reward = 0.0
        done = False

        # Decode action
        action_type = 0 if action[0] < 0.5 else 1

        current_price = self.prices[self.current_step]
        current_atr = self.atr[self.current_step]

        # Entry range
        entry_spread = action[1] * current_atr * 0.5

        if action_type == 0:  # Long
            entry_min = current_price - entry_spread
            entry_max = current_price
        else:  # Short
            entry_min = current_price
            entry_max = current_price + entry_spread

        entry_avg = (entry_min + entry_max) / 2

        # Stop loss
        sl_multiplier = action[2] * 3.0 + 1.0  # 1.0 to 4.0 ATR

        if action_type == 0:
            stop_loss = entry_min - (current_atr * sl_multiplier)
        else:
            stop_loss = entry_max + (current_atr * sl_multiplier)

        # Take profits
        tp1_mult = action[3] * 3.0 + 1.0  # 1.0 to 4.0 ATR
        tp2_mult = action[4] * 5.0 + 2.0  # 2.0 to 7.0 ATR
        tp3_mult = action[5] * 7.0 + 3.0  # 3.0 to 10.0 ATR

        if action_type == 0:
            tp1 = entry_max + (current_atr * tp1_mult)
            tp2 = entry_max + (current_atr * tp2_mult)
            tp3 = entry_max + (current_atr * tp3_mult)
        else:
            tp1 = entry_min - (current_atr * tp1_mult)
            tp2 = entry_min - (current_atr * tp2_mult)
            tp3 = entry_min - (current_atr * tp3_mult)

        # Leverage
        leverage = int(action[6] * (self.max_leverage - 1)) + 1

        # Hold time
        hold_time = int(action[7] * 99) + 1

        # Check existing position
        if self.position is not None:
            pos_entry = self.position['entry_avg']
            pos_action = self.position['action']
            pos_sl = self.position['stop_loss']
            pos_tp1 = self.position['tp1']
            pos_tp2 = self.position['tp2']
            pos_tp3 = self.position['tp3']
            pos_leverage = self.position['leverage']
            max_hold = self.position['max_hold_step']

            # Check exit conditions
            should_exit = False
            exit_reason = None

            if pos_action == 0:  # Long
                if current_price >= pos_tp3:
                    should_exit, exit_reason = True, 'tp3'
                elif current_price >= pos_tp2:
                    should_exit, exit_reason = True, 'tp2'
                elif current_price >= pos_tp1:
                    should_exit, exit_reason = True, 'tp1'
                elif current_price <= pos_sl:
                    should_exit, exit_reason = True, 'sl'
            else:  # Short
                if current_price <= pos_tp3:
                    should_exit, exit_reason = True, 'tp3'
                elif current_price <= pos_tp2:
                    should_exit, exit_reason = True, 'tp2'
                elif current_price <= pos_tp1:
                    should_exit, exit_reason = True, 'tp1'
                elif current_price >= pos_sl:
                    should_exit, exit_reason = True, 'sl'

            if self.current_step >= max_hold:
                should_exit, exit_reason = True, 'time'

            if should_exit:
                # Calculate profit
                if pos_action == 0:
                    profit = (current_price - pos_entry) / pos_entry
                else:
                    profit = (pos_entry - current_price) / pos_entry

                # Apply leverage
                profit *= pos_leverage

                # Apply transaction costs
                profit -= 2 * self.transaction_cost

                # Reward based on profit and exit reason
                reward = profit * 100  # Scale reward

                # Bonus for hitting TP targets
                if exit_reason == 'tp3':
                    reward *= 1.5
                elif exit_reason == 'tp2':
                    reward *= 1.3
                elif exit_reason == 'tp1':
                    reward *= 1.1
                elif exit_reason == 'sl':
                    reward *= 0.7

                self.total_profit += profit
                self.trades.append({
                    'profit': profit,
                    'exit_reason': exit_reason,
                    'hold_time': self.current_step - self.position['entry_step']
                })

                self.position = None
        else:
            # Create new position
            self.position = {
                'action': action_type,
                'entry_avg': entry_avg,
                'entry_step': self.current_step,
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'leverage': leverage,
                'max_hold_step': self.current_step + hold_time
            }

            # Small penalty for opening position
            reward = -self.transaction_cost

        # Move to next step
        self.current_step += 1

        if self.current_step >= len(self.prices) - 1:
            done = True

            # Close any open position
            if self.position is not None:
                if self.position['action'] == 0:
                    profit = (current_price - self.position['entry_avg']) / self.position['entry_avg']
                else:
                    profit = (self.position['entry_avg'] - current_price) / self.position['entry_avg']

                profit *= self.position['leverage']
                profit -= self.transaction_cost
                self.total_profit += profit
                reward += profit * 100

        obs = self._get_observation()
        info = {
            'total_profit': self.total_profit,
            'num_trades': len(self.trades)
        }

        return obs, reward, done, False, info
