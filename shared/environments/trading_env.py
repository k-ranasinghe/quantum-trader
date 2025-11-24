import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import IntEnum


class Actions(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2


class MultiAssetTradingEnv(gym.Env):
    """
    Production-grade multi-asset trading environment
    Compatible with FinRL and Stable-Baselines3
    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            df: pd.DataFrame,
            asset_type: str = "stock",
            initial_capital: float = 100000.0,
            transaction_cost_pct: float = 0.001,
            slippage_pct: float = 0.0005,
            max_position_pct: float = 0.2,
            risk_free_rate: float = 0.02
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.asset_type = asset_type
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.risk_free_rate = risk_free_rate

        # State space: [cash, shares, price, features...]
        self.n_features = len(df.columns) - 1  # Exclude 'close'
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 3,),  # cash, shares, price + features
            dtype=np.float32
        )

        # The action could range from -1 (SELL) to 1 (BUY), with 0 being HOLD
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Episode tracking
        self.current_step = 0
        self.max_steps = len(df) - 1

        # Portfolio state
        self.cash = initial_capital
        self.shares = 0.0
        self.portfolio_value = initial_capital

        # Performance tracking
        self.portfolio_history = []
        self.trades = []
        self.returns = []

        # Risk metrics
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.portfolio_value = self.initial_capital

        self.portfolio_history = [self.initial_capital]
        self.trades = []
        self.returns = []

        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        current_price = self.df.iloc[self.current_step]['close']
        features = self.df.iloc[self.current_step].drop('close').values

        # Normalize cash and shares
        cash_normalized = self.cash / self.initial_capital
        shares_normalized = (self.shares * current_price) / self.initial_capital
        price_normalized = current_price / self.df['close'].max()

        obs = np.array([
            cash_normalized,
            shares_normalized,
            price_normalized,
            *features
        ], dtype=np.float32)

        # Check for NaN values in the observation
        if np.isnan(obs).any():
            print(f"Warning: NaN in observation: {obs}")

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        current_price = self.df.iloc[self.current_step]['close']

        # Action is a continuous value between -1 (SELL) and 1 (BUY)
        action_value = action[0]  # Scalar action

        # Initialize variables
        reward = 0.0
        done = False

        if action_value < 0:  # SELL
            # Action value represents a fraction of the current position to sell
            sell_amount = abs(action_value) * self.shares  # how much to sell
            if sell_amount > self.shares:
                sell_amount = self.shares  # Ensure we don't sell more than we own

            # Perform the selling operation
            self.shares -= sell_amount
            self.cash += sell_amount * current_price  # Add cash from the sale
            reward = -sell_amount * current_price  # Negative reward for selling (a cost of selling)

        elif action_value > 0:  # BUY
            # Action value represents the fraction of available cash to use for buying
            buy_amount = action_value * self.cash / current_price  # amount to buy

            if buy_amount * current_price > self.cash:
                buy_amount = self.cash / current_price  # Ensure we don't buy more than available cash

            # Perform the buying operation
            self.shares += buy_amount
            self.cash -= buy_amount * current_price  # Spend cash for the buy
            reward = buy_amount * current_price  # Positive reward for buying (profit from asset price increase)

        else:  # HOLD
            reward = 0  # No reward for holding

        # Update the portfolio stats (e.g., equity, portfolio value)
        self.portfolio_value = self.shares * current_price + self.cash

        # Check if the episode is done (end of trading day or max steps reached)
        if self.current_step >= len(self.df) - 1:
            done = True

        # Move to the next step in the data
        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True  # End the episode if we have processed all the data

        # Get next observation
        obs = self._get_observation() if not done else self._get_observation()

        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.max_drawdown
        }

        return obs, reward, done, False, info

    def _calculate_reward(self, step_return: float) -> float:
        """
        Calculate risk-adjusted reward
        Incorporates CVaR and Sharpe-like metrics
        """
        # Base reward: portfolio return
        reward = step_return

        # Penalize drawdown
        reward -= self.max_drawdown * 0.1

        # Penalize excessive risk (if we have enough history)
        if len(self.returns) > 20:
            recent_returns = np.array(self.returns[-20:])
            volatility = np.std(recent_returns)

            # CVaR penalty (5% worst returns)
            negative_returns = recent_returns[recent_returns < 0]
            if len(negative_returns) > 0:
                cvar = np.percentile(negative_returns, 5)
                reward += cvar * 0.5  # Penalize (cvar is negative)

            # Sharpe-like bonus
            if volatility > 0:
                sharpe = (np.mean(recent_returns) - self.risk_free_rate / 252) / volatility
                reward += sharpe * 0.01

        # Penalize holding cash for too long
        cash_ratio = self.cash / self.portfolio_value
        if cash_ratio > 0.8:
            reward -= 0.001

        return reward

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Shares: {self.shares:.2f}")
            print(f"Total Return: {(self.portfolio_value - self.initial_capital) / self.initial_capital * 100:.2f}%")
            print(f"Max Drawdown: {self.max_drawdown * 100:.2f}%")
            print("-" * 50)

    def get_portfolio_stats(self) -> Dict:
        """Calculate comprehensive portfolio statistics"""
        returns_array = np.array(self.returns)

        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        # Annualized metrics (assuming daily data)
        trading_days = 252

        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        if len(returns_array) > 0:
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return > 0:
                sharpe_ratio = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(trading_days)

            # Sortino (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino_ratio = (mean_return - self.risk_free_rate / 252) / downside_std * np.sqrt(trading_days)

        # Win rate
        winning_trades = sum(1 for r in returns_array if r > 0)
        win_rate = winning_trades / len(returns_array) if len(returns_array) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_portfolio_value': self.portfolio_value
        }