from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BacktestEngine:
    """
    Comprehensive backtesting engine with walk-forward analysis
    """

    def __init__(
            self,
            env_class,
            initial_capital: float = 100000.0,
            transaction_cost_pct: float = 0.001,
            slippage_pct: float = 0.0005
    ):
        self.env_class = env_class
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

    def run_backtest(
            self,
            model,
            data: pd.DataFrame,
            test_period: Tuple[str, str]
    ) -> Dict:
        """
        Run backtest on specified period
        """
        # Filter data for test period
        test_data = data[
            (data.index >= test_period[0]) &
            (data.index <= test_period[1])
            ]

        # Create environment
        env = self.env_class(
            df=test_data,
            initial_capital=self.initial_capital,
            transaction_cost_pct=self.transaction_cost_pct,
            slippage_pct=self.slippage_pct
        )

        # Run episode
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        # Get results
        stats = env.get_portfolio_stats()

        return {
            'period': test_period,
            'stats': stats,
            'portfolio_history': env.portfolio_history,
            'trades': env.trades,
            'returns': env.returns
        }

    def walk_forward_analysis(
            self,
            model,
            data: pd.DataFrame,
            train_window: int = 252,  # 1 year
            test_window: int = 63,  # 3 months
            step: int = 21  # 1 month
    ) -> List[Dict]:
        """
        Perform walk-forward analysis
        """
        results = []

        data_len = len(data)
        start_idx = train_window

        while start_idx + test_window < data_len:
            # Define train and test periods
            train_start = start_idx - train_window
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + test_window, data_len)

            train_period = (
                data.index[train_start].strftime('%Y-%m-%d'),
                data.index[train_end].strftime('%Y-%m-%d')
            )
            test_period = (
                data.index[test_start].strftime('%Y-%m-%d'),
                data.index[test_end].strftime('%Y-%m-%d')
            )

            # Run backtest
            result = self.run_backtest(model, data, test_period)
            result['train_period'] = train_period
            results.append(result)

            # Move window
            start_idx += step

        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all walk-forward periods"""
        all_returns = []
        all_sharpe = []
        all_drawdown = []

        for result in results:
            all_returns.extend(result['returns'])
            all_sharpe.append(result['stats']['sharpe_ratio'])
            all_drawdown.append(result['stats']['max_drawdown'])

        returns_array = np.array(all_returns)

        metrics = {
            'total_return': np.prod(1 + returns_array) - 1,
            'mean_sharpe': np.mean(all_sharpe),
            'median_sharpe': np.median(all_sharpe),
            'mean_drawdown': np.mean(all_drawdown),
            'max_drawdown': np.max(all_drawdown),
            'win_rate': (returns_array > 0).sum() / len(returns_array),
            'calmar_ratio': np.mean(returns_array) * 252 / np.max(all_drawdown) if np.max(all_drawdown) > 0 else 0
        }

        return metrics

    def plot_results(self, results: List[Dict], save_path: Optional[str] = None):
        """Generate comprehensive backtest visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Portfolio value over time
        all_portfolio_values = []
        for result in results:
            all_portfolio_values.extend(result['portfolio_history'])

        axes[0, 0].plot(all_portfolio_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)

        # Returns distribution
        all_returns = []
        for result in results:
            all_returns.extend(result['returns'])

        axes[0, 1].hist(all_returns, bins=50, edgecolor='black')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')

        # Sharpe ratio over time
        sharpe_ratios = [r['stats']['sharpe_ratio'] for r in results]
        axes[1, 0].plot(sharpe_ratios, marker='o')
        axes[1, 0].set_title('Sharpe Ratio Over Walk-Forward Periods')
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True)

        # Drawdown
        drawdowns = [r['stats']['max_drawdown'] * 100 for r in results]
        axes[1, 1].plot(drawdowns, marker='o', color='red')
        axes[1, 1].set_title('Maximum Drawdown Over Periods')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].grid(True)

        # Win rate
        win_rates = [r['stats']['win_rate'] * 100 for r in results]
        axes[2, 0].bar(range(len(win_rates)), win_rates)
        axes[2, 0].set_title('Win Rate by Period')
        axes[2, 0].set_xlabel('Period')
        axes[2, 0].set_ylabel('Win Rate (%)')
        axes[2, 0].axhline(y=50, color='r', linestyle='--')

        # Cumulative returns
        cumulative_returns = np.cumprod(1 + np.array(all_returns)) - 1
        axes[2, 1].plot(cumulative_returns * 100)
        axes[2, 1].set_title('Cumulative Returns')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('Cumulative Return (%)')
        axes[2, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()