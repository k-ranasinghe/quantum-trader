import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import torch
import os
import pickle


def validate_signal(signal) -> bool:
    """
    Validate signal parameters

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check entry range
        if signal.entry_min >= signal.entry_max:
            print("Error: entry_min must be < entry_max")
            return False

        # Check directional logic
        if signal.action == "Long":
            if signal.stop_loss >= signal.entry_min:
                print("Error: For Long, stop_loss must be < entry_min")
                return False

            if not (signal.take_profit_1 > signal.entry_max and
                    signal.take_profit_2 > signal.take_profit_1 and
                    signal.take_profit_3 > signal.take_profit_2):
                print("Error: For Long, TP1 < TP2 < TP3 and all > entry_max")
                return False

        elif signal.action == "Short":
            if signal.stop_loss <= signal.entry_max:
                print("Error: For Short, stop_loss must be > entry_max")
                return False

            if not (signal.take_profit_1 < signal.entry_min and
                    signal.take_profit_2 < signal.take_profit_1 and
                    signal.take_profit_3 < signal.take_profit_2):
                print("Error: For Short, TP1 > TP2 > TP3 and all < entry_min")
                return False

        # Check leverage
        if not (1 <= signal.leverage <= 20):
            print(f"Error: leverage must be 1-20, got {signal.leverage}")
            return False

        # Check confidence
        if not (0 <= signal.confidence <= 1):
            print(f"Error: confidence must be 0-1, got {signal.confidence}")
            return False

        # Check risk-reward ratio
        if signal.risk_reward_ratio < 0.5:
            print(f"Warning: Low risk-reward ratio: {signal.risk_reward_ratio:.2f}")

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


def plot_signal_on_chart(df: pd.DataFrame, signal, save_path: str = None):
    """
    Plot signal on price chart

    Args:
        df: DataFrame with OHLC data
        signal: Signal object
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Price chart
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)

    # Entry range
    ax1.axhline(y=signal.entry_min, color='blue', linestyle='--',
                label='Entry Min', alpha=0.7)
    ax1.axhline(y=signal.entry_max, color='blue', linestyle='--',
                label='Entry Max', alpha=0.7)
    ax1.fill_between(df.index, signal.entry_min, signal.entry_max,
                     alpha=0.2, color='blue')

    # Stop loss
    ax1.axhline(y=signal.stop_loss, color='red', linestyle='-',
                label='Stop Loss', linewidth=2)

    # Take profits
    ax1.axhline(y=signal.take_profit_1, color='green', linestyle=':',
                label='TP1', alpha=0.7)
    ax1.axhline(y=signal.take_profit_2, color='green', linestyle='--',
                label='TP2', alpha=0.8)
    ax1.axhline(y=signal.take_profit_3, color='green', linestyle='-',
                label='TP3', linewidth=2)

    ax1.set_title(f"{signal.asset} - {signal.action} Signal", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Volume
    ax2.bar(df.index, df['Volume'], alpha=0.5, color='gray')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")

    plt.show()


def calculate_position_size(
        signal,
        account_balance: float,
        risk_per_trade: float = 0.02
) -> Dict:
    """
    Calculate position size based on signal and risk parameters

    Args:
        signal: Signal object
        account_balance: Total account balance
        risk_per_trade: Percentage of account to risk (default 2%)

    Returns:
        Dictionary with position sizing details
    """
    entry_avg = (signal.entry_min + signal.entry_max) / 2
    risk_per_unit = abs(entry_avg - signal.stop_loss)

    # Maximum risk amount
    max_risk_amount = account_balance * risk_per_trade

    # Position size without leverage
    position_size_base = max_risk_amount / risk_per_unit

    # Position value
    position_value = position_size_base * entry_avg

    # With leverage
    required_margin = position_value / signal.leverage

    # Potential profits
    tp1_profit = position_size_base * abs(signal.take_profit_1 - entry_avg) * signal.leverage
    tp2_profit = position_size_base * abs(signal.take_profit_2 - entry_avg) * signal.leverage
    tp3_profit = position_size_base * abs(signal.take_profit_3 - entry_avg) * signal.leverage

    # Potential loss
    max_loss = max_risk_amount * signal.leverage

    return {
        'position_size': position_size_base,
        'position_value': position_value,
        'required_margin': required_margin,
        'leverage': signal.leverage,
        'max_loss': max_loss,
        'tp1_profit': tp1_profit,
        'tp2_profit': tp2_profit,
        'tp3_profit': tp3_profit,
        'risk_amount': max_risk_amount,
        'entry_avg': entry_avg
    }


def backtest_signals(
        df: pd.DataFrame,
        signals: List,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001
) -> Dict:
    """
    Simple backtest of generated signals

    Returns:
        Dictionary with backtest results
    """
    balance = initial_balance
    trades = []

    for signal in signals:
        # Find entry and exit in data
        try:
            # Entry
            entry_time = pd.to_datetime(signal.id.split('_')[1] + '_' + signal.id.split('_')[2],
                                        format='%Y%m%d_%H%M%S')
            entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]

            entry_price = (signal.entry_min + signal.entry_max) / 2

            # Find exit
            hold_hours = int(signal.expected_hold_duration.split()[0])
            exit_idx = min(entry_idx + hold_hours, len(df) - 1)

            # Simulate trade
            for i in range(entry_idx, exit_idx + 1):
                current_price = df['Close'].iloc[i]

                # Check exit conditions
                if signal.action == "Long":
                    if current_price >= signal.take_profit_3:
                        exit_price = signal.take_profit_3
                        exit_reason = 'TP3'
                        break
                    elif current_price >= signal.take_profit_2:
                        exit_price = signal.take_profit_2
                        exit_reason = 'TP2'
                        break
                    elif current_price >= signal.take_profit_1:
                        exit_price = signal.take_profit_1
                        exit_reason = 'TP1'
                        break
                    elif current_price <= signal.stop_loss:
                        exit_price = signal.stop_loss
                        exit_reason = 'SL'
                        break
                else:  # Short
                    if current_price <= signal.take_profit_3:
                        exit_price = signal.take_profit_3
                        exit_reason = 'TP3'
                        break
                    elif current_price <= signal.take_profit_2:
                        exit_price = signal.take_profit_2
                        exit_reason = 'TP2'
                        break
                    elif current_price <= signal.take_profit_1:
                        exit_price = signal.take_profit_1
                        exit_reason = 'TP1'
                        break
                    elif current_price >= signal.stop_loss:
                        exit_price = signal.stop_loss
                        exit_reason = 'SL'
                        break
            else:
                exit_price = df['Close'].iloc[exit_idx]
                exit_reason = 'Time'

            # Calculate P&L
            if signal.action == "Long":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            # Apply leverage and costs
            pnl_pct = (pnl_pct * signal.leverage) - (2 * transaction_cost)

            # Update balance
            balance *= (1 + pnl_pct)

            trades.append({
                'signal_id': signal.id,
                'action': signal.action,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'balance': balance
            })

        except Exception as e:
            print(f"Error backtesting signal {signal.id}: {e}")
            continue

    # Calculate metrics
    if trades:
        total_return = (balance - initial_balance) / initial_balance
        num_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0

        pnls = [t['pnl_pct'] for t in trades]
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0

        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': num_trades - winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades
        }
    else:
        results = {
            'error': 'No trades executed'
        }

    return results


def print_backtest_results(results: Dict):
    """Print backtest results in formatted output"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Initial Balance:  ${results['initial_balance']:,.2f}")
    print(f"Final Balance:    ${results['final_balance']:,.2f}")
    print(f"Total Return:     {results['total_return']:.2%}")
    print("-" * 70)
    print(f"Total Trades:     {results['num_trades']}")
    print(f"Winning Trades:   {results['winning_trades']}")
    print(f"Losing Trades:    {results['losing_trades']}")
    print(f"Win Rate:         {results['win_rate']:.2%}")
    print("-" * 70)
    print(f"Average Win:      {results['avg_win']:.2%}")
    print(f"Average Loss:     {results['avg_loss']:.2%}")
    print("=" * 70)


def save_model_artifacts(
        models,
        rl_agent,
        scaler,
        regime_detector,
        meta_model,
        save_dir
):
    """Save all model artifacts"""

    os.makedirs(save_dir, exist_ok=True)

    # Save PyTorch models
    for model, model_name in models:
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), f"{save_dir}/{model_name}.pth")

    # Save RL agent
    rl_agent.save(f"{save_dir}/enhanced_rl_agent")

    # Save scaler
    with open(f"{save_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Save XGBoost models
    regime_detector.save_model(f"{save_dir}/regime_detector.json")
    meta_model.save_model(f"{save_dir}/meta_model.json")

    print(f"All artifacts saved to {save_dir}/")
