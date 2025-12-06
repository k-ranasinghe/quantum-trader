from dataclasses import dataclass
from typing import Dict
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_dim: int = 15
    seq_len: int = 60
    d_model: int = 256
    num_heads: int = 8
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    device: str = "cuda"


@dataclass
class SignalConfig:
    """Signal generation parameters"""
    atr_period: int = 14
    entry_spread_multiplier: float = 0.5
    sl_atr_multiplier: float = 2.0
    tp1_atr_multiplier: float = 2.0
    tp2_atr_multiplier: float = 4.0
    tp3_atr_multiplier: float = 6.0
    max_hold_hours: int = 48
    min_leverage: int = 1
    max_leverage: int = 20


@dataclass
class DataConfig:
    """Data configuration"""
    asset: str = "BTC-USD"
    timeframe: str = "1h"
    lookback_period: str = "5y"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class Signal:
    """Trading Signal"""
    id: str
    asset: str
    action: str  # 'Long' or 'Short'

    # Entry range
    entry_min: float
    entry_max: float

    # Stop loss
    stop_loss: float

    # Multiple take profit levels
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float

    # Risk parameters
    leverage: int
    confidence: float
    models_agreement: float
    risk_reward_ratio: float

    # Market context
    regime: str
    volatility: float

    # Time
    expected_hold_duration: str  # e.g., "4 hours", "2 days"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'asset': self.asset,
            'action': self.action,
            'entry_min': self.entry_min,
            'entry_max': self.entry_max,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'leverage': self.leverage,
            'confidence': self.confidence,
            'models_agreement': self.models_agreement,
            'risk_reward_ratio': self.risk_reward_ratio,
            'regime': self.regime,
            'volatility': self.volatility,
            'expected_hold_duration': self.expected_hold_duration
        }


class ConfigManager:
    """Manage all configurations"""

    def __init__(self, config_path: str = None):
        if config_path and config_path.endswith('.yaml'):
            self.load_from_yaml(config_path)
        else:
            self.model = ModelConfig()
            self.training = TrainingConfig()
            self.signal = SignalConfig()
            self.data = DataConfig()

    def load_from_yaml(self, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        self.model = ModelConfig(**config.get('model', {}))
        self.training = TrainingConfig(**config.get('training', {}))
        self.signal = SignalConfig(**config.get('signal', {}))
        self.data = DataConfig(**config.get('data', {}))

    def save_to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'signal': self.signal.__dict__,
            'data': self.data.__dict__
        }

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'signal': self.signal.__dict__,
            'data': self.data.__dict__
        }
