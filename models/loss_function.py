import torch
import torch.nn as nn
from typing import Dict, Tuple


class TradingLoss(nn.Module):
    """
    Multi-task loss with hybrid learning
    Combines supervised predictions with unsupervised reconstruction
    """

    def __init__(
        self,
        alpha=1.0,      # Action loss weight
        beta=1.0,       # Entry range loss weight
        gamma=1.0,      # Stop loss loss weight
        delta=1.0,      # Take profit loss weight
        epsilon=0.5,    # Leverage loss weight
        zeta=0.5,       # Hold time loss weight
        eta=0.3,        # Confidence loss weight
        theta=0.3,      # Volatility loss weight
        iota=0.5        # Reconstruction loss weight
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.theta = theta
        self.iota = iota

        self.action_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.HuberLoss()
        self.confidence_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate hybrid loss

        Args:
            predictions: Model outputs
            targets: Ground truth values
        """
        # Supervised losses
        loss_action = self.action_loss(
            predictions['action_logits'],
            targets['action']
        )

        loss_entry_range = self.regression_loss(
            predictions['entry_range'],
            targets['entry_range']
        )

        loss_stop_loss = self.regression_loss(
            predictions['stop_loss'],
            targets['stop_loss']
        )

        loss_take_profits = self.regression_loss(
            predictions['take_profits'],
            targets['take_profits']
        )

        loss_leverage = self.regression_loss(
            predictions['leverage'],
            targets['leverage']
        )

        loss_hold_time = self.regression_loss(
            predictions['hold_time'],
            targets['hold_time']
        )

        loss_confidence = self.confidence_loss(
            predictions['confidence'],
            targets['confidence']
        )

        loss_volatility = self.regression_loss(
            predictions['volatility'],
            targets['volatility']
        )

        # Unsupervised reconstruction loss (hybrid learning)
        loss_reconstruction = self.reconstruction_loss(
            predictions['reconstructed'],
            predictions['original']
        )

        # Total weighted loss
        total_loss = (
            self.alpha * loss_action +
            self.beta * loss_entry_range +
            self.gamma * loss_stop_loss +
            self.delta * loss_take_profits +
            self.epsilon * loss_leverage +
            self.zeta * loss_hold_time +
            self.eta * loss_confidence +
            self.theta * loss_volatility +
            self.iota * loss_reconstruction
        )

        loss_dict = {
            'action': loss_action.item(),
            'entry_range': loss_entry_range.item(),
            'stop_loss': loss_stop_loss.item(),
            'take_profits': loss_take_profits.item(),
            'leverage': loss_leverage.item(),
            'hold_time': loss_hold_time.item(),
            'confidence': loss_confidence.item(),
            'volatility': loss_volatility.item(),
            'reconstruction': loss_reconstruction.item()
        }

        return total_loss, loss_dict
