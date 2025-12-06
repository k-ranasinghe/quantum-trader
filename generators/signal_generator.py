import torch
import torch.nn as nn

from models.cnn import MultiHeadCNNAttention
from models.lstm import BiLSTMWithHybridLoss


class SignalGenerator(nn.Module):
    """
    Signal generator with multi-task heads for complete signal specification
    """

    def __init__(self, input_dim: int = 15, seq_len: int = 60, d_model: int = 256):
        super().__init__()

        self.seq_len = seq_len

        # Multi-head CNN with attention
        self.feature_extractor = MultiHeadCNNAttention(input_dim, d_model)

        # Bi-directional LSTM
        self.temporal_model = BiLSTMWithHybridLoss(d_model)

        # Task-specific heads
        # 1. Action prediction (Long=0, Short=1)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # 2. Entry range prediction (min, max as % offsets from current)
        self.entry_range_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # [entry_min_offset, entry_max_offset]
        )

        # 3. Stop loss prediction (% offset from current)
        self.stop_loss_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # 4. Multiple take profit levels (% offsets from current)
        self.take_profit_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # [tp1, tp2, tp3]
        )

        # 5. Leverage prediction (1-20)
        self.leverage_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Scale to 0-1, then multiply by 20
        )

        # 6. Hold time prediction (in hours)
        self.hold_time_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        # 7. Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 8. Volatility estimation (ATR)
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        # Multi-scale feature extraction with attention
        features = self.feature_extractor(x)

        # Bi-directional temporal modeling
        temporal_features, reconstructed = self.temporal_model(features)

        # Use last timestep
        final_features = temporal_features[:, -1, :]

        # Multi-task predictions
        action_logits = self.action_head(final_features)
        entry_range = self.entry_range_head(final_features)
        stop_loss = self.stop_loss_head(final_features)
        take_profits = self.take_profit_head(final_features)
        leverage_raw = self.leverage_head(final_features)
        leverage = leverage_raw * 19 + 1  # Scale to 1-20
        hold_time = self.hold_time_head(final_features)
        confidence = self.confidence_head(final_features)
        volatility = self.volatility_head(final_features)

        return {
            'action_logits': action_logits,
            'entry_range': entry_range,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'leverage': leverage,
            'hold_time': hold_time,
            'confidence': confidence,
            'volatility': volatility,
            'reconstructed': reconstructed,
            'original': features
        }

    def load(self, checkpoint_path, device="cpu"):
        """Smart loader that auto-adjusts model architecture to the checkpoint"""
        state = torch.load(checkpoint_path, map_location=device)

        # Detect d_model from checkpoint shapes
        # fusion.weight shape = [d_model, 192]
        fusion_weight_shape = state["feature_extractor.fusion.weight"].shape
        old_d_model = fusion_weight_shape[0]

        if old_d_model != self.feature_extractor.fusion.out_features:
            print(
                f"[INFO] Adjusting model: checkpoint d_model={old_d_model}, current={self.feature_extractor.fusion.out_features}")

            # Rebuild model with correct d_model
            self.__init__(input_dim=15, seq_len=self.seq_len, d_model=old_d_model)

        # Now load weights
        self.load_state_dict(state)
        self.to(device)
        self.eval()

        print(f"[OK] Loaded model with d_model={old_d_model}")
