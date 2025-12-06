import torch
import torch.nn as nn


class MultiHeadCNNAttention(nn.Module):
    """
    Multi-head CNN with self-attention for different temporal scales
    Based on SOTA architectures for algorithmic trading
    """

    def __init__(self, input_dim: int = 15, d_model: int = 256):
        super().__init__()

        # Three specialized CNN heads for different timeframes
        self.minute_head = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.hourly_head = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.daily_head = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Feature fusion
        self.fusion = nn.Linear(192, d_model)

        # Self-attention
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x_conv = x.permute(0, 2, 1)  # [batch, features, seq_len]

        # Multi-scale feature extraction
        minute_feat = self.minute_head(x_conv)
        hourly_feat = self.hourly_head(x_conv)
        daily_feat = self.daily_head(x_conv)

        # Concatenate and reshape
        combined = torch.cat([minute_feat, hourly_feat, daily_feat], dim=1)
        combined = combined.permute(0, 2, 1)  # [batch, seq_len/2, 192]

        # Fusion
        fused = self.fusion(combined)

        # Self-attention
        attended, _ = self.attention(fused, fused, fused)

        return attended
