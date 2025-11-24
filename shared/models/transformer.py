import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer for time-series forecasting
    SOTA architecture for financial prediction
    """

    def __init__(
            self,
            input_dim: int = 10,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            output_dim: int = 1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, src):
        # src: [batch, seq_len, features]
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use last timestep for prediction
        output = self.output_projection(x[:, -1, :])
        return output