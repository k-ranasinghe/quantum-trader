import torch.nn as nn


class BiLSTMWithHybridLoss(nn.Module):
    """
    Bi-directional LSTM with hybrid learning structure
    Incorporates both forecasting and reconstruction capabilities
    """

    def __init__(self, d_model: int = 256):
        super().__init__()

        # Bi-directional LSTM
        self.bilstm = nn.LSTM(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Reconstruction head (for hybrid loss)
        self.reconstruction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        lstm_out, _ = self.bilstm(x)

        # Reconstruction for hybrid loss
        reconstructed = self.reconstruction(lstm_out)

        return lstm_out, reconstructed
