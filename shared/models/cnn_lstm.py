import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid for pattern recognition + temporal dependencies
    """

    def __init__(
            self,
            input_dim: int = 10,
            conv_channels: int = 64,
            lstm_hidden: int = 128,
            num_lstm_layers: int = 2,
            output_dim: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            num_lstm_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]

        x, _ = self.lstm(x)
        output = self.fc(x[:, -1, :])
        return output