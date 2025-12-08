import torch
from datetime import timedelta

from config.settings import settings


class CONSTANTS:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ASSET = "AAPL"
    SEQ_LEN = 60
    INPUT_DIM = 15
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 1e-3
    TIMEFRAME = "1h"
    PERIOD = "1y"

    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility', 'rsi',
        'ma20', 'ma60', 'price_ratio', 'vol_spike', 'bb_upper', 'bb_lower', 'bb_position'
    ]

    INTERVAL_MAX_PERIOD = {
        "1m":   timedelta(days=7),
        "2m":   timedelta(days=60),
        "5m":   timedelta(days=60),
        "15m":  timedelta(days=60),
        "30m":  timedelta(days=60),
        "90m":  timedelta(days=60),
        "1h":   timedelta(days=730),
        # daily and above have no real limit → use 10 years as safe default
        "1d":   timedelta(days=365*10),
        "5d":   timedelta(days=365*10),
        "1wk":  timedelta(days=365*10),
        "1mo":  timedelta(days=365*10),
        "3mo":  timedelta(days=365*10),
    }

    SAVE_PATH = settings.MODEL_REGISTRY_DIR
