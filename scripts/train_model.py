import sys

sys.path.append('..')

from config.constants import CONSTANTS
from services.train_service import TrainService


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("Signal Generation System Training")
    print("=" * 70)

    # Initialize services
    preprocess = TrainService(
        asset=CONSTANTS.ASSET,
        timeframe=CONSTANTS.TIMEFRAME,
        period=CONSTANTS.PERIOD,
        seq_len=CONSTANTS.SEQ_LEN
    )

    # Run full training pipeline
    preprocess.run_full_training_pipeline()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
