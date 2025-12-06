import sys

sys.path.append('..')

from config.constants import CONSTANTS
from services.inference_service import InferenceService


def main():
    """Main runtime execution"""
    print("=" * 70)
    print("Signal Generation Runtime")
    print("=" * 70)

    # Model paths
    model_paths = {
        'model1': f'{CONSTANTS.SAVE_PATH}/enhanced_model1.pth',
        'model2': f'{CONSTANTS.SAVE_PATH}/enhanced_model2.pth',
        'rl_agent': f'{CONSTANTS.SAVE_PATH}/enhanced_rl_agent'
    }

    # Initialize runtime
    runtime = InferenceService(
        model_paths=model_paths,
        asset=CONSTANTS.ASSET,
        timeframe=CONSTANTS.TIMEFRAME,
    )

    # Generate and display signal
    signal = runtime.generate_and_display_signal()

    # Save signal
    runtime.save_signal(signal, CONSTANTS.SAVE_PATH)


if __name__ == "__main__":
    main()
