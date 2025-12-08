import sys

sys.path.append('..')

from config.constants import CONSTANTS
from config.settings import settings
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
        asset_symbol=CONSTANTS.ASSET,
        model_registry_path=settings.MODEL_REGISTRY_DIR
    )

    # Generate signal
    signal = runtime.generate_signal()

    # Displaysignal
    print(signal)


if __name__ == "__main__":
    main()
