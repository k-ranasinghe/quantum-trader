import sys

sys.path.append('..')

from config.constants import CONSTANTS
from controllers.ensemble import SignalEnsemble
from generators.signal_generator import SignalGenerator
from services.train_service import TrainService
from utilities.helpers import save_model_artifacts, validate_signal, plot_signal_on_chart


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("Signal Generation System Training")
    print("=" * 70)

    preprocess = TrainService(
        asset=CONSTANTS.ASSET,
        timeframe=CONSTANTS.TIMEFRAME,
        period=CONSTANTS.PERIOD,
        seq_len=CONSTANTS.SEQ_LEN
    )

    # Load data
    df = preprocess.load_and_prepare_data()
    df_original = df.copy()

    # Create labeled dataset
    X, y, scaler = preprocess.create_labeled_dataset()

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess.train_test_split_dict(X, y, val_ratio=0.15, test_ratio=0.15)

    # Train models
    input_dim = X_train.shape[2]

    model1 = SignalGenerator(input_dim=input_dim, seq_len=CONSTANTS.SEQ_LEN)
    model1, _ = preprocess.train_model(model1, X_train, y_train, X_val, y_val, "EnhancedModel1")

    model2 = SignalGenerator(input_dim=input_dim, seq_len=CONSTANTS.SEQ_LEN, d_model=512)
    model2, _ = preprocess.train_model(model2, X_train, y_train, X_val, y_val, "EnhancedModel2")

    # Train RL agent
    rl_agent = preprocess.train_rl_agent()

    # Train regime detector
    regime_detector = preprocess.train_regime_detector()

    # Train meta-model
    models_dict = {'model1': model1, 'model2': model2}
    meta_model = preprocess.train_meta_model(models_dict, X_val, y_val)

    # Create ensemble
    models_dict['rl_agent'] = rl_agent
    ensemble = SignalEnsemble(
        models=models_dict,
        regime_detector=regime_detector,
        meta_model=meta_model,
        asset_name=CONSTANTS.ASSET,
        timeframe=CONSTANTS.TIMEFRAME
    )

    # Evaluate
    test_start = len(X_train) + len(X_val) + CONSTANTS.SEQ_LEN
    df_test = df.iloc[test_start:].copy()

    signals = preprocess.evaluate_ensemble(ensemble, X_test, y_test, df_test)

    # Save models
    print("\nSaving models...")
    save_model_artifacts([(model1, "enhanced_model1"), (model2, "enhanced_model2")],
                         rl_agent, scaler, regime_detector, meta_model, CONSTANTS.SAVE_PATH)

    # Display sample signals
    print("\n" + "=" * 70)
    print("Sample Generated Signals:")
    print("=" * 70)

    def descale_signal(signal):
        close_idx = CONSTANTS.FEATURE_COLS.index('Close')
        price_fields = ['entry_min', 'entry_max', 'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3']
        for field in price_fields:
            setattr(signal, field, getattr(signal, field) * scaler.scale_[close_idx] + scaler.mean_[close_idx])
        return signal

    for i, signal in enumerate(signals[:3]):
        signal = descale_signal(signal)
        validate_signal(signal)
        print(f"\nSignal {i+1}:")
        print(f"  Asset: {signal.asset}")
        print(f"  Action: {signal.action}")
        print(f"  Entry Range: ${signal.entry_min:.2f} - ${signal.entry_max:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit 1: ${signal.take_profit_1:.2f}")
        print(f"  Take Profit 2: ${signal.take_profit_2:.2f}")
        print(f"  Take Profit 3: ${signal.take_profit_3:.2f}")
        print(f"  Leverage: {signal.leverage}x")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Models Agreement: {signal.models_agreement:.2%}")
        print(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}")
        print(f"  Regime: {signal.regime}")
        print(f"  Expected Hold: {signal.expected_hold_duration}")

        plot_signal_on_chart(df_original, signal)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
