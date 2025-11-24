from datetime import datetime, timedelta
import uuid
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from shared.environments.trading_env import Actions


class SignalGenerator:
    """
    Generate trading signals with comprehensive metadata
    """

    def __init__(
            self,
            confidence_threshold: float = 0.65,
            min_models_agreement: float = 0.6
    ):
        self.confidence_threshold = confidence_threshold
        self.min_models_agreement = min_models_agreement

    def generate_signal(
            self,
            asset: str,
            asset_type: str,
            action: int,
            confidence: float,
            current_price: float,
            metadata: Dict,
            market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Generate trading signal if criteria are met

        Returns None if signal criteria not met
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Check models agreement
        if metadata.get('models_agreement', 0) < self.min_models_agreement:
            return None

        # Skip HOLD signals
        if action == Actions.HOLD:
            return None

        # Calculate signal parameters
        stop_loss, take_profit = self._calculate_targets(
            action,
            current_price,
            market_data
        )

        position_size = self._calculate_position_size(
            confidence,
            current_price,
            market_data
        )

        expected_hold_days = self._estimate_hold_period(
            market_data,
            metadata.get('regime', 1)
        )

        signal = {
            'signal_id': str(uuid.uuid4()),
            'asset': asset,
            'asset_type': asset_type,
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'BUY' if action == Actions.BUY else 'SELL',
            'entry_price': float(current_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'position_size': float(position_size),
            'confidence': float(confidence),
            'models_agreement': float(metadata.get('models_agreement', 0)),
            'regime': self._regime_to_string(metadata.get('regime', 1)),
            'expected_hold_days': int(expected_hold_days),
            'model_versions': metadata.get('individual_predictions', []),
            'features': self._extract_signal_features(market_data)
        }

        return signal

    def _calculate_targets(
            self,
            action: int,
            current_price: float,
            market_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels"""
        # Use ATR for dynamic targets
        atr = market_data['atr'].iloc[-1]
        volatility = market_data['volatility'].iloc[-1]

        if action == Actions.BUY:
            # Long position
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        else:
            # Short position
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)

        return stop_loss, take_profit

    def _calculate_position_size(
            self,
            confidence: float,
            current_price: float,
            market_data: pd.DataFrame
    ) -> float:
        """
        Calculate position size using Kelly Criterion
        Modified for confidence and volatility
        """
        # Base position size (as percentage of capital)
        base_size = 0.05  # 5% of capital

        # Adjust for confidence
        confidence_multiplier = confidence / self.confidence_threshold

        # Adjust for volatility (reduce size in high volatility)
        recent_vol = market_data['volatility'].tail(20).mean()
        vol_percentile = (market_data['volatility'] <= recent_vol).mean()
        vol_multiplier = 1.0 if vol_percentile < 0.7 else 0.5

        position_size = base_size * confidence_multiplier * vol_multiplier

        # Cap at max position size
        return min(position_size, 0.10)  # Max 10% of capital

    def _estimate_hold_period(
            self,
            market_data: pd.DataFrame,
            regime: int
    ) -> int:
        """Estimate expected holding period"""
        # Base on historical trends and regime
        if regime == 0:  # Low volatility
            return 10  # Hold longer
        elif regime == 1:  # Medium volatility
            return 5
        else:  # High volatility
            return 2  # Quick exit

    def _regime_to_string(self, regime: int) -> str:
        """Convert regime number to descriptive string"""
        regimes = {
            0: "low_volatility_trending",
            1: "medium_volatility_mixed",
            2: "high_volatility_choppy"
        }
        return regimes.get(regime, "unknown")

    def _extract_signal_features(self, market_data: pd.DataFrame) -> Dict:
        """Extract relevant features for signal validation"""
        recent = market_data.tail(1).iloc[0]

        return {
            'rsi': float(recent['rsi']),
            'macd': float(recent['macd']),
            'volatility': float(recent['volatility']),
            'volume_ratio': float(recent.get('volume_ratio', 1.0)),
            'trend_strength': float(recent.get('trend_strength', 0.0))
        }