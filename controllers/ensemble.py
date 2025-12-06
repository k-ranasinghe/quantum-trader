import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import xgboost as xgb

from config.models import Signal


class SignalEnsemble:
    """
    Ensemble controller for complete signal generation
    """

    def __init__(
        self,
        models: Dict,
        regime_detector: xgb.XGBClassifier,
        meta_model: xgb.XGBClassifier,
        asset_name: str = "BTC-USD",
        timeframe: str = "1h"
    ):
        self.models = models
        self.regime_detector = regime_detector
        self.meta_model = meta_model
        self.asset_name = asset_name
        self.timeframe = timeframe

        # Model performance tracking
        self.model_weights = {name: 1.0 for name in models.keys()}

        # Timeframe to readable format
        self.timeframe_readable = {
            '1m': 'minutes',
            '5m': 'minutes',
            '15m': 'minutes',
            '1h': 'hours',
            '4h': 'hours',
            '1d': 'days'
        }

    def generate_signal(
        self,
        observation: np.ndarray,
        current_price: float,
        current_atr: float,
        market_data: pd.DataFrame,
        signal_id: str = None
    ) -> Signal:
        """
        Generate trading signal
        """
        # 1. Detect market regime
        regime = self._detect_regime(market_data)

        # 2. Get predictions from all models
        predictions = self._get_model_predictions(
            observation, current_price, current_atr
        )

        # 3. Calculate models agreement
        actions = [p['action'] for p in predictions]
        most_common_action = max(set(actions), key=actions.count)
        models_agreement = actions.count(most_common_action) / len(actions)

        # 4. Weighted voting for action
        final_action = self._weighted_vote_action(predictions)

        # 5. Aggregate signal parameters
        entry_min, entry_max = self._aggregate_entry_range(predictions, final_action)
        stop_loss = self._aggregate_stop_loss(predictions, final_action)
        tp1, tp2, tp3 = self._aggregate_take_profits(predictions, final_action)
        leverage = self._aggregate_leverage(predictions)
        hold_hours = self._aggregate_hold_time(predictions)

        # 6. Calculate confidence using meta-model
        confidence = self._calculate_confidence(predictions, regime)

        # 7. Calculate risk-reward ratio
        entry_avg = (entry_min + entry_max) / 2
        risk = abs(entry_avg - stop_loss)
        reward = abs(tp2 - entry_avg)  # Use TP2 as target
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        # 8. Apply regime-based adjustments
        (confidence, entry_min, entry_max, stop_loss,
         tp1, tp2, tp3, leverage) = self._apply_regime_adjustments(
            confidence, entry_min, entry_max, stop_loss,
            tp1, tp2, tp3, leverage, regime, final_action
        )

        # 9. Format hold duration
        hold_duration = self._format_hold_duration(hold_hours)

        # 10. Create signal
        if signal_id is None:
            signal_id = f"{self.asset_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        action_str = "Long" if final_action == 0 else "Short"
        regime_str = self._regime_to_string(regime)

        signal = Signal(
            id=signal_id,
            asset=self.asset_name,
            action=action_str,
            entry_min=entry_min,
            entry_max=entry_max,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            leverage=leverage,
            confidence=confidence,
            models_agreement=models_agreement,
            risk_reward_ratio=risk_reward_ratio,
            regime=regime_str,
            volatility=current_atr,
            expected_hold_duration=hold_duration
        )

        return signal

    def _get_model_predictions(
        self,
        observation: np.ndarray,
        current_price: float,
        current_atr: float
    ) -> List[Dict]:
        """Get predictions from all models"""
        predictions = []

        for model_name, model in self.models.items():
            try:
                pred = self._get_single_model_prediction(
                    model, model_name, observation, current_price, current_atr
                )
                predictions.append(pred)
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                continue

        return predictions

    def _get_single_model_prediction(
        self,
        model,
        model_name: str,
        observation: np.ndarray,
        current_price: float,
        current_atr: float
    ) -> Dict:
        """Get prediction from a single model"""
        if isinstance(model, torch.nn.Module):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                output = model(obs_tensor)

                # Extract predictions
                action_probs = torch.softmax(output['action_logits'], dim=1)
                action = torch.argmax(action_probs, dim=1).item()

                # Get parameters
                entry_range = output['entry_range'][0].cpu().numpy()
                stop_loss_offset = output['stop_loss'][0].item()
                take_profits = output['take_profits'][0].cpu().numpy()
                leverage = output['leverage'][0].item()
                hold_hours = output['hold_time'][0].item()
                confidence = output['confidence'][0].item()
                volatility = output['volatility'][0].item()

                # Convert offsets to prices
                entry_min = current_price * (1 + entry_range[0])
                entry_max = current_price * (1 + entry_range[1])
                stop_loss = current_price * (1 + stop_loss_offset)
                tp1 = current_price * (1 + take_profits[0])
                tp2 = current_price * (1 + take_profits[1])
                tp3 = current_price * (1 + take_profits[2])

                return {
                    'model': model_name,
                    'action': action,
                    'entry_min': entry_min,
                    'entry_max': entry_max,
                    'stop_loss': stop_loss,
                    'take_profit_1': tp1,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'leverage': int(leverage),
                    'hold_hours': hold_hours,
                    'confidence': confidence,
                    'volatility': volatility
                }

        elif hasattr(model, 'predict_signal'):
            # RL agent
            signal_dict = model.predict_signal(observation, current_price, current_atr)
            signal_dict['model'] = model_name
            return signal_dict

        else:
            raise ValueError(f"Unknown model type: {type(model)}")

    def _weighted_vote_action(self, predictions: List[Dict]) -> int:
        """Weighted voting for action"""
        action_scores = {0: 0.0, 1: 0.0}

        for pred in predictions:
            weight = self.model_weights[pred['model']] * pred['confidence']
            action_scores[pred['action']] += weight

        return max(action_scores, key=action_scores.get)

    def _aggregate_entry_range(
        self,
        predictions: List[Dict],
        final_action: int
    ) -> tuple:
        """Aggregate entry range predictions"""
        matching_preds = [p for p in predictions if p['action'] == final_action]

        if not matching_preds:
            matching_preds = predictions

        total_weight = sum(
            self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        )

        entry_min = sum(
            p['entry_min'] * self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        ) / total_weight

        entry_max = sum(
            p['entry_max'] * self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        ) / total_weight

        return entry_min, entry_max

    def _aggregate_stop_loss(
        self,
        predictions: List[Dict],
        final_action: int
    ) -> float:
        """Aggregate stop loss predictions"""
        matching_preds = [p for p in predictions if p['action'] == final_action]

        if not matching_preds:
            matching_preds = predictions

        total_weight = sum(
            self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        )

        stop_loss = sum(
            p['stop_loss'] * self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        ) / total_weight

        return stop_loss

    def _aggregate_take_profits(
        self,
        predictions: List[Dict],
        final_action: int
    ) -> tuple:
        """Aggregate take profit predictions"""
        matching_preds = [p for p in predictions if p['action'] == final_action]

        if not matching_preds:
            matching_preds = predictions

        total_weight = sum(
            self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        )

        tp1 = sum(
            p['take_profit_1'] * self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        ) / total_weight

        tp2 = sum(
            p['take_profit_2'] * self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        ) / total_weight

        tp3 = sum(
            p['take_profit_3'] * self.model_weights[p['model']] * p['confidence']
            for p in matching_preds
        ) / total_weight

        return tp1, tp2, tp3

    def _aggregate_leverage(self, predictions: List[Dict]) -> int:
        """Aggregate leverage predictions"""
        leverages = [p['leverage'] for p in predictions]
        weights = [self.model_weights[p['model']] * p['confidence'] for p in predictions]

        weighted_leverage = sum(l * w for l, w in zip(leverages, weights)) / sum(weights)
        return int(np.clip(weighted_leverage, 1, 20))

    def _aggregate_hold_time(self, predictions: List[Dict]) -> float:
        """Aggregate hold time predictions"""
        hold_times = [p['hold_hours'] for p in predictions]
        return float(np.median(hold_times))

    def _detect_regime(self, market_data: pd.DataFrame) -> int:
        """Detect market regime"""
        recent_data = market_data.tail(100)

        if len(recent_data) < 100:
            return 1  # Normal

        returns = recent_data['returns'].values
        volatility = recent_data['volatility'].values

        features = np.array([[
            np.mean(returns),
            np.std(returns),
            np.mean(volatility),
            np.max(volatility),
            np.percentile(returns, 25),
            np.percentile(returns, 75)
        ]])

        regime = self.regime_detector.predict(features)[0]
        return regime

    def _calculate_confidence(
        self,
        predictions: List[Dict],
        regime: int
    ) -> float:
        """Calculate confidence using meta-model"""
        confidences = [p['confidence'] for p in predictions]
        actions = [p['action'] for p in predictions]

        most_common = max(set(actions), key=actions.count)
        agreement = actions.count(most_common) / len(actions)

        avg_conf = np.mean(confidences)
        var_conf = np.var(confidences)

        regime_features = np.zeros(3)
        regime_features[regime] = 1.0

        meta_features = np.array([[
            agreement,
            avg_conf,
            var_conf,
            *regime_features
        ]])

        try:
            confidence = self.meta_model.predict_proba(meta_features)[0][1]
        except:
            confidence = avg_conf

        return float(confidence)

    def _apply_regime_adjustments(
        self,
        confidence: float,
        entry_min: float,
        entry_max: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float,
        leverage: int,
        regime: int,
        action: int
    ) -> tuple:
        """Apply regime-specific adjustments"""

        # High volatility - wider ranges, lower leverage
        if regime == 2:
            confidence *= 0.8
            leverage = max(1, int(leverage * 0.7))

            # Widen all ranges by 20%
            entry_spread = entry_max - entry_min
            entry_mid = (entry_min + entry_max) / 2
            entry_min = entry_mid - entry_spread * 0.6
            entry_max = entry_mid + entry_spread * 0.6

            if action == 0:  # Long
                stop_loss -= abs(stop_loss - entry_min) * 0.2
                tp1 += abs(tp1 - entry_max) * 0.2
                tp2 += abs(tp2 - entry_max) * 0.2
                tp3 += abs(tp3 - entry_max) * 0.2
            else:  # Short
                stop_loss += abs(stop_loss - entry_max) * 0.2
                tp1 -= abs(entry_min - tp1) * 0.2
                tp2 -= abs(entry_min - tp2) * 0.2
                tp3 -= abs(entry_min - tp3) * 0.2

        # Low volatility - tighter ranges, higher leverage
        elif regime == 0:
            confidence *= 1.1
            confidence = min(confidence, 1.0)
            leverage = min(20, int(leverage * 1.2))

            # Tighten ranges by 20%
            entry_spread = entry_max - entry_min
            entry_mid = (entry_min + entry_max) / 2
            entry_min = entry_mid - entry_spread * 0.4
            entry_max = entry_mid + entry_spread * 0.4

            if action == 0:
                stop_loss += abs(stop_loss - entry_min) * 0.2
                tp1 -= abs(tp1 - entry_max) * 0.2
                tp2 -= abs(tp2 - entry_max) * 0.2
                tp3 -= abs(tp3 - entry_max) * 0.2
            else:
                stop_loss -= abs(stop_loss - entry_max) * 0.2
                tp1 += abs(entry_min - tp1) * 0.2
                tp2 += abs(entry_min - tp2) * 0.2
                tp3 += abs(entry_min - tp3) * 0.2

        return confidence, entry_min, entry_max, stop_loss, tp1, tp2, tp3, leverage

    def _format_hold_duration(self, hours: float) -> str:
        """Format hold duration as readable string"""
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} minutes"
        elif hours < 24:
            return f"{int(hours)} hours"
        else:
            days = hours / 24
            return f"{days:.1f} days"

    def _regime_to_string(self, regime: int) -> str:
        """Convert regime code to string"""
        regime_map = {
            0: "Low Volatility",
            1: "Normal",
            2: "High Volatility"
        }
        return regime_map.get(regime, "Unknown")

    def update_model_weights(self, model_name: str, performance: float):
        """Update model weights based on performance"""
        self.model_weights[model_name] = (
            0.9 * self.model_weights[model_name] +
            0.1 * performance
        )
